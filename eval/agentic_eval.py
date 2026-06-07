#!/usr/bin/env python3
"""agentic_eval.py — head-to-head: ONE-SHOT vs AGENTIC verifier-in-the-loop on the SAME
held-out task set, for the EXISTING Ollama model (default ecdsa-coder-1.5b-sft).

This is the PROOF that the verifier-in-the-loop harness multiplies capability WITHOUT
retraining. Both arms run on an identical task set drawn from:
  * unseen   : in-distribution families at eval-only RNG seeds (disjoint from training seeds),
               sampled across curriculum bands B0..B6.
  * heldout  : the generalization families never used in training (fused multiply-accumulate,
               arbitrary n=6 S-box).

Arms:
  ONE-SHOT : a single completion (n=1), greedy-ish temperature, verify once. This mirrors the
             existing eval_proxy.py one-shot protocol.
  AGENTIC  : agentic_solve(n_samples=8, max_turns=4) — best-of-N seed then iterative repair
             from the verifier's exact reason+witness.

Reported per arm:
  valid_rate        fraction of tasks with a VALID circuit (all 4 gates + correct function)
  win_rate          fraction VALID and strictly cheaper than the reference circuit
  mean_vs_ref       mean (cost / reference_cost) over valid solutions
  mean_turns        (agentic) mean turns-to-first/best valid over SOLVED tasks
Plus a side-by-side comparison and one full example transcript.

Usage:
  /usr/bin/python3 agentic_eval.py --model ecdsa-coder-1.5b-sft --per-band 3 --n-samples 8 --max-turns 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ecdsa-model
sys.path.insert(0, os.path.join(BASE, "proxy"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import proxy_env  # noqa: E402
import tasks      # noqa: E402

from eval_proxy import extract_opstream, SYSTEM  # noqa: E402
from agentic_solve import agentic_solve, ollama_chat_messages, _verify_safe  # noqa: E402


# ----------------------------------------------------------------------------------
# Build a fixed held-out evaluation set: a spread of unseen-seed instances across bands
# plus the dedicated held-out generalization tasks. >= 24 tasks by default.
# ----------------------------------------------------------------------------------
def build_eval_instances(per_band=3, seed0=60000, include_heldout=True):
    insts = []
    seen = set()
    s = seed0
    per = {b: 0 for b in tasks.BANDS}
    # Walk eval-only seeds (disjoint from SFT seeds 1000-1089 / GRPO 3407) collecting unseen
    # parameterizations of trained families, balanced across bands.
    while s < seed0 + 1200:
        cur = tasks.build_curriculum(rng=random.Random(s))
        for band, lst in cur.items():
            for inst in lst:
                if per.get(band, 0) >= per_band:
                    continue
                if inst.prompt in seen:
                    continue
                seen.add(inst.prompt)
                per[band] = per.get(band, 0) + 1
                insts.append(("unseen:" + band, inst))
        if all(per.get(b, 0) >= per_band for b in tasks.BANDS):
            break
        s += 1
    if include_heldout:
        for inst in tasks.heldout_tasks():
            insts.append(("heldout", inst))
    return insts


# ----------------------------------------------------------------------------------
# ONE-SHOT arm: single completion, verify once (mirrors eval_proxy one-shot protocol).
# ----------------------------------------------------------------------------------
def oneshot_solve(model, task_spec, prompt, temperature=0.2):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]
    try:
        comp = ollama_chat_messages(model, msgs, temperature=temperature, seed=7)
    except Exception as e:
        comp = f"__error__ {e}"
    ops = extract_opstream(comp)
    v = _verify_safe(ops, task_spec)
    return {
        "best_opstream": ops,
        "valid": bool(v.get("valid")),
        "cost": v.get("cost"),
        "vs_ref_pct": v.get("vs_ref_pct"),
        "reason": v.get("reason"),
        "frac_correct": v.get("frac_correct"),
        "frac_ancilla_clean": v.get("frac_ancilla_clean"),
        "turns": 0,
        "raw": comp,
    }


# ----------------------------------------------------------------------------------
# Metrics.
# ----------------------------------------------------------------------------------
def summarize(rows, ref_costs):
    n = len(rows)
    if n == 0:
        return {}
    valid = [r for r in rows if r["valid"]]
    wins, vsref = [], []
    for r in valid:
        rc = ref_costs.get(r["task_id"])
        c = r.get("cost")
        if rc and c is not None:
            vsref.append(c / rc)
            if c < rc:
                wins.append(r)
    solved_turns = [r["turns"] for r in valid]
    return {
        "n": n,
        "valid_rate": round(len(valid) / n, 3),
        "win_rate": round(len(wins) / n, 3),
        "mean_vs_ref": round(sum(vsref) / len(vsref), 3) if vsref else None,
        "mean_turns_to_solve": round(sum(solved_turns) / len(solved_turns), 3) if solved_turns else None,
        "n_valid": len(valid),
        "n_wins": len(wins),
    }


def by_tag_summary(rows, ref_costs):
    groups = {}
    for r in rows:
        groups.setdefault(r["tag"].split(":")[0], []).append(r)
    return {k: summarize(v, ref_costs) for k, v in sorted(groups.items())}


# ----------------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ecdsa-coder-1.5b-sft")
    ap.add_argument("--per-band", type=int, default=4, help="unseen instances per band")
    ap.add_argument("--n-samples", type=int, default=8, help="agentic best-of-N at turn 0")
    ap.add_argument("--max-turns", type=int, default=4, help="agentic repair turns")
    ap.add_argument("--oneshot-temp", type=float, default=0.2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    insts = build_eval_instances(per_band=a.per_band)
    print(f"[agentic_eval] model={a.model} tasks={len(insts)} "
          f"(per_band={a.per_band}, agentic n_samples={a.n_samples} max_turns={a.max_turns})")
    for i, (tag, inst) in enumerate(insts):
        print(f"   #{i:2d} {tag:14s} {inst.family:16s} "
              f"n_in={inst.task_spec['n_in']} ref={inst.task_spec['reference_cost']:.0f}")

    ref_costs = {}
    oneshot_rows, agentic_rows = [], []
    example_transcript = None

    t0 = time.time()
    for i, (tag, inst) in enumerate(insts):
        spec = inst.task_spec
        tid = f"{i}:{inst.family}"
        ref_costs[tid] = spec.get("reference_cost")

        # ----- one-shot -----
        os_res = oneshot_solve(a.model, spec, inst.prompt, temperature=a.oneshot_temp)
        oneshot_rows.append({
            "task_id": tid, "tag": tag, "family": inst.family,
            "valid": os_res["valid"], "cost": os_res["cost"],
            "reason": str(os_res["reason"])[:60], "turns": 0,
        })

        # ----- agentic -----
        ag_res = agentic_solve(a.model, spec, inst.prompt,
                               n_samples=a.n_samples, max_turns=a.max_turns)
        agentic_rows.append({
            "task_id": tid, "tag": tag, "family": inst.family,
            "valid": ag_res["valid"], "cost": ag_res["cost"],
            "reason": str(ag_res["reason"])[:60], "turns": ag_res["turns"],
        })

        # Capture one illustrative transcript: prefer a task the AGENTIC arm solved that the
        # one-shot arm FAILED (the clearest demonstration of the lift). Fall back to any solve.
        if example_transcript is None and ag_res["valid"] and not os_res["valid"]:
            example_transcript = {"task_id": tid, "tag": tag, "family": inst.family,
                                  "prompt": inst.prompt, "result": ag_res}

        print(f"   #{i:2d} {tag:14s} {inst.family:16s} "
              f"oneshot={'V' if os_res['valid'] else '.'}  "
              f"agentic={'V' if ag_res['valid'] else '.'} (turns={ag_res['turns']}, "
              f"cost={ag_res['cost']})  agentic_reason={str(ag_res['reason'])[:34]}")

    # Fall back: if no flip example, capture any agentic solve.
    if example_transcript is None:
        for i, (tag, inst) in enumerate(insts):
            r = agentic_rows[i]
            if r["valid"]:
                # re-run is wasteful; just note we had no clean flip example
                break

    dt = time.time() - t0

    os_overall = summarize(oneshot_rows, ref_costs)
    ag_overall = summarize(agentic_rows, ref_costs)
    os_break = by_tag_summary(oneshot_rows, ref_costs)
    ag_break = by_tag_summary(agentic_rows, ref_costs)

    # -------------------- print the comparison --------------------
    print("\n" + "=" * 72)
    print(f"AGENTIC VERIFIER-IN-THE-LOOP vs ONE-SHOT  —  model={a.model}")
    print(f"tasks={len(insts)}  wall={dt:.0f}s  "
          f"(agentic: n_samples={a.n_samples}, max_turns={a.max_turns})")
    print("=" * 72)
    hdr = f"{'metric':<22}{'one-shot':>14}{'agentic':>14}{'delta':>14}"
    print(hdr); print("-" * len(hdr))

    def _fmt(x):
        return "—" if x is None else f"{x:.3f}"

    def _row(name, ko, ka):
        vo = os_overall.get(ko); va = ag_overall.get(ka if ka else ko)
        delta = (va - vo) if (isinstance(vo, (int, float)) and isinstance(va, (int, float))) else None
        print(f"{name:<22}{_fmt(vo):>14}{_fmt(va):>14}{_fmt(delta):>14}")

    _row("valid_rate", "valid_rate", "valid_rate")
    _row("win_rate", "win_rate", "win_rate")
    _row("mean_vs_ref", "mean_vs_ref", "mean_vs_ref")
    print(f"{'mean_turns_to_solve':<22}{_fmt(0.0):>14}"
          f"{_fmt(ag_overall.get('mean_turns_to_solve')):>14}{'':>14}")
    print(f"{'n_valid':<22}{os_overall.get('n_valid',0):>14d}{ag_overall.get('n_valid',0):>14d}"
          f"{ag_overall.get('n_valid',0)-os_overall.get('n_valid',0):>14d}")

    print("\nBy task group (valid_rate one-shot -> agentic):")
    for grp in sorted(set(list(os_break) + list(ag_break))):
        o = os_break.get(grp, {}); g = ag_break.get(grp, {})
        print(f"  {grp:10s} n={g.get('n', o.get('n','?')):>3}  "
              f"valid {_fmt(o.get('valid_rate'))} -> {_fmt(g.get('valid_rate'))}   "
              f"win {_fmt(o.get('win_rate'))} -> {_fmt(g.get('win_rate'))}")

    # -------------------- print one example transcript --------------------
    print("\n" + "=" * 72)
    print("EXAMPLE TRANSCRIPT (agentic solve, chosen where one-shot failed)")
    print("=" * 72)
    if example_transcript:
        et = example_transcript
        print(f"task #{et['task_id']}  tag={et['tag']}  family={et['family']}")
        print("\n--- PROMPT (task) ---")
        print(et["prompt"])
        for rec in et["result"]["transcript"]:
            if rec["kind"] == "best_of_n":
                print(f"\n--- TURN 0: best-of-{rec['n_samples']} ---")
                for s in rec["samples"]:
                    print(f"  sample {s['sample']} (T={s['temperature']}): "
                          f"valid={s['valid']} reason={s['reason']} "
                          f"fc={s['frac_correct']} cost={s['cost']}")
            else:
                print(f"\n--- TURN {rec['turn']}: repair ---")
                print("  [feedback to model]")
                for ln in rec["feedback"].splitlines():
                    print("    " + ln)
                print("  [model op-stream]")
                for ln in rec["opstream"].splitlines():
                    print("    " + ln)
                print(f"  [verify] valid={rec['valid']} reason={rec['reason']} "
                      f"fc={rec['frac_correct']} cost={rec['cost']}")
        print(f"\nFINAL: valid={et['result']['valid']} cost={et['result']['cost']} "
              f"turns={et['result']['turns']}")
    else:
        print("(no task where agentic succeeded while one-shot failed; "
              "see per-group table for the lift)")

    # -------------------- persist --------------------
    result = {
        "model": a.model,
        "wall_s": round(dt, 1),
        "n_tasks": len(insts),
        "config": {"per_band": a.per_band, "n_samples": a.n_samples,
                   "max_turns": a.max_turns, "oneshot_temp": a.oneshot_temp},
        "oneshot": {"overall": os_overall, "by_group": os_break},
        "agentic": {"overall": ag_overall, "by_group": ag_break},
    }
    print("\n" + json.dumps(result, indent=2))

    out = a.out or os.path.join(BASE, "eval", "results",
                                f"agentic_{a.model.replace(':', '_').replace('/', '_')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"result": result,
                   "oneshot_rows": oneshot_rows,
                   "agentic_rows": agentic_rows,
                   "example_transcript": example_transcript}, f, indent=2)
    print(f"\n[agentic_eval] wrote {out}")
    return result


if __name__ == "__main__":
    main()
