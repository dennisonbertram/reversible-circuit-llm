#!/usr/bin/env python3
"""tooluse_eval.py — THE DECISIVE EXPERIMENT: does state-externalization beat the
one-shot reasoning/imitation ceiling?

For the SAME held-out tasks (unseen seeds >= 60000) and the SAME existing Ollama model
(ecdsa-coder-1.5b-v4, no retraining), we measure two regimes side by side:

  (a) NO-TOOL  : the existing one-shot / best-of-N approach (eval_proxy style). The model must
                 emit a WHOLE circuit; we sample N completions and keep the best valid one.
                 This is where the model plateaus at valid_rate ~4% (only the trivial band B0).

  (b) TOOL     : the STATE-EXTERNALIZING loop (tooluse_solve). A tool tracks the cumulative
                 circuit state; the model only picks ONE gate per turn given current-vs-target.

We focus on the bands the model FAILS one-shot but that ARE tool-solvable:
  gf2_linear at B1/B2/B3 (pure CX, Toffoli-free solutions exist via Gaussian elimination),
  const_add / reg_add / controlled_addsub at B1/B2 (ripple-carry state tracking).

Output: a per-(family,band) table of no-tool valid_rate vs TOOL valid_rate, mean steps-to-solve,
mean cost; plus ONE full tool-use transcript (state -> gate -> state -> ... -> solved/stopped).

Usage:
  python3 tooluse_eval.py --model ecdsa-coder-1.5b-v4 --per-cell 5 --n-baseline 8 --max-steps 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ecdsa-model
sys.path.insert(0, os.path.join(BASE, "proxy"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # eval/

import proxy_env  # noqa: E402
import tasks      # noqa: E402
from eval_proxy import extract_opstream, SYSTEM, OLLAMA  # noqa: E402
from tooluse_solve import tooluse_solve, ollama_chat_messages  # noqa: E402


# Which (family, band) cells to probe. These are the bands the model fails one-shot.
# "B0" for gf2_linear is a 2-bit CONTROL (same width as the few-shot worked example) — it
# isolates "does state-externalization beat one-shot" from "can a 1.5B model read a wide
# residual". n_in is taken from the band plan, except gf2_linear B0 which we pin to 2 bits.
TARGET_CELLS = [
    ("gf2_linear", "B0"),          # 2-bit control (worked-example scale)
    ("gf2_linear", "B1"),
    ("gf2_linear", "B2"),
    ("gf2_linear", "B3"),          # B3 plan has no gf2_linear by default -> built ad hoc below
    ("const_add", "B1"),
    ("const_add", "B2"),
    ("reg_add", "B2"),
    ("controlled_addsub", "B1"),
    ("controlled_addsub", "B2"),
]

# gf2_linear width per band (overrides band-plan n_in where needed).
_GF2_BITS = {"B0": 2, "B1": 3, "B2": 4, "B3": 5}


def _sample_cell_instances(family, band, n, seed0):
    """Sample n DISTINCT instances of (family, band) using eval-only seeds (>= seed0)."""
    insts = []
    seen = set()
    s = seed0
    guard = 0
    while len(insts) < n and guard < 4000:
        guard += 1
        rng = random.Random(s)
        s += 1
        try:
            if family == "gf2_linear":
                # Build a gf2_linear at the chosen width directly (uniform across B0..B3). Require
                # a NON-trivial map: not the identity, and at least 2 rows have an off-diagonal
                # coupling (so the model must actually do >=2 row reductions; this excludes the
                # near-identity degenerate instances that the band sampler can emit).
                nbits = _GF2_BITS.get(band, tasks._BAND_PLAN[band]["n_in"])
                M = tasks._rand_invertible_gf2(nbits, rng)
                ident = [1 << i for i in range(nbits)]
                offdiag_rows = sum(1 for i in range(nbits) if (M[i] & ~(1 << i)) != 0)
                if M == ident or offdiag_rows < 2:
                    continue
                inst = tasks.build_gf2_linear(nbits, M)
                inst.band = band
                inst.prompt = tasks.render_prompt(inst)
            else:
                inst = tasks.sample_instance(family, band, rng)
        except Exception:
            continue
        key = inst.prompt
        if key in seen:
            continue
        seen.add(key)
        insts.append(inst)
    return insts


# ----------------------------------------------------------------------------------
# NO-TOOL baseline: one-shot best-of-N (mirrors eval_proxy.evaluate's best-of-k logic).
# ----------------------------------------------------------------------------------
def baseline_best_of_n(model, inst, n_samples, base_temperature=0.4):
    spec = inst.task_spec
    best = None
    for i in range(n_samples):
        t = base_temperature + (i / max(1, n_samples - 1)) * 0.6
        seed = 1000 + i
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": inst.prompt}]
        try:
            comp = ollama_chat_messages(model, msgs, temperature=t, seed=seed, num_predict=1024)
        except Exception as e:
            comp = f"__error__ {e}"
        ops = extract_opstream(comp)
        try:
            v = proxy_env.verify(ops, spec) if ops.strip() else {"valid": False, "reason": "empty",
                                                                  "cost": None}
        except Exception as e:
            v = {"valid": False, "reason": f"verify-exc:{e}", "cost": None}
        cand = {"valid": bool(v.get("valid")), "cost": v.get("cost"),
                "reason": str(v.get("reason"))[:50]}
        key = (cand["valid"], -(cand["cost"] or 1e18) if cand["valid"] else -1e18)
        if best is None or key > best[0]:
            best = (key, cand)
    return best[1]


# ----------------------------------------------------------------------------------
# Run both regimes over all cells.
# ----------------------------------------------------------------------------------
def run(model, per_cell, n_baseline, max_steps, temperature, seed0, keep_transcript_for):
    results = {}        # (family,band) -> dict of aggregates
    sample_transcript = None
    for (family, band) in TARGET_CELLS:
        insts = _sample_cell_instances(family, band, per_cell, seed0)
        if not insts:
            continue
        b_valid = 0
        t_valid = 0
        t_steps = []
        t_costs = []
        b_costs = []
        ref_costs = []
        per_inst = []
        for idx, inst in enumerate(insts):
            ref_cost = inst.task_spec.get("reference_cost")
            ref_costs.append(ref_cost)

            # (a) no-tool baseline
            bret = baseline_best_of_n(model, inst, n_baseline)
            if bret["valid"]:
                b_valid += 1
                if bret["cost"] is not None:
                    b_costs.append(bret["cost"])

            # (b) tool
            spec = dict(inst.task_spec)
            spec["family"] = inst.family
            keep = (keep_transcript_for is not None
                    and family == keep_transcript_for[0] and band == keep_transcript_for[1]
                    and sample_transcript is None)
            tret = tooluse_solve(model, spec, inst.prompt, max_steps=max_steps,
                                 temperature=temperature, verbose=False)
            if tret["solved"]:
                t_valid += 1
                t_steps.append(tret["steps"])
                if tret["cost"] is not None:
                    t_costs.append(tret["cost"])
                if keep:
                    sample_transcript = {"family": family, "band": band,
                                         "ref_cost": ref_cost, "params": inst.params,
                                         "result": tret}
            per_inst.append({"params": inst.params, "ref_cost": ref_cost,
                             "baseline_valid": bret["valid"], "baseline_reason": bret["reason"],
                             "tool_solved": tret["solved"], "tool_steps": tret["steps"],
                             "tool_cost": tret["cost"], "tool_reason_n_mismatch": tret.get("n_mismatch")})
            print(f"  [{family:18s} {band}] inst {idx+1}/{len(insts)} "
                  f"ref={ref_cost:.0f} | baseline_valid={bret['valid']} "
                  f"| tool_solved={tret['solved']} steps={tret['steps']} cost={tret['cost']}")

        # If we wanted a transcript for this cell but nothing solved, keep the longest attempt.
        if (keep_transcript_for is not None and family == keep_transcript_for[0]
                and band == keep_transcript_for[1] and sample_transcript is None and insts):
            # re-run the first instance with verbose transcript retained
            inst = insts[0]
            spec = dict(inst.task_spec); spec["family"] = inst.family
            tret = tooluse_solve(model, spec, inst.prompt, max_steps=max_steps,
                                 temperature=temperature, verbose=False)
            sample_transcript = {"family": family, "band": band,
                                 "ref_cost": inst.task_spec.get("reference_cost"),
                                 "params": inst.params, "result": tret, "note": "best attempt (not solved)"}

        n = len(insts)
        results[(family, band)] = {
            "n": n,
            "baseline_valid_rate": round(b_valid / n, 3),
            "tool_valid_rate": round(t_valid / n, 3),
            "tool_mean_steps": round(sum(t_steps) / len(t_steps), 1) if t_steps else None,
            "tool_mean_cost": round(sum(t_costs) / len(t_costs), 2) if t_costs else None,
            "baseline_mean_cost": round(sum(b_costs) / len(b_costs), 2) if b_costs else None,
            "mean_ref_cost": round(sum(ref_costs) / len(ref_costs), 1) if ref_costs else None,
            "per_inst": per_inst,
        }
    return results, sample_transcript


def print_table(model, results):
    print("\n" + "=" * 100)
    print(f"SIDE-BY-SIDE: no-tool (best-of-N one-shot) vs TOOL (state-externalized)  model={model}")
    print("=" * 100)
    hdr = (f"{'family':18s} {'band':4s} {'n':>3s} | {'NO-TOOL':>8s} | {'TOOL':>6s} "
           f"| {'tool_steps':>10s} | {'tool_cost':>9s} | {'ref_cost':>8s}")
    print(hdr)
    print("-" * 100)
    agg_b = agg_t = agg_n = 0
    for (family, band), r in results.items():
        agg_b += r["baseline_valid_rate"] * r["n"]
        agg_t += r["tool_valid_rate"] * r["n"]
        agg_n += r["n"]
        print(f"{family:18s} {band:4s} {r['n']:>3d} | "
              f"{r['baseline_valid_rate']*100:>6.1f}% | "
              f"{r['tool_valid_rate']*100:>4.1f}% | "
              f"{str(r['tool_mean_steps']):>10s} | "
              f"{str(r['tool_mean_cost']):>9s} | "
              f"{str(r['mean_ref_cost']):>8s}")
    print("-" * 100)
    if agg_n:
        print(f"{'OVERALL':18s} {'':4s} {agg_n:>3d} | "
              f"{agg_b/agg_n*100:>6.1f}% | {agg_t/agg_n*100:>4.1f}% |")
    print("=" * 100)


def print_transcript(st):
    if not st:
        print("\n[no transcript captured]")
        return
    print("\n" + "=" * 100)
    note = st.get("note", "SOLVED")
    print(f"FULL TOOL-USE TRANSCRIPT  [{st['family']} {st['band']}]  ref_cost={st['ref_cost']:.0f}"
          f"  params={st['params']}  ({note})")
    print("=" * 100)
    res = st["result"]
    for ev in res["transcript"]:
        if ev["kind"] == "model":
            print(f"\n  TURN {ev['turn']}  model replies: {ev['op']!r}")
            raw = ev.get("raw", "")
            if raw and raw != ev["op"]:
                print(f"           (raw: {raw[:80]!r})")
        else:
            if ev.get("error"):
                print(f"           tool: REJECTED '{ev.get('applied')}' -> {ev['error'][:70]}")
            elif ev.get("rejected_cycle"):
                print(f"           tool: CYCLE-REVERTED '{ev.get('applied')}' "
                      f"(n_mismatch now {ev.get('n_mismatch')})")
            else:
                print(f"           tool: applied '{ev.get('applied')}' -> "
                      f"n_mismatch={ev.get('n_mismatch')} cost={ev.get('cost')} "
                      f"done={ev.get('done')}")
    print(f"\n  RESULT: solved={res['solved']} steps={res['steps']} cost={res['cost']}")
    print("  FINAL OPSTREAM:")
    for ln in res["opstream"].splitlines():
        print("    " + ln)
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ecdsa-coder-1.5b-v4")
    ap.add_argument("--per-cell", type=int, default=5, help="instances per (family,band)")
    ap.add_argument("--n-baseline", type=int, default=8, help="best-of-N for the no-tool baseline")
    ap.add_argument("--max-steps", type=int, default=40, help="gate budget for the tool loop")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed0", type=int, default=60000)
    ap.add_argument("--transcript-cell", default="gf2_linear:B0",
                    help="family:band to capture a full transcript for")
    ap.add_argument("--cells", default="",
                    help="comma-separated family:band cells to run (default: all TARGET_CELLS)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    global TARGET_CELLS
    if a.cells.strip():
        TARGET_CELLS = [tuple(c.split(":")) for c in a.cells.split(",") if ":" in c]

    tc = tuple(a.transcript_cell.split(":")) if ":" in a.transcript_cell else None

    print(f"[tooluse_eval] model={a.model} per_cell={a.per_cell} n_baseline={a.n_baseline} "
          f"max_steps={a.max_steps} cells={len(TARGET_CELLS)}")
    t0 = time.time()
    results, transcript = run(a.model, a.per_cell, a.n_baseline, a.max_steps,
                              a.temperature, a.seed0, tc)
    dt = time.time() - t0

    print_table(a.model, results)
    print_transcript(transcript)
    print(f"\n[tooluse_eval] wall={dt:.1f}s")

    out = a.out or os.path.join(BASE, "eval", "results",
                                f"tooluse_{a.model.replace(':', '_').replace('/', '_')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    serial = {f"{fam}:{band}": r for (fam, band), r in results.items()}
    with open(out, "w") as f:
        json.dump({"model": a.model, "wall_s": round(dt, 1), "results": serial,
                   "transcript": transcript}, f, indent=2, default=str)
    print(f"[tooluse_eval] wrote {out}")


if __name__ == "__main__":
    main()
