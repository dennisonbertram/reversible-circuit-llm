#!/usr/bin/env python3
"""Evaluate an Ollama model on HELD-OUT proxy reversible-circuit tasks.

Drives the model to emit op-streams for tasks it did NOT train on, scores each with the
faithful proxy verifier (proxy_env.verify, bit-identical to the real ECDSA-fail simulator),
and reports the headline capability metrics. Run base vs trained to prove the model learned
the skill and generalizes.

Metrics per model:
  parse_rate     fraction whose extracted op-stream parses as a valid circuit shape
  valid_rate     fraction that pass ALL 4 validity gates AND compute the target function
  win_rate       fraction that are VALID and strictly cheaper than the reference circuit
  mean_vs_ref    mean (cost / reference_cost) over valid solutions  (<1.0 = beats reference)
  mean_reward    canonical training reward (proxy_env.reward on the RAW completion)

Eval sets:
  --set unseen   : in-distribution families at unseen params (eval-only RNG seeds)
  --set heldout  : the generalization families never used in training (fma, sbox6)
  --set both     : both (default)

Usage:
  python3 eval_proxy.py --model qwen2.5-coder:1.5b --per-band 6
  python3 eval_proxy.py --model ecdsa-coder-1.5b   --per-band 6   # after GGUF+ollama create
"""
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
import proxy_env  # noqa: E402
import tasks      # noqa: E402

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
SYSTEM = open(os.path.join(BASE, "proxy", "system_prompt.txt")).read().strip()

OP_KINDS = {
    "X", "Z", "CX", "CZ", "CCX", "CCZ", "SWAP", "R", "HMR", "NEG",
    "BIT_INVERT", "BIT_STORE0", "BIT_STORE1", "REGISTER", "APPEND_TO_REGISTER",
    "PUSH_CONDITION", "POP_CONDITION", "DEBUG_PRINT",
}


def extract_opstream(text):
    """Pull a clean op-stream out of a model completion: prefer the first fenced block,
    else keep only lines whose first token is a known op KIND (drops prose)."""
    fenced = re.findall(r"```[a-zA-Z0-9_]*\n(.*?)```", text, re.S)
    body = fenced[0] if fenced else text
    body = body.replace(";", "\n")  # some models emit ;-separated ops on one line
    out = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.split()[0].upper() in OP_KINDS:
            out.append(s)
    return "\n".join(out)


def ollama_chat(model, user, temperature=0.2, timeout=120):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    return obj["message"]["content"]


def build_eval_instances(which, per_band, seed0=60000):
    insts = []
    if which in ("unseen", "both"):
        # in-distribution families, eval-only seeds (disjoint from SFT seeds 1000-1089
        # and GRPO seed 3407) -> unseen parameterizations of trained families.
        seen = set()
        s = seed0
        per = {b: 0 for b in tasks.BANDS} if hasattr(tasks, "BANDS") else {}
        while s < seed0 + 400:
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
            if all(per.get(b, 0) >= per_band for b in cur):
                break
            s += 1
    if which in ("heldout", "both"):
        try:
            held = tasks.heldout_tasks()
            # heldout_tasks() may return list[Instance] or {name: Instance}
            items = held.values() if isinstance(held, dict) else held
            for inst in items:
                insts.append(("heldout", inst))
        except Exception as e:
            print(f"[eval] heldout_tasks() unavailable ({e}); trying builders")
            for fn in ("build_heldout_fma", "build_heldout_sbox6"):
                try:
                    insts.append(("heldout", getattr(tasks, fn)(rng=random.Random(7))))
                except Exception as e2:
                    print(f"[eval] {fn} failed: {e2}")
    return insts


def evaluate(model, instances, temperature, samples):
    rows = []
    for tag, inst in instances:
        spec = inst.task_spec
        ref_cost = spec.get("reference_cost")
        best = None
        for _k in range(samples):
            try:
                comp = ollama_chat(model, inst.prompt, temperature=temperature)
            except Exception as e:
                comp = f"__error__ {e}"
            ops = extract_opstream(comp)
            try:
                v = proxy_env.verify(ops, spec) if ops.strip() else {"valid": False, "reason": "empty"}
            except Exception as e:
                v = {"valid": False, "reason": f"verify-exc:{e}"}
            try:
                rew = proxy_env.reward(prompts=[inst.prompt], completions=[comp], task_spec=[spec])[0]
            except Exception:
                rew = None
            cand = {
                "tag": tag, "family": getattr(inst, "family", "?"),
                "band": getattr(inst, "band", "?"),
                "valid": bool(v.get("valid")),
                "cost": v.get("cost"), "ref_cost": ref_cost,
                "reason": str(v.get("reason"))[:60], "reward": rew,
                "parsed": bool(ops.strip()),
            }
            # keep the best sample by (valid, -cost, reward)
            key = (cand["valid"], -(cand["cost"] or 1e18) if cand["valid"] else -1e18, cand["reward"] or -9)
            if best is None or key > best[0]:
                best = (key, cand)
        rows.append(best[1])
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {}
    valid = [r for r in rows if r["valid"]]
    wins = [r for r in valid if r["ref_cost"] and r["cost"] is not None and r["cost"] < r["ref_cost"]]
    vsref = [r["cost"] / r["ref_cost"] for r in valid if r["ref_cost"]]
    rewards = [r["reward"] for r in rows if r["reward"] is not None]
    return {
        "n": n,
        "parse_rate": round(sum(r["parsed"] for r in rows) / n, 3),
        "valid_rate": round(len(valid) / n, 3),
        "win_rate": round(len(wins) / n, 3),
        "mean_vs_ref": round(sum(vsref) / len(vsref), 3) if vsref else None,
        "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ollama model tag")
    ap.add_argument("--set", choices=["unseen", "heldout", "both"], default="both")
    ap.add_argument("--per-band", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--samples", type=int, default=1, help="best-of-k samples per task")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    insts = build_eval_instances(a.set, a.per_band)
    print(f"[eval] model={a.model} tasks={len(insts)} set={a.set} samples={a.samples}")
    t0 = time.time()
    rows = evaluate(a.model, insts, a.temperature, a.samples)
    dt = time.time() - t0

    overall = summarize(rows)
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"].split(":")[0], []).append(r)
    breakdown = {k: summarize(v) for k, v in sorted(by_tag.items())}

    result = {"model": a.model, "wall_s": round(dt, 1), "overall": overall, "breakdown": breakdown}
    print(json.dumps(result, indent=2))
    out = a.out or os.path.join(BASE, "eval", "results", f"proxy_{a.model.replace(':', '_').replace('/', '_')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"result": result, "rows": rows}, f, indent=2)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
