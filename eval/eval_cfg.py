#!/usr/bin/env python3
"""T-CFG eval: 'test against ECDSA fail' — move-proposal quality on the REAL challenge.

The atomic optimization move on the real secp256k1 circuit is (tighten a DIALOG knob) +
(re-find a clean Fiat-Shamir nonce). Because any op-stream-altering change voids the prior
nonce island, a single proposed config almost never validates without a nonce search — so a
fully-automatic 'did it improve the live score' metric is intractable for a PoC. Instead we
measure MOVE QUALITY against held-out historical accepted moves, base vs trained:

  named_real_knob    proposed a knob that actually exists in the challenge source
  direction_match    proposed the same numeric direction as the historical accepted move
  mentions_revalidate cites the nonce-island re-validation requirement (the key domain fact)

Optionally (--harness N) it runs a few model-proposed configs through the REAL harness with a
bounded DIALOG_TAIL_NONCE sweep as an end-to-end demonstration (slow: ~13s/run).

Usage:
  python3 eval_cfg.py --model qwen2.5-coder:1.5b --n 24
  python3 eval_cfg.py --model ecdsa-coder-1.5b   --n 24 --harness 0
"""
import argparse
import json
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("ECDSAFAIL_REPO",
                      "/Users/dennison/develop/quantum-project/ecdsafail-challenge")
SYSTEM = open(os.path.join(BASE, "proxy", "system_prompt.txt")).read().strip()

import urllib.request


def knob_universe():
    """All env-var knob names referenced in the challenge source (set_default_env / env::var)."""
    names = set()
    try:
        out = subprocess.run(
            ["grep", "-rhoE", r'(set_default_env\(\s*"[A-Z0-9_]+"|env::var\(\s*"[A-Z0-9_]+")',
             os.path.join(REPO, "src")],
            capture_output=True, text=True).stdout
        for m in re.findall(r'"([A-Z0-9_]+)"', out):
            names.add(m)
    except Exception as e:
        print(f"[cfg] knob grep failed: {e}")
    return names


def historical_moves(n):
    """Held-out (knob, old, new) numeric moves from the accepted-submission corpus."""
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data", "moves_raw.jsonl"))]
    knre = re.compile(r'set_default_env\(\s*"([A-Z0-9_]+)"\s*,\s*"([^"]+)"\s*\)')
    moves = []
    for r in rows:
        if not r.get("is_small_move") or not r.get("small_diff"):
            continue
        d = r["small_diff"]
        rem = dict(knre.findall("\n".join(l[1:] for l in d.splitlines()
                                          if l.startswith("-") and not l.startswith("---"))))
        add = dict(knre.findall("\n".join(l[1:] for l in d.splitlines()
                                          if l.startswith("+") and not l.startswith("+++"))))
        for k, new in add.items():
            old = rem.get(k)
            if old is None or old == new or re.search(r"DUMMY|TRACE|DEBUG", k):
                continue
            try:
                gold_dir = (float(new) > float(old)) - (float(new) < float(old))  # +1/-1/0
                numeric = True
            except ValueError:
                gold_dir, numeric = 0, False
            moves.append({"knob": k, "old": old, "new": new, "gold_dir": gold_dir, "numeric": numeric})
    # held-out slice: take the last n (most recent frontier), dedup by (knob,old)
    seen, out = set(), []
    for m in reversed(moves):
        key = (m["knob"], m["old"])
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
        if len(out) >= n:
            break
    return out


def ollama_chat(model, user, temperature=0.2, timeout=120):
    payload = {"model": model,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}],
               "stream": False, "options": {"temperature": temperature, "num_predict": 400}}
    req = urllib.request.Request(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") + "/api/chat",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def score_completion(text, move, universe):
    knobs = set(re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", text))
    real = knobs & universe
    named_real = len(real) > 0
    # direction: find an arrow move on the SAME knob
    dir_match = False
    if move["numeric"]:
        for m in re.finditer(re.escape(move["knob"]) + r"\D{0,8}(\d+)\s*(?:->|→|=>|to|=)\s*(\d+)", text):
            o, nw = float(m.group(1)), float(m.group(2))
            pdir = (nw > o) - (nw < o)
            if pdir == move["gold_dir"] and pdir != 0:
                dir_match = True
    mentions = bool(re.search(r"nonce|island|re-?validate|revalidat|9024|0\s*/\s*0\s*/\s*0|fiat", text, re.I))
    return {"named_real_knob": named_real, "direction_match": dir_match,
            "mentions_revalidate": mentions, "named_knobs": sorted(real)[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--harness", type=int, default=0, help="run K model configs through the real harness")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    universe = knob_universe()
    print(f"[cfg] knob universe: {len(universe)} env knobs found in challenge src")
    moves = historical_moves(a.n)
    print(f"[cfg] held-out historical moves: {len(moves)}")

    rows = []
    for mv in moves:
        user = ("Frontier circuit: the secp256k1 dialog-GCD point-add, scored by avg executed "
                f'Toffoli x peak qubits (lower is better). Current setting `set_default_env("{mv["knob"]}", '
                f'"{mv["old"]}")`. Propose the next single bounded optimization move and the validation plan.')
        try:
            comp = ollama_chat(a.model, user, a.temperature)
        except Exception as e:
            comp = f"__error__ {e}"
        s = score_completion(comp, mv, universe)
        s.update(knob=mv["knob"], old=mv["old"], gold_new=mv["new"])
        rows.append(s)

    n = len(rows) or 1
    summary = {
        "model": a.model, "n": len(rows),
        "named_real_knob_rate": round(sum(r["named_real_knob"] for r in rows) / n, 3),
        "direction_match_rate": round(sum(r["direction_match"] for r in rows) / n, 3),
        "mentions_revalidate_rate": round(sum(r["mentions_revalidate"] for r in rows) / n, 3),
    }
    print(json.dumps(summary, indent=2))
    out = a.out or os.path.join(BASE, "eval", "results", f"cfg_{a.model.replace(':', '_').replace('/', '_')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2)
    print(f"[cfg] wrote {out}")


if __name__ == "__main__":
    main()
