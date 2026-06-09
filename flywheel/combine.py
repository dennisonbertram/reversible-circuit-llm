"""combine.py — build the cumulative expert-iteration SFT set for one flywheel round.

Merge the seed expert traces with every harvested-solve jsonl produced so far, DEDUP by the
underlying GF(2) task (parsed out of each trace's first user render), and for each task keep the
trace with the FEWEST assistant gates (cheapest play; ties -> first seen). This is the STaR /
expert-iteration replay buffer: the model's own cheaper solutions REPLACE the expert demo for a
task once it finds one, while harvested NEW tasks expand coverage.

Usage:
  python flywheel/combine.py \
      --expert data/sft_tool_8base.jsonl \
      --harvests artifacts/harvest_iter1.jsonl,artifacts/harvest_iter2.jsonl \
      --out data/sft_flywheel_iter2.jsonl \
      [--cap-per-band 1400] [--max-tokens 6200]

Keying: (n, tuple(target_M)) parsed from the `target y{i} = x.. ^ x..` rows of the first user
turn. A trace whose target cannot be parsed is kept as-is under a unique fallback key (never
dropped, never dedup'd) so we never silently lose data.

Prints a composition report (per band: from expert vs harvested, total kept) to stderr and a
one-line COMBINE_STATS json to stdout.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

_ROW_RE = re.compile(r"target y(\d+)\s*=\s*([x0-9\s\^]+)")


def parse_target_key(messages):
    """Return (n, tuple(M)) parsed from the first user render, or None if unparseable."""
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    rows_by_i = {}
    for m in _ROW_RE.finditer(first_user):
        i = int(m.group(1))
        rhs = m.group(2).strip()
        mask = 0
        if rhs and rhs != "0":
            for tok in rhs.split("^"):
                tok = tok.strip()
                if tok.startswith("x") and tok[1:].isdigit():
                    mask |= (1 << int(tok[1:]))
        rows_by_i[i] = mask
    if not rows_by_i:
        return None
    n = max(rows_by_i) + 1
    return (n, tuple(rows_by_i.get(i, 0) for i in range(n)))


def n_gates(messages):
    return sum(1 for m in messages if m["role"] == "assistant")


def approx_tokens(messages):
    return sum(len(m["content"]) for m in messages) // 3


def band_of(key, messages):
    n = key[0] if key is not None else None
    if n is None:
        # fall back to parsing "on N bits" from the intro user turn
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        mm = re.search(r"on (\d+) bits", first_user)
        n = int(mm.group(1)) if mm else 0
    return {2: "B0", 3: "B1", 4: "B2", 5: "B3", 6: "B4", 7: "B5", 8: "B6"}.get(n, f"n{n}")


def load(path):
    rows = []
    try:
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
    except FileNotFoundError:
        print(f"[combine] WARN: {path} not found, skipping", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True, help="seed expert traces jsonl")
    ap.add_argument("--harvests", default="", help="comma-separated harvested-solve jsonls")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-per-band", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = no filter")
    args = ap.parse_args()

    sources = [("expert", args.expert)]
    for h in [x for x in args.harvests.split(",") if x.strip()]:
        sources.append(("harvest", h))

    # best[key] = (origin, n_gates, messages)
    best = {}
    fallback = []  # unparseable-target traces, always kept
    seen_per_source = defaultdict(int)
    fallback_uid = 0
    for origin, path in sources:
        for r in load(path):
            msgs = r["messages"]
            seen_per_source[path] += 1
            if args.max_tokens and approx_tokens(msgs) > args.max_tokens:
                continue
            key = parse_target_key(msgs)
            if key is None:
                fallback.append((origin, msgs))
                fallback_uid += 1
                continue
            g = n_gates(msgs)
            prev = best.get(key)
            # keep fewest gates; on a tie, prefer an existing entry (expert seen first -> stable)
            if prev is None or g < prev[1]:
                best[key] = (origin, g, msgs)

    # assemble, optional per-band cap (keep cheapest within band)
    by_band_items = defaultdict(list)
    for key, (origin, g, msgs) in best.items():
        by_band_items[band_of(key, msgs)].append((g, origin, msgs))
    for origin, msgs in fallback:
        by_band_items[band_of(None, msgs)].append((n_gates(msgs), origin, msgs))

    kept = []
    comp = {}
    for band, items in sorted(by_band_items.items()):
        items.sort(key=lambda t: t[0])  # cheapest first
        if args.cap_per_band and len(items) > args.cap_per_band:
            # ALWAYS keep harvested traces — they are the flywheel's self-improvement signal and
            # are often pricier than the synth-optimal expert demos, so a naive cheapest-N cap
            # would silently discard exactly the model-found solutions we want to reinforce.
            # Fill the remaining budget with the cheapest expert demos.
            harv = [it for it in items if it[1] == "harvest"]
            exp = [it for it in items if it[1] == "expert"]
            keep_exp = exp[: max(0, args.cap_per_band - len(harv))]
            items = sorted(harv + keep_exp, key=lambda t: t[0])
        n_exp = sum(1 for _g, o, _m in items if o == "expert")
        n_har = sum(1 for _g, o, _m in items if o == "harvest")
        comp[band] = {"total": len(items), "expert": n_exp, "harvest": n_har}
        kept.extend({"messages": m} for _g, _o, m in items)

    import random
    random.Random(7).shuffle(kept)
    with open(args.out, "w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")

    stats = {
        "out": args.out,
        "total_kept": len(kept),
        "unique_tasks": len(best),
        "fallback_unparseable": len(fallback),
        "sources": {p: seen_per_source[p] for _o, p in sources},
        "by_band": comp,
    }
    print("[combine] composition (per band: total / from-expert / from-harvest):", file=sys.stderr)
    for band, c in sorted(comp.items()):
        print(f"    {band}: {c['total']:5d}  (expert {c['expert']:5d}  harvest {c['harvest']:5d})",
              file=sys.stderr)
    print(f"[combine] wrote {len(kept)} traces -> {args.out}", file=sys.stderr)
    print("COMBINE_STATS " + json.dumps(stats))


if __name__ == "__main__":
    main()
