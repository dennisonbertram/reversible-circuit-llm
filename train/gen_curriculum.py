"""Mass-generate the v2 optimal-target curriculum on Modal CPU fan-out.

For tens of thousands of proxy tasks: take the (valid) reference, run the synthesis engine
(proxy/synth.py) to find a near-optimal/optimal VALID circuit, and emit
  - SFT target : prompt -> the OPTIMAL op-stream (teaches optimality, not just format)
  - DPO pair   : (chosen = optimal, rejected = the costlier reference)  (collapse-proof)

The verifier is microsecond-cheap, so this fans out across many CPU containers. Workers
RECONSTRUCT each task deterministically from (seed, band, index) — task_spec holds a live
callable `f` that can't be pickled, so we never ship it.

  modal run train/gen_curriculum.py --n-seeds 4000 --shards 80 --time-budget 1.5
Writes data/sft_v2.jsonl + data/dpo_v2.jsonl locally (held-out seeds 60000+ are NOT used).
"""
import json
from functools import lru_cache
from pathlib import Path
import modal

app = modal.App("ecdsa-gen")
HERE = Path(__file__).parent.resolve()
PROXY_DIR = (HERE.parent / "proxy").resolve()

img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("numpy")
       .add_local_dir(str(PROXY_DIR), remote_path="/root/proxy"))


@lru_cache(maxsize=512)
def _curric(seed, npf):
    import random
    import sys
    if "/root/proxy" not in sys.path:
        sys.path.insert(0, "/root/proxy")
    import tasks
    return tasks.build_curriculum(rng=random.Random(seed), n_per_family=npf)


def _synth_one(args):
    """Worker: reconstruct one instance from (seed, band, idx) and optimize it."""
    seed, band, idx, time_budget, npf = args
    import sys
    if "/root/proxy" not in sys.path:
        sys.path.insert(0, "/root/proxy")
    import synth
    try:
        inst = _curric(seed, npf)[band][idx]
        r = synth.synthesize(inst.task_spec, inst.reference_ops, time_budget_s=time_budget, seed=seed)
        return {"prompt": inst.prompt, "chosen": r["opstream"], "rejected": inst.reference_ops,
                "chosen_cost": r.get("cost"), "rejected_cost": r.get("ref_cost"),
                "ratio": r.get("cost_ratio"), "family": getattr(inst, "family", "?"),
                "band": band, "optimal": bool(r.get("optimal"))}
    except Exception as e:
        return {"error": str(e)[:100]}


@app.function(image=img, cpu=8.0, timeout=3600)
def gen_shard(seeds, time_budget, npf):
    from concurrent.futures import ProcessPoolExecutor
    work, seen = [], set()
    for s in seeds:
        try:
            cur = _curric(s, npf)
        except Exception:
            continue
        for band, insts in cur.items():
            for i, inst in enumerate(insts):
                if inst.prompt in seen:
                    continue
                seen.add(inst.prompt)
                work.append((s, band, i, time_budget, npf))
    out = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for rec in ex.map(_synth_one, work, chunksize=8):
            if rec and "error" not in rec:
                out.append(rec)
    return out


@app.local_entrypoint()
def main(n_seeds: int = 4000, shards: int = 80, time_budget: float = 1.5, n_per_family: int = 1,
         out_sft: str = "data/sft_v2.jsonl", out_dpo: str = "data/dpo_v2.jsonl"):
    SYSTEM = open(PROXY_DIR / "system_prompt.txt").read().strip()
    seeds = list(range(1, n_seeds + 1))  # training seeds (disjoint from eval seeds 60000+)
    buckets = [seeds[i::shards] for i in range(shards)]
    print(f"[gen] {n_seeds} seeds x npf={n_per_family} over {shards} shards, budget={time_budget}s/task")

    records, seen = [], set()
    for shard_recs in gen_shard.starmap([(b, time_budget, n_per_family) for b in buckets]):
        for r in shard_recs:
            if r["prompt"] in seen:
                continue
            seen.add(r["prompt"])
            records.append(r)
    print(f"[gen] {len(records)} distinct optimized instances")

    root = HERE.parent
    with open(root / out_sft, "w") as f:
        for r in records:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["chosen"]}]}) + "\n")
    n_dpo = 0
    with open(root / out_dpo, "w") as f:
        for r in records:
            if r.get("rejected") and r.get("chosen") and r["chosen"].strip() != r["rejected"].strip() \
               and (r.get("chosen_cost") or 1e18) < (r.get("rejected_cost") or 0) - 1e-9:
                f.write(json.dumps({
                    "prompt": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": r["prompt"]}],
                    "chosen": [{"role": "assistant", "content": r["chosen"]}],
                    "rejected": [{"role": "assistant", "content": r["rejected"]}]}) + "\n")
                n_dpo += 1
    ratios = sorted(r["ratio"] for r in records if r.get("ratio"))
    med = ratios[len(ratios) // 2] if ratios else None
    print(f"[gen] wrote {len(records)} SFT -> {out_sft} ; {n_dpo} DPO pairs -> {out_dpo}")
    print(f"[gen] median cost ratio {med}; optimal "
          f"{sum(r.get('optimal') for r in records)}/{len(records)}")
