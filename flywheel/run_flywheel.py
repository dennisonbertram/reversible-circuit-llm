"""run_flywheel.py — the expert-iteration flywheel DRIVER (Tier 1).

One iteration = harvest -> combine -> SFT(fresh base on cumulative) -> merge -> eval.
Repeat until a stop condition trips:
  * SUCCESS  : overall_solve_rate >= --target-overall AND B4 solve_rate > 0
  * PLATEAU  : overall gain < --plateau (default 0.03) for 2 consecutive iterations
  * BUDGET   : --iters reached

Expert iteration (STaR): each round retrains a FRESH base model on the cumulative replay buffer
(seed expert demos + every harvested solve so far, deduped per task keeping the cheapest play).
The HARVEST source is always the LATEST merged model, so the policy that generates next round's
data is the best one so far. Held-out eval seeds (>=60000) never enter the training pool.

Each external step is a blocking `modal run`; we parse a result marker from its output and retry
once on transient HF 503 / network blips. All state is logged to flywheel/<tag>_history.json so a
restart can be reasoned about.

Example (fast mechanism-validation on the 1.5B):
  python flywheel/run_flywheel.py --tag fly15 \
     --seed-merged /artifacts/qwen-tool-v2-merged \
     --base-model unsloth/Qwen2.5-Coder-1.5B-Instruct \
     --expert data/sft_tooltrace_v2.jsonl \
     --iters 3 --bands 3,4,5 --n-per-band 120 --seq 4608 --bs 4 --epochs 2

Example (success-condition attempt on the compact-8B base):
  python flywheel/run_flywheel.py --tag fly8 \
     --seed-merged /artifacts/qwen3-8b-toolbase-merged \
     --base-model Qwen/Qwen3-8B \
     --expert data/sft_tool_8base.jsonl \
     --iters 4 --bands 3,4,5,6 --n-per-band 160 --n-restarts 5 \
     --seq 6144 --bs 2 --epochs 2 --gpu-harvest L40S --eval-per-band 8
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_modal(args_list, marker, logname, retries=1):
    """Run `modal run ...`, stream to a log, return the parsed-json after `marker` or None.

    Retries once (fresh process) if the marker is absent and the output smells transient (HF 503,
    connection reset, 5xx). Raises on hard non-transient failure so the driver can stop loudly.
    """
    env = dict(os.environ, ECDSA_BUILD_HEAVY="1")
    logpath = os.path.join(ROOT, "infra", logname)
    for attempt in range(retries + 1):
        with open(logpath, "w") as logf:
            logf.write(f"# attempt {attempt} {time.ctime()}\n# " + " ".join(args_list) + "\n")
            logf.flush()
            proc = subprocess.run(args_list, cwd=ROOT, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out = proc.stdout or ""
            logf.write(out)
        found = None
        for ln in out.splitlines():
            if ln.startswith(marker):
                payload = ln[len(marker):].strip()
                try:
                    found = json.loads(payload)
                except Exception:
                    # tooleval prints a python dict (single quotes) — eval it safely-ish
                    try:
                        found = json.loads(payload.replace("'", '"'))
                    except Exception:
                        found = {"_raw": payload}
                break
        if found is not None:
            return found
        transient = bool(re.search(r"503|Service Unavailable|Connection reset|"
                                   r"Temporary failure|timed out|ReadTimeout|5\d\d Server",
                                   out))
        print(f"[driver]   marker '{marker}' not found (attempt {attempt}, "
              f"exit={proc.returncode}, transient={transient}). log: infra/{logname}",
              flush=True)
        if not transient or attempt == retries:
            tail = "\n".join(out.splitlines()[-25:])
            raise RuntimeError(f"step failed ({' '.join(args_list[:4])}...); "
                               f"marker '{marker}' absent. tail:\n{tail}")
        time.sleep(30)
    return None


def vol_get(remote, local):
    if os.path.exists(local):
        os.remove(local)
    subprocess.run(["modal", "volume", "get", "ecdsa-artifacts", remote, local],
                   cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def combine(expert, harvests, out, cap_per_band, max_tokens):
    cmd = [sys.executable, os.path.join(ROOT, "flywheel", "combine.py"),
           "--expert", expert, "--harvests", ",".join(harvests), "--out", out]
    if cap_per_band:
        cmd += ["--cap-per-band", str(cap_per_band)]
    if max_tokens:
        cmd += ["--max-tokens", str(max_tokens)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(proc.stderr, flush=True)
    for ln in proc.stdout.splitlines():
        if ln.startswith("COMBINE_STATS"):
            return json.loads(ln[len("COMBINE_STATS"):].strip())
    raise RuntimeError(f"combine failed:\n{proc.stdout}\n{proc.stderr}")


def b4_rate(eval_res):
    bb = eval_res.get("by_band", {})
    return bb.get("B4", {}).get("solve_rate", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed-merged", required=True, help="iter-0 harvest source (merged dir on volume)")
    ap.add_argument("--base-model", required=True, help="fresh SFT base (HF id) retrained each round")
    ap.add_argument("--expert", required=True, help="seed expert traces jsonl (in repo data/)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--bands", default="3,4,5")
    ap.add_argument("--n-per-band", type=int, default=120)
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--harvest-max-steps", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seq", type=int, default=4608)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-per-band", type=int, default=6)
    ap.add_argument("--gpu-harvest", default="A10G")
    ap.add_argument("--target-overall", type=float, default=0.65)
    ap.add_argument("--plateau", type=float, default=0.03)
    ap.add_argument("--cap-per-band", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--baseline-overall", type=float, default=None,
                    help="known iter-0 eval overall (skip re-eval of the seed)")
    args = ap.parse_args()

    hist_path = os.path.join(ROOT, "flywheel", f"{args.tag}_history.json")
    history = []
    if args.baseline_overall is not None:
        history.append({"iter": 0, "model": args.seed_merged,
                        "eval": {"overall_solve_rate": args.baseline_overall}, "note": "seed"})

    current = args.seed_merged
    harvests = []
    last_overall = args.baseline_overall if args.baseline_overall is not None else 0.0
    small_gains = 0

    for k in range(1, args.iters + 1):
        t_iter = time.time()
        print(f"\n{'='*72}\n[driver] ITER {k}/{args.iters}  harvest-source={current}\n{'='*72}",
              flush=True)

        # 1) HARVEST -------------------------------------------------------------
        out_name = f"{args.tag}_harvest_iter{k}"
        seed_base = 1 + (k - 1) * 12000
        hstats = run_modal(
            ["modal", "run", "train/flywheel_harvest.py",
             "--model-dir", current, "--bands", args.bands,
             "--n-per-band", str(args.n_per_band), "--n-restarts", str(args.n_restarts),
             "--max-steps", str(args.harvest_max_steps), "--temperature", str(args.temperature),
             "--out-name", out_name, "--seed-base", str(seed_base), "--gpu", args.gpu_harvest],
            marker="HARVEST_STATS", logname=f"{args.tag}_harvest_iter{k}.log", retries=1)
        print(f"[driver] harvested {hstats.get('total_harvested')} traces "
              f"(solved {hstats.get('total_solved')}/{hstats.get('total_attempted')}); "
              f"by_band={hstats.get('by_band')}", flush=True)

        # 2) DOWNLOAD harvested jsonl --------------------------------------------
        local_h = os.path.join(ROOT, "artifacts", f"{out_name}.jsonl")
        os.makedirs(os.path.dirname(local_h), exist_ok=True)
        vol_get(f"{out_name}.jsonl", local_h)
        harvests.append(local_h)

        # 3) COMBINE cumulative replay buffer ------------------------------------
        combined = os.path.join(ROOT, "data", f"{args.tag}_flywheel_iter{k}.jsonl")
        cstats = combine(os.path.join(ROOT, args.expert), harvests, combined,
                         args.cap_per_band, args.max_tokens)
        print(f"[driver] combined -> {cstats['total_kept']} traces "
              f"({cstats['unique_tasks']} unique tasks)", flush=True)

        # 4) SFT fresh base on cumulative ----------------------------------------
        run_name = f"{args.tag}-iter{k}"
        run_modal(
            ["modal", "run", "train/modal_app.py", "--stage", "sft",
             "--base-model", args.base_model, "--run-name", run_name,
             "--dataset-path", f"/root/data/{args.tag}_flywheel_iter{k}.jsonl",
             "--epochs", str(args.epochs), "--max-seq-len", str(args.seq),
             "--per-device-bs", str(args.bs)],
            marker="[sft] DONE", logname=f"{args.tag}_sft_iter{k}.log", retries=1)
        # [sft] DONE is plain text, not json -> run_modal returns {"_raw":...}; that's fine.

        # 5) MERGE ----------------------------------------------------------------
        merged = f"/artifacts/{run_name}-merged"
        run_modal(
            ["modal", "run", "train/merge_lora.py", "--base-model", args.base_model,
             "--adapter", f"/ckpts/sft/{run_name}", "--out-name", f"{run_name}-merged"],
            marker="[merge] DONE", logname=f"{args.tag}_merge_iter{k}.log", retries=2)

        # 6) EVAL -----------------------------------------------------------------
        ev = run_modal(
            ["modal", "run", "train/tooleval_modal.py", "--model-dir", merged,
             "--per-band", str(args.eval_per_band), "--max-steps", "40"],
            marker="TOOLEVAL_RESULT", logname=f"{args.tag}_eval_iter{k}.log", retries=1)
        overall = ev.get("overall_solve_rate", 0.0)
        b4 = b4_rate(ev)
        gain = overall - last_overall
        print(f"[driver] ITER {k} EVAL: overall={overall}  B4={b4}  "
              f"(gain {gain:+.3f} vs {last_overall})  by_band={ev.get('by_band')}  "
              f"[{time.time()-t_iter:.0f}s]", flush=True)

        history.append({"iter": k, "model": merged, "eval": ev,
                        "harvest": {kk: hstats.get(kk) for kk in
                                    ("total_harvested", "total_solved", "total_attempted", "by_band")},
                        "combined_total": cstats["total_kept"],
                        "gain": round(gain, 4), "elapsed_s": round(time.time() - t_iter, 1)})
        with open(hist_path, "w") as fh:
            json.dump(history, fh, indent=2)

        # 7) advance + stop checks ------------------------------------------------
        current = merged
        if overall >= args.target_overall and b4 > 0:
            print(f"[driver] ✅ SUCCESS: overall {overall} >= {args.target_overall} AND B4 {b4} > 0",
                  flush=True)
            break
        small_gains = small_gains + 1 if gain < args.plateau else 0
        last_overall = overall
        if small_gains >= 2:
            print(f"[driver] ⏹ PLATEAU: <{args.plateau} gain for 2 consecutive iters. stopping.",
                  flush=True)
            break

    print("\nFLYWHEEL_DONE " + json.dumps({"tag": args.tag, "iters_run": len(history) - (1 if args.baseline_overall is not None else 0),
                                          "final": history[-1]["eval"].get("overall_solve_rate"),
                                          "history": history}))


if __name__ == "__main__":
    main()
