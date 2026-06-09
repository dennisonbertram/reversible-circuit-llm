"""flywheel_harvest.py — the HARVESTER for the expert-iteration flywheel.

Drive a TRAINED model through the state-externalizing ToolEnv at scale over TRAINING tasks
(seeds in [1, 50000), DISJOINT from the held-out eval seeds >= 60000), and SAVE every
verifier-SOLVED multi-turn trajectory as new SFT data. The model's OWN solutions become the
next round's SFT set — the engine of a self-improving loop.

This file is `tooleval_modal.py` with three changes:
  (a) it samples TRAINING gf2_linear tasks (fresh invertible matrices, training-namespace seeds),
  (b) it RECORDS the successful trajectory in the tooltrace chat format (train == eval identical
      to the SFT traces emitted by proxy/tooltrace_gen.py), and
  (c) it runs at scale, keeping the CHEAPEST (then fewest-gate) solving trajectory per task.

The trace format is reused verbatim from tooltrace_gen / tooluse_solve so harvested traces are
byte-identical in framing to the hand-built SFT traces:
  messages[0] = {"role":"system",  "content": TOOL_SYSTEM}
  messages[1] = {"role":"user",    "content": INTRO_PREFIX + render()}     # first turn
  per played gate g_t (state BEFORE the gate):
     {"role":"assistant","content": g_t}                                   # ONE op line
     {"role":"user",     "content": render() + REPLY_SUFFIX}               # next state
  the trailing user turn after the SOLVING gate is dropped.

Output (artifacts volume):
  /artifacts/<out_name>.jsonl        one {"messages":[...]} per solved task
  /artifacts/<out_name>.stats.json   per-band attempted/solved/solve_rate + medians

Smoke:
  ECDSA_BUILD_HEAVY=1 modal run train/flywheel_harvest.py \
      --model-dir /artifacts/qwen-tool-v1-merged \
      --n-per-band 8 --bands "3,4,5" --max-steps 30 --n-restarts 2

Scale (later, e.g. 7B):
  ECDSA_BUILD_HEAVY=1 modal run train/flywheel_harvest.py \
      --model-dir /artifacts/qwen3-8b-tool-v1-merged \
      --n-per-band 400 --bands "3,4,5,6,7" --gpu L40S
"""
import os
from pathlib import Path
import modal

app = modal.App("ecdsa-flywheel-harvest")
HERE = Path(__file__).parent.resolve()
PROXY_DIR = (HERE.parent / "proxy").resolve()
EVAL_DIR = (HERE.parent / "eval").resolve()
BUILD_HEAVY = os.environ.get("ECDSA_BUILD_HEAVY", "0") == "1"

img = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
       .entrypoint([]).apt_install("git")
       .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("transformers", "accelerate", "numpy", "safetensors")
       .env({"ECDSA_BUILD_HEAVY": "1", "OLLAMA_HOST": "http://127.0.0.1:11434",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
       .add_local_dir(str(PROXY_DIR), remote_path="/root/proxy")
       .add_local_dir(str(EVAL_DIR), remote_path="/root/eval")) if BUILD_HEAVY \
    else modal.Image.from_registry("python:3.12-slim")

artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Training-seed namespace (DISJOINT from eval seeds >= 60000 and held-out fixed seeds).
_SEED_LO = 1
_SEED_HI = 50000

# Map a harvest "band width" n -> the curriculum band label (B1..B5+) for stats/labels.
_N_TO_BAND = {2: "B0", 3: "B1", 4: "B2", 5: "B3", 6: "B4", 7: "B5", 8: "B6"}


@app.function(image=img, gpu="A10G", timeout=24 * 3600,
              volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache})
def harvest(model_dir: str,
            bands: str = "3,4,5,6,7",
            n_per_band: int = 60,
            n_restarts: int = 4,
            max_steps: int = 40,
            temperature: float = 0.7,
            out_name: str = "flywheel_harvest_r0",
            batch_size: int = 16,
            seed_base: int = 1,
            progress_patience: int = 10):
    """Drive `model_dir` through ToolEnv over TRAINING gf2_linear tasks; harvest solved traces.

    bands        comma-separated register widths n to harvest (e.g. "3,4,5,6,7" -> B1..B5+).
    n_per_band   number of distinct TRAINING tasks per band.
    n_restarts   best-of-N sampled rollouts per task (keep cheapest solving trajectory).
    max_steps    max accepted gates per rollout (turn budget = 2*max_steps, like tooleval).
    temperature  sampling temperature for the rollouts.
    out_name     /artifacts/<out_name>.jsonl + .stats.json.
    batch_size   tasks driven concurrently per generation call (throughput; verifier ~free).
    seed_base    starting training seed cursor (lets successive rounds use fresh tasks).
    """
    if not BUILD_HEAVY:
        raise RuntimeError("Relaunch with ECDSA_BUILD_HEAVY=1.")
    import sys, json, random, time
    from collections import defaultdict
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/root/proxy"); sys.path.insert(0, "/root/eval")
    import tasks
    from tooluse import ToolEnv

    # Reuse the EXACT eval-driver protocol strings so harvested traces == SFT traces.
    try:
        from tooluse_solve import TOOL_SYSTEM, extract_one_op
    except Exception:
        from tooltrace_gen import TOOL_SYSTEM  # type: ignore
        import re
        _OP_HEAD = re.compile(r"^\s*(X|Z|CX|CZ|CCX|CCZ|SWAP|NEG)\b", re.IGNORECASE)

        def extract_one_op(text):
            t = text
            fenced = re.findall(r"```[a-zA-Z0-9_]*\n(.*?)```", t, re.S)
            if fenced:
                t = fenced[0]
            for raw in t.replace(";", "\n").splitlines():
                s = raw.strip().strip("`").strip()
                if not s:
                    continue
                if s.upper() == "UNDO":
                    return "UNDO"
                m = _OP_HEAD.match(s)
                if m:
                    toks = s.split()
                    kept = [toks[0].upper()]
                    for tk in toks[1:]:
                        if tk.startswith("q") and tk[1:].isdigit():
                            kept.append(tk)
                        elif tk.startswith("b") and tk[1:].isdigit():
                            kept.append(tk)
                        elif tk in ("if",):
                            kept.append(tk)
                        else:
                            break
                    return " ".join(kept)
            return ""

    # First-turn framing + per-turn suffix (verbatim from tooltrace_gen / tooluse_solve).
    INTRO_PREFIX = "Here is the task. Reply with ONE op line each turn to drive mismatches to 0.\n\n"
    REPLY_SUFFIX = "\n\nReply with ONE op line."

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # left-pad so batched generation slices new tokens cleanly off the right edge.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()

    # ---- build the TRAINING task set: fresh invertible gf2 matrices per band/width. ----
    widths = [int(b) for b in str(bands).split(",") if b.strip()]
    # tasks[i] = dict(env, msgs, n_steps, done, best_messages, best_gates, best_cost, ...)
    task_list = []
    seed = max(_SEED_LO, int(seed_base))
    for n in widths:
        band = _N_TO_BAND.get(n, f"n{n}")
        c = 0
        seen = set()
        while c < n_per_band and seed < _SEED_HI:
            rng = random.Random(seed); seed += 1
            try:
                M = tasks._rand_invertible_gf2(n, rng)
            except Exception:
                continue
            # skip the degenerate identity (already-solved; no single-step lesson).
            if M == [1 << i for i in range(n)]:
                continue
            key = (n, tuple(M))
            if key in seen:
                continue
            seen.add(key)
            inst = tasks.build_gf2_linear(n, M)
            inst.band = band
            spec = dict(inst.task_spec)
            spec["family"] = "gf2_linear"
            task_list.append({
                "n": n, "band": band, "M": list(M), "spec": spec,
                "best_messages": None, "best_cost": None, "best_gates": None,
            })
            c += 1
    print(f"[harvest] built {len(task_list)} training tasks over widths {widths} "
          f"(n_per_band={n_per_band}); restarts={n_restarts} max_steps={max_steps} "
          f"temp={temperature} batch={batch_size}", flush=True)

    @torch.no_grad()
    def gen_batch(batch_msgs):
        """One generation call over a list of message-lists. Returns list of decoded replies."""
        enc = tok.apply_chat_template(
            batch_msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=24, do_sample=True,
                             temperature=temperature, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True) for g in gen]

    by_band = defaultdict(lambda: {"attempted": 0, "solved": 0})
    for t in task_list:
        by_band[t["band"]]["attempted"] += 1

    t0 = time.time()
    # ---- best-of-N restarts: each restart drives ALL tasks (batched) to done/budget. ----
    for r in range(n_restarts):
        # active rollouts: one per task NOT yet solved on a PRIOR restart? No — best-of-N means
        # we try every task each restart and keep the cheapest solve, so include all tasks. To
        # save compute we DROP a task from later restarts only once it has a solve (more solves
        # rarely beat an existing 0-Toffoli gf2 solve on cost; the fewest-gate tiebreak is the
        # only gain, which we still allow by keeping unsolved + a light re-try of solved cheap).
        active = []
        for ti, t in enumerate(task_list):
            # Only (re)try UNSOLVED tasks each restart. Re-running solved tasks to shave a gate is
            # a marginal gain that ~doubles harvest cost, so we skip it (the first solve is kept).
            if t["best_messages"] is None:
                env = ToolEnv(t["spec"])
                if env.done:
                    # trivially solved empty circuit: nothing to harvest (no lesson). Mark solved
                    # with an empty trajectory so stats count it, but do NOT write a trace.
                    t["_trivial"] = True
                    continue
                msgs = [{"role": "system", "content": TOOL_SYSTEM},
                        {"role": "user", "content": INTRO_PREFIX + env.render()}]
                active.append({
                    "ti": ti, "env": env, "msgs": msgs,
                    "played": [],          # the (assistant_gate, pre_state_user) turns actually played
                    "first_user": msgs[1]["content"],
                    "steps": 0, "turn": 0, "done": False,
                    # progress signal for early-drop: lowest n_mismatch seen + turns since improved.
                    "best_nm": env.n_mismatch, "stale": 0,
                })

        turn_budget = max_steps * 2 + 4
        step = 0
        while active and step < turn_budget:
            step += 1
            # process in mini-batches for throughput.
            for b0 in range(0, len(active), batch_size):
                chunk = active[b0:b0 + batch_size]
                replies = gen_batch([a["msgs"] for a in chunk])
                for a, reply in zip(chunk, replies):
                    op = extract_one_op(reply)
                    if op == "" or op == "UNDO":
                        # no usable op (or UNDO, which we don't record in harvested traces):
                        # append a nudge user turn so the model can recover; do not record a gate.
                        a["msgs"].append({"role": "assistant", "content": reply.strip()[:200]})
                        a["msgs"].append({"role": "user", "content":
                                          "Reply with EXACTLY one op line like `CX q0 q2`.\n\n"
                                          + a["env"].render() + REPLY_SUFFIX})
                    else:
                        res = a["env"].step(op)
                        a["msgs"].append({"role": "assistant", "content": op})
                        if res["error"]:
                            a["msgs"].append({"role": "user", "content":
                                              f"REJECTED: {res['error']}\n\n" + a["env"].render()
                                              + REPLY_SUFFIX})
                        else:
                            a["steps"] += 1
                            # RECORD the played gate (state BEFORE the gate is the prior user turn).
                            a["played"].append(op)
                            # progress signal: track lowest n_mismatch; reset/grow the stale counter.
                            nm = res.get("n_mismatch", a["best_nm"])
                            if nm < a["best_nm"]:
                                a["best_nm"] = nm; a["stale"] = 0
                            else:
                                a["stale"] += 1
                            if res["done"]:
                                a["done"] = True
                            else:
                                a["msgs"].append({"role": "user", "content":
                                                  a["env"].render() + REPLY_SUFFIX})
                    # bound the live context window. The render is MARKOV (the current residual
                    # rows fully specify the remaining problem), so a short window suffices and
                    # slashes prefill cost — the dominant harvest expense at n=6. The SAVED trace
                    # is rebuilt clean by _record_solution, so truncation here never corrupts data.
                    if len(a["msgs"]) > 10:
                        a["msgs"] = [a["msgs"][0]] + a["msgs"][-8:]

            # harvest finished rollouts; prune them + budget-exhausted ones from `active`.
            still = []
            for a in active:
                if a["done"]:
                    _record_solution(task_list[a["ti"]], a, INTRO_PREFIX, REPLY_SUFFIX)
                elif a["steps"] >= max_steps or a["stale"] >= progress_patience:
                    pass  # budget spent OR stuck (no residual progress in `patience` turns); drop.
                else:
                    still.append(a)
            active = still
        # end while: any rollout left in `active` ran out of turn_budget -> dropped.
        n_solved_now = sum(1 for t in task_list
                           if t["best_messages"] is not None or t.get("_trivial"))
        print(f"[harvest] restart {r+1}/{n_restarts} done: "
              f"{n_solved_now}/{len(task_list)} tasks solved so far "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- tally + write outputs. ----
    kept = []
    gates_kept = []
    costs_kept = []
    for t in task_list:
        solved = t["best_messages"] is not None or t.get("_trivial")
        if solved:
            by_band[t["band"]]["solved"] += 1
        if t["best_messages"] is not None:
            kept.append({"messages": t["best_messages"]})
            gates_kept.append(len(t["best_gates"]))
            costs_kept.append(float(t["best_cost"]))

    def _median(xs):
        if not xs:
            return None
        s = sorted(xs); k = len(s)
        return float(s[k // 2]) if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2.0

    out_path = f"/artifacts/{out_name}.jsonl"
    with open(out_path, "w") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")

    stats = {
        "model_dir": model_dir,
        "out_path": out_path,
        "bands": widths,
        "n_per_band": n_per_band,
        "n_restarts": n_restarts,
        "max_steps": max_steps,
        "temperature": temperature,
        "seed_base": int(seed_base),
        "seed_end": seed,
        "total_attempted": sum(v["attempted"] for v in by_band.values()),
        "total_solved": sum(v["solved"] for v in by_band.values()),
        "total_harvested": len(kept),
        "median_gates": _median(gates_kept),
        "median_cost": _median(costs_kept),
        "elapsed_s": round(time.time() - t0, 1),
        "by_band": {
            b: {**v, "solve_rate": round(v["solved"] / max(1, v["attempted"]), 3)}
            for b, v in sorted(by_band.items())
        },
    }
    with open(f"/artifacts/{out_name}.stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    artifacts.commit()

    print("HARVEST_STATS", json.dumps(stats), flush=True)
    return stats


def _record_solution(task, rollout, intro_prefix, reply_suffix):
    """Re-replay the played gates through a FRESH ToolEnv to build the canonical trace + cost,
    and keep it iff it beats the current best (lower Toffoli cost; ties broken by fewer gates).

    We REBUILD from scratch rather than trusting the live `msgs` so the recorded trace is exactly
    the tooltrace format (no nudges/rejections), every assistant turn is a single canonical op,
    and we re-VERIFY the solution drives to done==True (the whole point of harvesting).
    """
    import sys
    sys.path.insert(0, "/root/proxy")
    from tooluse import ToolEnv

    gates = list(rollout["played"])
    if not gates:
        return
    env = ToolEnv(task["spec"])
    if env.done:
        return  # trivial; nothing to harvest
    messages = [rollout["msgs"][0]]   # system (== TOOL_SYSTEM)
    first_user = intro_prefix + env.render()
    ok = True
    for idx, g in enumerate(gates):
        if idx == 0:
            messages.append({"role": "user", "content": first_user})
        else:
            messages.append({"role": "user", "content": env.render() + reply_suffix})
        messages.append({"role": "assistant", "content": g})
        res = env.step(g)
        if res["error"]:
            ok = False
            break
        if res["done"]:
            # drop any trailing gates the rollout emitted after solving (shouldn't happen).
            break
    if not ok or not env.done:
        return  # could not reproduce a clean solving trace; skip.

    cost = float(env.cost)
    n_gates = len(env.emitted)
    cur = task["best_cost"]
    cur_g = len(task["best_gates"]) if task["best_gates"] is not None else None
    # keep cheapest cost; tiebreak on fewer gates (efficient play).
    better = (cur is None) or (cost < cur) or (cost == cur and n_gates < cur_g)
    if better:
        task["best_messages"] = messages
        task["best_cost"] = cost
        task["best_gates"] = list(env.emitted)


@app.function(image=img, timeout=3600, volumes={"/artifacts": artifacts})
def reverify(out_name: str = "flywheel_harvest_r0", sample: int = 0):
    """Re-drive harvested traces through a FRESH ToolEnv to confirm they reach done==True.

    Each harvested trace only stores `messages`; we recover the verifying GF(2) target by parsing
    the `target y{i} = x.. ^ x..` rows out of the FIRST user turn's render, rebuild the exact
    task via tasks.build_gf2_linear, then feed the trace's assistant gates in order and assert the
    env reaches done. Also checks the system prompt matches the eval driver and every assistant
    turn is a single parseable op. Returns a summary; raises if any trace fails to re-drive.

    sample=0 checks ALL traces; otherwise a random `sample` of them.
    """
    if not BUILD_HEAVY:
        raise RuntimeError("Relaunch with ECDSA_BUILD_HEAVY=1.")
    import sys, json, re, random
    sys.path.insert(0, "/root/proxy"); sys.path.insert(0, "/root/eval")
    import tasks
    from tooluse import ToolEnv
    try:
        from tooluse_solve import TOOL_SYSTEM, extract_one_op
    except Exception:
        from tooltrace_gen import TOOL_SYSTEM  # type: ignore
        extract_one_op = None

    path = f"/artifacts/{out_name}.jsonl"
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    idxs = list(range(len(rows)))
    if sample and sample < len(rows):
        idxs = random.Random(17).sample(idxs, sample)

    _row_re = re.compile(r"target y(\d+)\s*=\s*([x0-9\s\^]+)")

    def _target_M_from_render(text, n_hint=None):
        """Parse 'target y{i} = x0 ^ x2' rows from a render into the M row-bitmask list."""
        rows_by_i = {}
        for m in _row_re.finditer(text):
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
        return [rows_by_i.get(i, 0) for i in range(n)]

    n_checked = 0
    n_done = 0
    n_sys_ok = 0
    n_single_ok = 0
    failures = []
    for ix in idxs:
        msgs = rows[ix]["messages"]
        n_checked += 1
        if msgs[0]["role"] == "system" and msgs[0]["content"] == TOOL_SYSTEM:
            n_sys_ok += 1
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        M = _target_M_from_render(first_user)
        ops = [m["content"] for m in msgs if m["role"] == "assistant"]
        single_ok = all(("\n" not in op and ";" not in op
                         and (extract_one_op is None or extract_one_op(op) == op)) for op in ops)
        if single_ok:
            n_single_ok += 1
        if M is None:
            failures.append({"idx": ix, "reason": "could not parse target rows"})
            continue
        spec = dict(tasks.build_gf2_linear(len(M), M).task_spec)
        spec["family"] = "gf2_linear"
        env = ToolEnv(spec)
        err = None
        for op in ops:
            r = env.step(op)
            if r["error"]:
                err = r["error"]; break
        if err is None and env.done:
            n_done += 1
        else:
            failures.append({"idx": ix, "reason": err or "not done", "n_ops": len(ops)})

    summary = {
        "out_name": out_name,
        "checked": n_checked,
        "redrive_done": n_done,
        "redrive_fraction": round(n_done / max(1, n_checked), 4),
        "system_prompt_match": n_sys_ok,
        "single_op_turns": n_single_ok,
        "failures": failures[:10],
    }
    print("REVERIFY_SUMMARY", json.dumps(summary), flush=True)
    assert n_done == n_checked, f"re-drive failures: {failures[:10]}"
    return summary


@app.local_entrypoint()
def main(model_dir: str = "/artifacts/qwen-tool-v1-merged",
         bands: str = "3,4,5,6,7",
         n_per_band: int = 60,
         n_restarts: int = 4,
         max_steps: int = 40,
         temperature: float = 0.7,
         out_name: str = "flywheel_harvest_r0",
         batch_size: int = 16,
         seed_base: int = 1,
         progress_patience: int = 10,
         gpu: str = "A10G"):
    # 7B/8B+ policies won't fit the A10G (22GB) -> route to a bigger card at call time.
    fn = harvest.with_options(gpu=gpu) if gpu and gpu.upper() != "A10G" else harvest
    stats = fn.remote(
        model_dir=model_dir, bands=bands, n_per_band=n_per_band, n_restarts=n_restarts,
        max_steps=max_steps, temperature=temperature, out_name=out_name,
        batch_size=batch_size, seed_base=seed_base, progress_patience=progress_patience)
    print("=" * 72)
    print("FLYWHEEL HARVEST COMPLETE")
    print("=" * 72)
    print(f"  model            : {stats['model_dir']}")
    print(f"  out file         : {stats['out_path']}")
    print(f"  harvested traces : {stats['total_harvested']}  "
          f"(solved {stats['total_solved']}/{stats['total_attempted']})")
    print(f"  median gates     : {stats['median_gates']}   median cost: {stats['median_cost']}")
    print(f"  elapsed          : {stats['elapsed_s']}s")
    print("  per-band:")
    for b, v in stats["by_band"].items():
        print(f"    {b}: solved {v['solved']}/{v['attempted']}  "
              f"solve_rate={v['solve_rate']}")
    return stats
