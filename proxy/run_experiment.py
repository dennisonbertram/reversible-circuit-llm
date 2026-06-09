"""
run_experiment.py — capable-policy-drives-the-tool experiment on held-out gf2_linear tasks.

The policy reduces the tool-reported TARGET vs current GF(2) rows by Gauss-Jordan and emits
the realizing circuit ONE gate at a time through ToolEnv.step(). After EVERY gate it RE-READS
the tool's live current rows and asserts they match the policy's prediction — the tool, not
the policy, owns the cumulative state; the policy only ever reacts to / verifies against it.

Records per task: solved (verify valid=True), #steps, Toffoli cost (gf2 => 0), opstream.
"""
from __future__ import annotations
import random
import proxy_env as pe
import tasks
from tooluse import ToolEnv
from policy_driver import tool_rows, _plan, gate_text


def make_task(n, M):
    inst = tasks.build_gf2_linear(n, M)
    inst.family = "gf2_linear"
    spec = dict(inst.task_spec)
    spec["family"] = "gf2_linear"
    return inst, spec


def _sim_gate(rows, g):
    """Pure GF(2) prediction of one gate's effect on the current rows (policy's own model)."""
    r = list(rows)
    if g[0] == "CX":
        r[g[2]] ^= r[g[1]]      # CX q{src} q{dst}: row_dst ^= row_src
    else:
        r[g[1]], r[g[2]] = r[g[2]], r[g[1]]
    return r


def drive(env: ToolEnv, max_steps=80, transcript=False):
    """Drive the env gate-by-gate. Returns (solved, n_steps, log).

    Step 0: READ live tool rows (current=identity, target=M). Compute the elimination plan
    that realizes target from the live current. Then execute it ONE gate at a time, and after
    each env.step() RE-READ the tool's live rows and verify they equal the policy's prediction.
    If the tool ever disagrees, the policy re-reads and re-plans from the live state (robust to
    any surprise) — this is what makes it state-reactive rather than blind replay.
    """
    n = env.n_in
    log = []
    steps = 0
    cur, tgt = tool_rows(env)                 # READ live tool state (start)
    plan = _plan(cur, tgt, n)                 # policy plan from live rows
    pi = 0
    while not env.done and steps < max_steps:
        if pi >= len(plan):
            # plan exhausted but not done: re-read live state and re-plan (defensive).
            cur, tgt = tool_rows(env)
            if cur == tgt:
                break
            plan = _plan(cur, tgt, n)
            pi = 0
            if not plan:
                break
        g = plan[pi]
        pi += 1
        gt = gate_text(g)
        predicted = _sim_gate(cur, g)         # policy predicts new current
        before_render = env.render() if transcript else None
        res = env.step(gt)                    # APPLY one gate through the tool
        steps += 1
        if res["error"]:
            log.append(("ERROR", gt, res["error"]))
            env.undo()
            break
        cur, _tgt = tool_rows(env)            # RE-READ live tool state
        if cur != predicted:
            # tool disagreed with policy model -> trust the tool, re-plan from live rows.
            plan = _plan(cur, tgt, n)
            pi = 0
        if transcript:
            log.append((before_render, gt, res["render"], res["done"], res["n_mismatch"]))
    return env.done, steps, log


def main(capture_transcript_for=None):
    B2 = [(60000, [1, 12, 15, 11]), (60002, [4, 6, 15, 12]), (60005, [11, 6, 2, 12])]
    B3 = []
    for s in (60000, 60003, 60007):
        rng = random.Random(s)
        M = tasks._rand_invertible_gf2(5, rng)
        B3.append((s, M))

    results = []
    transcripts = {}
    print("=" * 78)
    print("CAPABLE POLICY + STATE-EXTERNALIZING TOOL — held-out gf2_linear tasks")
    print("=" * 78)
    for label, n, items in (("B2(4-bit)", 4, B2), ("B3(5-bit)", 5, B3)):
        for seed, M in items:
            inst, spec = make_task(n, M)
            env = ToolEnv(spec)
            ref_cost = inst.task_spec["reference_cost"]
            want_tr = (capture_transcript_for == (label, seed))
            solved, steps, log = drive(env, max_steps=80, transcript=want_tr)
            v = pe.verify(env.solved_opstream(), inst.task_spec)
            assert v["valid"] == solved, (v, solved)
            if solved:
                assert v["cost"] == env.cost
            results.append({
                "band": label, "seed": seed, "n": n, "M": M,
                "solved": solved, "steps": steps,
                "toffoli": env.cost, "ref_cost": ref_cost,
                "verify_valid": v["valid"], "verify_cost": v["cost"],
                "opstream": env.solved_opstream().replace("\n", "; "),
            })
            if want_tr:
                transcripts[(label, seed)] = log
            print(f"[{label} seed={seed}] M={M}  solved={solved}  steps={steps}  "
                  f"Toffoli_cost={env.cost:.0f}  ref_cost={ref_cost:.0f}  verify_valid={v['valid']}")
            print(f"             opstream: {env.solved_opstream().replace(chr(10), '; ')}")

    n_solved = sum(r["solved"] for r in results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_steps_solved = (sum(r["steps"] for r in results if r["solved"]) / max(1, n_solved))
    print("=" * 78)
    print(f"SOLVED {n_solved}/{len(results)} | avg steps(all)={avg_steps:.1f} "
          f"avg steps(solved)={avg_steps_solved:.1f} | Toffoli costs: {[r['toffoli'] for r in results]}")
    print("=" * 78)
    return results, transcripts


if __name__ == "__main__":
    main()
