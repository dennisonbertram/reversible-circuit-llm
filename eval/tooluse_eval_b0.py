#!/usr/bin/env python3
"""tooluse_eval_b0.py — the 2-bit (B0-scale) gf2_linear CONTROL for the externalization
experiment.

The main tooluse_eval probes B1/B2/B3 (3-5 bit). This control probes the SAME mechanism at the
2-bit scale that the few-shot worked example demonstrates, to isolate "does state-externalization
beat one-shot" from "can a 1.5B model read a wide residual". Same model, same no-tool baseline
(best-of-8), same tool loop.
"""
from __future__ import annotations

import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "proxy"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tasks  # noqa: E402
from tooluse_solve import tooluse_solve  # noqa: E402
from tooluse_eval import baseline_best_of_n  # noqa: E402


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ecdsa-coder-1.5b-v4"
    n_baseline = 8
    max_steps = 20

    # All invertible 2x2 GF(2) matrices that are NON-trivial (not identity) -> 5 distinct maps.
    # M rows as bitmasks over {x0,x1}. Invertible 2x2 over GF(2): GL(2,2) has 6 elements; drop I.
    candidates = [[1, 3], [3, 1], [2, 3], [3, 2], [2, 1]]  # 5 non-identity invertible maps
    insts = []
    for M in candidates:
        try:
            inst = tasks.build_gf2_linear(2, M)
            inst.family = "gf2_linear"
            inst.band = "B0"
            inst.prompt = tasks.render_prompt(inst)
            insts.append(inst)
        except Exception as e:
            print("skip", M, e)

    print(f"[b0] model={model} gf2_linear B0 (2-bit) tasks={len(insts)} n_baseline={n_baseline}")
    t0 = time.time()
    b_valid = t_valid = 0
    t_steps = []
    for i, inst in enumerate(insts):
        ref = inst.task_spec["reference_cost"]
        bret = baseline_best_of_n(model, inst, n_baseline)
        spec = dict(inst.task_spec); spec["family"] = "gf2_linear"
        tret = tooluse_solve(model, spec, inst.prompt, max_steps=max_steps, temperature=0.2)
        b_valid += int(bret["valid"])
        t_valid += int(tret["solved"])
        if tret["solved"]:
            t_steps.append(tret["steps"])
        print(f"  M={inst.params['M']} ref={ref:.0f} | baseline_valid={bret['valid']} "
              f"| tool_solved={tret['solved']} steps={tret['steps']} "
              f"ops={tret['opstream'].replace(chr(10), '; ')}")
    n = len(insts)
    print("\n" + "=" * 70)
    print(f"gf2_linear B0 (2-bit) CONTROL  model={model}")
    print(f"  NO-TOOL baseline valid_rate : {b_valid}/{n} = {b_valid/n*100:.1f}%")
    print(f"  TOOL valid_rate             : {t_valid}/{n} = {t_valid/n*100:.1f}%")
    print(f"  tool mean steps-to-solve    : "
          f"{(sum(t_steps)/len(t_steps)):.1f}" if t_steps else "  tool mean steps : n/a")
    print(f"  wall={time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
