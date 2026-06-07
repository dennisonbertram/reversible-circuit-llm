#!/usr/bin/env python3
"""
build_curriculum.py — validate every generated reference instance through
proxy_env.verify, emit curriculum_manifest.json + sample_task.txt, and run a >=50-instance
validation pass spanning all families/bands. Use /usr/bin/python3.

Run:  /usr/bin/python3 build_curriculum.py
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

import proxy_env as pe
import tasks


HERE = os.path.dirname(os.path.abspath(__file__))


def _verify(inst: tasks.Instance) -> Dict:
    rep = pe.verify(inst.reference_ops, inst.task_spec)
    return {
        "valid": rep["valid"],
        "reason": rep["reason"],
        "toffoli": rep["toffoli"],
        "peak_width": rep["peak_width"],
        "cost": rep["cost"],
        "frac_correct": rep["frac_correct"],
        "frac_ancilla_clean": rep["frac_ancilla_clean"],
    }


def main() -> None:
    rng = random.Random(20240606)

    # -------- 1. validation pass over >= 50 instances spanning all families/bands --------
    validated = 0
    failed = 0
    fail_detail: List[Dict] = []
    per_band: Dict[str, List[tasks.Instance]] = {}

    # sample several instances per family per band to exceed 50 total.
    N_PER_FAMILY = 3
    for band in tasks.BANDS:
        insts = tasks.sample_band(band, rng, n_per_family=N_PER_FAMILY)
        per_band[band] = insts

    # also include the held-out tasks in the validation pass (validated, never trained).
    heldout = tasks.heldout_tasks()

    all_insts: List[tasks.Instance] = []
    for band in tasks.BANDS:
        all_insts.extend(per_band[band])
    all_insts.extend(heldout)

    results = []
    for inst in all_insts:
        r = _verify(inst)
        ok = bool(r["valid"] and r["frac_correct"] == 1.0
                  and r["frac_ancilla_clean"] == 1.0)
        if ok:
            validated += 1
        else:
            failed += 1
            fail_detail.append({"family": inst.family, "band": inst.band,
                                "params": inst.params, **r})
        results.append((inst, r))

    # -------- 2. manifest: per-band families + one example instance each --------
    manifest = {
        "generated_by": "tasks.py + build_curriculum.py",
        "verifier": "proxy_env.verify (full 4-gate: correctness/reversibility/phase/ancilla)",
        "python": "/usr/bin/python3",
        "validation_summary": {
            "instances_validated": validated,
            "instances_failed": failed,
            "n_per_family_sampled": N_PER_FAMILY,
            "total_instances": len(all_insts),
            "all_pass": failed == 0,
        },
        "families": tasks.FAMILIES,
        "bands": {},
        "heldout": [],
        "failures": fail_detail,
    }

    # one representative (the first) example per family per band
    for band in tasks.BANDS:
        plan = tasks._BAND_PLAN[band]
        examples = []
        seen_fam = set()
        for inst in per_band[band]:
            if inst.family in seen_fam:
                continue
            seen_fam.add(inst.family)
            r = _verify(inst)
            examples.append({
                "family": inst.family,
                "params": inst.params,
                "in_qubits": inst.task_spec["in_qubits"],
                "out_qubits": inst.task_spec["out_qubits"],
                "n_in": inst.task_spec["n_in"],
                "width": inst.task_spec["width"],
                "max_width": inst.task_spec["max_width"],
                "reference_toffoli": r["toffoli"],
                "reference_peak_width": r["peak_width"],
                "reference_cost": r["cost"],
                "reference_n_ops": len([l for l in inst.reference_ops.splitlines()
                                        if l.strip() and not l.strip().startswith("#")]),
                "validation": {"valid": r["valid"], "reason": r["reason"],
                               "frac_correct": r["frac_correct"],
                               "frac_ancilla_clean": r["frac_ancilla_clean"]},
            })
        manifest["bands"][band] = {
            "n_in": plan["n_in"],
            "families": plan["families"],
            "example_instances": examples,
        }

    for inst in heldout:
        r = _verify(inst)
        manifest["heldout"].append({
            "family": inst.family,
            "params": inst.params,
            "in_qubits": inst.task_spec["in_qubits"],
            "out_qubits": inst.task_spec["out_qubits"],
            "n_in": inst.task_spec["n_in"],
            "width": inst.task_spec["width"],
            "reference_toffoli": r["toffoli"],
            "reference_peak_width": r["peak_width"],
            "reference_cost": r["cost"],
            "validation": {"valid": r["valid"], "reason": r["reason"],
                           "frac_correct": r["frac_correct"],
                           "frac_ancilla_clean": r["frac_ancilla_clean"]},
            "note": "HELD-OUT generalization probe — validated, NEVER used in training.",
        })

    manifest_path = os.path.join(HERE, "curriculum_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # -------- 3. sample_task.txt: one rendered prompt + reference op-stream + verdict --------
    # pick a representative mid-curriculum instance (B3 mod_mult) for the sample.
    rng2 = random.Random(7)
    sample = tasks.sample_instance("mod_mult", "B3", rng2)
    srep = pe.verify(sample.reference_ops, sample.task_spec)
    sample_path = os.path.join(HERE, "sample_task.txt")
    with open(sample_path, "w") as fh:
        fh.write("=" * 78 + "\n")
        fh.write("SAMPLE PROXY TASK — rendered prompt + reference op-stream + verifier verdict\n")
        fh.write("=" * 78 + "\n\n")
        fh.write("----- MODEL-FACING PROMPT -----\n")
        fh.write(sample.prompt + "\n\n")
        fh.write("----- REFERENCE OP-STREAM (baseline to beat) -----\n")
        fh.write(sample.reference_ops + "\n\n")
        fh.write("----- VERIFIER VERDICT (proxy_env.verify) -----\n")
        fh.write(json.dumps({
            "valid": srep["valid"],
            "reason": srep["reason"],
            "toffoli": srep["toffoli"],
            "peak_width": srep["peak_width"],
            "cost": srep["cost"],
            "vs_ref_pct": srep["vs_ref_pct"],
            "frac_correct": srep["frac_correct"],
            "frac_ancilla_clean": srep["frac_ancilla_clean"],
        }, indent=2) + "\n")

    # -------- 4. console report --------
    print("=" * 70)
    print("VALIDATION PASS (all families x all bands + held-out)")
    print("=" * 70)
    for inst, r in results:
        tag = "OK " if (r["valid"] and r["frac_correct"] == 1.0
                        and r["frac_ancilla_clean"] == 1.0) else "FAIL"
        print(f"  [{tag}] {inst.band or 'HELDOUT':7s} {inst.family:18s} "
              f"reason={r['reason']:14s} tof={r['toffoli']} pw={r['peak_width']} "
              f"cost={r['cost']}")
    print("-" * 70)
    print(f"  instances_validated = {validated}")
    print(f"  instances_failed    = {failed}")
    print(f"  total               = {len(all_insts)}")
    print(f"  manifest  -> {manifest_path}")
    print(f"  sample    -> {sample_path}")


if __name__ == "__main__":
    main()
