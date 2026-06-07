#!/usr/bin/env python3
"""Real ECDSA-fail harness wrapper.

Runs the challenge's `benchmark.sh` (build_circuit -> eval_circuit) with optional
DIALOG_* env-var overrides and returns a structured verdict: validity, score,
avg Toffoli, peak qubits, and the validation failure counts. ~13s per run on this
machine; safe to fan out on Modal CPU containers for parallel knob search / eval.

Usage:
    python3 run_harness.py                                  # baseline (no overrides)
    python3 run_harness.py --set DIALOG_TAIL_NONCE=431581 --set DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS=21
    python3 run_harness.py --json                           # machine-readable only
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time

REPO = pathlib.Path(os.environ.get(
    "ECDSAFAIL_REPO",
    "/Users/dennison/develop/quantum-project/ecdsafail-challenge",
))


def _grab_int(text, pat):
    m = re.search(pat, text)
    return int(m.group(1)) if m else None


def run_harness(env_overrides=None, note="ecdsa-model eval", timeout=300):
    """Run benchmark.sh with env overrides. Returns a dict verdict."""
    env = os.environ.copy()
    env["CARGO_NET_OFFLINE"] = "true"
    cc = subprocess.run(["bash", "-lc", "command -v clang || command -v cc || command -v gcc"],
                        capture_output=True, text=True).stdout.strip()
    if cc:
        env["CC"] = cc
        env["RUSTFLAGS"] = f"-C linker={cc}"
    for k, v in (env_overrides or {}).items():
        env[str(k)] = str(v)

    # benchmark.sh wipes these itself, but be defensive so a stale file can't be misread.
    for f in ("ops.bin", "score.json"):
        p = REPO / f
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    t0 = time.time()
    try:
        proc = subprocess.run(["bash", "./benchmark.sh", "--note", note],
                              cwd=str(REPO), env=env, capture_output=True,
                              text=True, timeout=timeout)
        rc, timed_out = proc.returncode, False
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, timed_out = -1, True
        out = (e.stdout or "") + "\n" + (e.stderr or "") if isinstance(e.stdout, str) else ""

    res = {
        "wall_s": round(time.time() - t0, 2),
        "returncode": rc,
        "timed_out": timed_out,
        "env_overrides": env_overrides or {},
        "classical_mismatches": _grab_int(out, r"classical mismatches\s*:\s*(\d+)"),
        "phase_garbage_batches": _grab_int(out, r"phase-garbage batches\s*:\s*(\d+)"),
        "ancilla_garbage_batches": _grab_int(out, r"ancilla-garbage batches\s*:\s*(\d+)"),
        "shots_ok": ("shots OK" in out),
    }
    m = re.search(r"avg executed Toffoli\s*:\s*([\d.]+)", out)
    res["avg_toffoli"] = float(m.group(1)) if m else None
    m = re.search(r"qubits\s*:\s*(\d+)", out)
    res["peak_qubits"] = int(m.group(1)) if m else None

    sj = REPO / "score.json"
    res["score_json"] = None
    if sj.exists():
        try:
            res["score_json"] = json.loads(sj.read_text())
        except Exception:
            pass

    res["score"] = res["score_json"]["score"] if res["score_json"] else None
    res["valid"] = bool(
        rc == 0 and not timed_out and res["score_json"] is not None
        and res["classical_mismatches"] == 0
        and res["phase_garbage_batches"] == 0
        and res["ancilla_garbage_batches"] == 0
    )
    if not res["valid"]:
        res["tail_stdout"] = out[-2000:]
    return res


def main():
    ap = argparse.ArgumentParser(description="Run the real ECDSA-fail harness with optional knob overrides.")
    ap.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE",
                    help="env-var override, repeatable")
    ap.add_argument("--note", default="ecdsa-model eval")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true", help="print only the JSON verdict")
    a = ap.parse_args()
    overrides = {}
    for s in a.set:
        k, _, v = s.partition("=")
        overrides[k] = v
    res = run_harness(overrides, note=a.note, timeout=a.timeout)
    if a.json:
        print(json.dumps(res))
    else:
        print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["valid"] else 2)


if __name__ == "__main__":
    main()
