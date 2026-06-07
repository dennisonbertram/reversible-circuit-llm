#!/usr/bin/env python3
"""
test_equivalence.py — cross-check the pure-Python proxy simulator (proxy_env.py) against the
REAL Rust ground truth (proxy_rs/proxy_verify, which vendors byte-identical copies of the
challenge's circuit.rs/sim.rs).

For >=300 random small op-streams (random sequences of X/CX/CCX/SWAP over W in 3..=8, random
valid in-range operands, no aliasing), we run BOTH:
  * the Python simulator  -> simulate_basis_states(text, W) -> (toffoli, peak_width, final_states)
  * the Rust proxy_verify -> one JSON line per request          -> same tuple

and ASSERT exact agreement on:
  * executed Toffoli count (sum over all 2^W basis inputs / batches),
  * peak_width (== analyze_ops num_qubits),
  * the final W-qubit state for every one of the 2^W enumerated basis inputs.

The Rust binary is fed one request per stdin line and emits one result line, so the whole
suite is a single subprocess round-trip. Prints PASSED/TOTAL and exits non-zero on any
mismatch.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import proxy_env as pe  # noqa: E402

RUST_BIN = os.path.normpath(
    os.path.join(HERE, "..", "proxy_rs", "target", "release", "proxy_verify")
)

# Gate set used for the equivalence sweep: bit-permutation ops only (X/CX/CCX/SWAP), which is
# exactly what the task specifies. (Phase ops and conditioning are covered by dedicated unit
# checks below; the random sweep focuses on the classical truth table + Toffoli accounting.)
GATES = ["X", "CX", "CCX", "SWAP"]
ARITY = {"X": 1, "CX": 2, "CCX": 3, "SWAP": 2}


def random_op(width: int, rng: random.Random) -> str:
    kind = rng.choice(GATES)
    n = ARITY[kind]
    # pick n distinct qubit indices in [0, width) (no aliasing — required by Op::validate)
    qs = rng.sample(range(width), n)
    return kind + " " + " ".join(f"q{q}" for q in qs)


def random_stream(rng: random.Random) -> tuple:
    width = rng.randint(3, 8)
    n_ops = rng.randint(0, 24)
    lines = [random_op(width, rng) for _ in range(n_ops)]
    # occasionally inject a comment / blank line to exercise the parser's skip logic
    if rng.random() < 0.3:
        pos = rng.randint(0, len(lines))
        lines.insert(pos, "# a comment line")
    if rng.random() < 0.2:
        pos = rng.randint(0, len(lines))
        lines.insert(pos, "")
    text = "\n".join(lines)
    return text, width


# ---------------------------------------------------------------------------------------------
# Broad sweep: the FULL op set (phase ops, classical bits, conditioning, PUSH/POP_CONDITION).
# This is where the subtle sim semantics live — weighted Toffoli under conditioning, phase
# tracking, nested condition stack, bit writes. We compare the full tuple
# (toffoli, peak_width, final_states, phase_bits, bit_states) against the Rust ground truth.
# ---------------------------------------------------------------------------------------------
BROAD_QUBIT_GATES = {
    "X": 1, "Z": 1, "CX": 2, "CZ": 2, "SWAP": 2, "CCX": 3, "CCZ": 3, "NEG": 0,
}
BROAD_BIT_GATES = ["BIT_INVERT", "BIT_STORE0", "BIT_STORE1"]


def random_broad_op(width: int, n_bits: int, rng: random.Random,
                    cond_depth: int) -> tuple:
    """Return (line, new_cond_depth). Generates a valid op over the full set, optionally with
    an `if bM` condition or a PUSH/POP_CONDITION. n_bits bits available for conditioning/writes."""
    roll = rng.random()
    # PUSH/POP_CONDITION to exercise the nested condition stack
    if n_bits > 0 and roll < 0.10:
        b = rng.randrange(n_bits)
        return f"PUSH_CONDITION if b{b}", cond_depth + 1
    if cond_depth > 0 and roll < 0.18:
        return "POP_CONDITION", cond_depth - 1

    # classical bit write
    if n_bits > 0 and roll < 0.35:
        kind = rng.choice(BROAD_BIT_GATES)
        b = rng.randrange(n_bits)
        line = f"{kind} b{b}"
    else:
        kind = rng.choice(list(BROAD_QUBIT_GATES.keys()))
        n = BROAD_QUBIT_GATES[kind]
        if n == 0:
            line = kind  # NEG
        else:
            qs = rng.sample(range(width), n)
            line = kind + " " + " ".join(f"q{q}" for q in qs)

    # optionally attach an `if bM` condition (allowed on all these kinds)
    if n_bits > 0 and rng.random() < 0.4:
        b = rng.randrange(n_bits)
        line += f" if b{b}"
    return line, cond_depth


def random_broad_stream(rng: random.Random) -> tuple:
    width = rng.randint(3, 8)
    n_bits = rng.randint(0, 3)
    n_ops = rng.randint(0, 28)
    lines = []
    cond_depth = 0
    for _ in range(n_ops):
        line, cond_depth = random_broad_op(width, n_bits, rng, cond_depth)
        lines.append(line)
    # balance any open PUSH_CONDITIONs (not required by grammar, but keeps streams realistic)
    text = "\n".join(lines)
    return text, width


def run_rust_batch(requests: list) -> list:
    """requests: list of (text, width). Returns list of dicts parsed from the Rust output lines."""
    lines = []
    for text, width in requests:
        obj = {"opstream": text, "width": width}
        lines.append(json.dumps(obj))
    stdin_data = "\n".join(lines) + "\n"
    proc = subprocess.run(
        [RUST_BIN],
        input=stdin_data,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"proxy_verify exited {proc.returncode}: {proc.stderr}")
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(out_lines) != len(requests):
        raise RuntimeError(
            f"expected {len(requests)} result lines, got {len(out_lines)}"
        )
    return [json.loads(ln) for ln in out_lines]


def main() -> int:
    if not os.path.exists(RUST_BIN):
        print(f"ERROR: Rust binary not found at {RUST_BIN}")
        print("Build it: cd proxy_rs && cargo build --release")
        return 2

    seed = int(os.environ.get("EQUIV_SEED", "12345"))
    n_basic = int(os.environ.get("EQUIV_N", "400"))
    n_broad = int(os.environ.get("EQUIV_N_BROAD", "400"))
    rng = random.Random(seed)

    total_passed = 0
    total_cases = 0
    fail = None

    # -------- Sweep 1: X/CX/CCX/SWAP (bit-permutation truth table + Toffoli) ----------
    basic_reqs = [random_stream(rng) for _ in range(n_basic)]
    basic_rust = run_rust_batch(basic_reqs)
    b_pass = 0
    for i, (text, width) in enumerate(basic_reqs):
        py = pe.simulate_full(text, width)
        rr = basic_rust[i]
        py_t = (py[0], py[1], py[2], py[3], py[4])
        rs_t = (rr["toffoli"], rr["peak_width"], rr["final_states"],
                rr["phase_bits"], rr["bit_states"])
        if py_t == rs_t:
            b_pass += 1
        elif fail is None:
            fail = ("basic", i, width, text, py_t, rs_t)
    print(f"  sweep basic (X/CX/CCX/SWAP):       PASSED {b_pass}/{len(basic_reqs)}")
    total_passed += b_pass
    total_cases += len(basic_reqs)

    # -------- Sweep 2: full op set (phase, bits, conditioning, PUSH/POP) -------------
    broad_reqs = [random_broad_stream(rng) for _ in range(n_broad)]
    broad_rust = run_rust_batch(broad_reqs)
    br_pass = 0
    for i, (text, width) in enumerate(broad_reqs):
        py = pe.simulate_full(text, width)
        rr = broad_rust[i]
        py_t = (py[0], py[1], py[2], py[3], py[4])
        rs_t = (rr["toffoli"], rr["peak_width"], rr["final_states"],
                rr["phase_bits"], rr["bit_states"])
        if py_t == rs_t:
            br_pass += 1
        elif fail is None:
            fail = ("broad", i, width, text, py_t, rs_t)
    print(f"  sweep broad (full op set + cond):  PASSED {br_pass}/{len(broad_reqs)}")
    total_passed += br_pass
    total_cases += len(broad_reqs)

    print(f"PASSED {total_passed}/{total_cases}")

    if fail is not None:
        which, idx, width, text, py_t, rs_t = fail
        print(f"\nFIRST MISMATCH (sweep={which}, index={idx}, width={width}):")
        print(f"  opstream:\n{text}")
        print(f"  python  : toffoli={py_t[0]} peak={py_t[1]}")
        print(f"            states={py_t[2]}")
        print(f"            phase ={py_t[3]}")
        print(f"            bits  ={py_t[4]}")
        print(f"  rust    : toffoli={rs_t[0]} peak={rs_t[1]}")
        print(f"            states={rs_t[2]}")
        print(f"            phase ={rs_t[3]}")
        print(f"            bits  ={rs_t[4]}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
