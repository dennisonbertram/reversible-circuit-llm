#!/usr/bin/env python3
"""
reason_gen.py — DETERMINISTIC algorithmic-reasoning trace generator for the
reversible-circuit-synthesis proxy tasks.

GOAL (the "reasoning-CoT" track): for each task instance, emit a chain-of-thought
that EXECUTES the family's synthesis ALGORITHM step-by-step on the instance's concrete
values, then the op-stream. The model is meant to learn to DERIVE the construction
(reason through it) rather than imitate/pattern-match a reference. The emitted op-stream
is ALWAYS re-checked through proxy_env.verify(...) and only verified traces are kept.

Families implemented:

  1) gf2_linear  (HIGHEST PRIORITY — pure CX/SWAP, no Toffoli; the band the model fails)
       Realize y = M x IN PLACE over GF(2) via Gauss-Jordan elimination. M (invertible)
       is factored into elementary row operations: "add row j into row i" = CX q_j q_i,
       "swap rows" = SWAP q_a q_b. Applying gates g_1..g_k to the register state x yields
       g_k ... g_1 x; we choose the gate order so the product equals M (it is the REVERSE
       of the elimination ops that reduce M -> I). The trace restates the y_i XOR
       equations, walks the elimination column-by-column, then lists the emitted gates.

  2) const_add / reg_add  (modular addition)
       - m == 2^n: a self-built Cuccaro ripple-carry adder (MAJ chain up, UMA chain down),
         narrated bit-by-bit; const_add loads the constant into a scratch register with
         free X gates, adds, then unloads. Self-emitted and verified.
       - prime m (fixed points on states >= m): narrate the value-level modular-add
         algorithm (binary of operands, ripple-carry sum, conditional subtract-m
         reduction) and emit the family's verified table-synthesized reference as the
         concrete realization of that permutation. Verified.

  3) controlled_addsub  (a conditional version of (2), narrated as "if c then add/sub").

Public API:
    gen_trace(inst) -> dict | None
        returns {"family","band","reasoning","opstream","verdict","n_ops","chars"} for a
        VERIFIED trace, or None if the instance is degenerate / too large / fails to verify.

    build_dataset(...) (see __main__) — samples training-seed instances, generates verified
    traces, and writes data/sft_reason.jsonl in chat format.

Run:  /usr/bin/python3 reason_gen.py        # builds the dataset + reports
"""
from __future__ import annotations

import json
import os
import random
import sys
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import proxy_env as pe  # noqa: E402
import tasks  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
OUT_PATH = os.path.join(DATA_DIR, "sft_reason.jsonl")
META_PATH = os.path.join(DATA_DIR, "sft_reason.meta.jsonl")
SYSTEM_PROMPT_PATH = os.path.join(HERE, "system_prompt.txt")

MAX_ASSISTANT_CHARS = 2500  # keep the whole assistant turn (reasoning + op-stream) bounded


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _opstream_lines(text: str) -> List[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _verify_keep(opstream: str, spec: dict) -> Optional[dict]:
    """Run the real verifier; return its dict iff valid + fully correct + ancilla-clean."""
    rep = pe.verify(opstream, spec)
    if rep["valid"] and rep["frac_correct"] == 1.0 and rep["frac_ancilla_clean"] == 1.0:
        return rep
    return None


def _bits_str(v: int, n: int) -> str:
    """Little-endian bit listing 'b0 b1 ... b_{n-1}' = MSB..LSB display as x = ... ."""
    return "".join(str((v >> i) & 1) for i in range(n - 1, -1, -1))


# =================================================================================
# 1) gf2_linear — Gaussian elimination over GF(2) into CX/SWAP
# =================================================================================
def decompose_gf2(M: Sequence[int], n: int,
                  compact: bool = False) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    """Factor invertible M (list of n row-bitmasks) into elementary row ops via Gauss-Jordan,
    then return the GATE sequence that realizes y = M x IN PLACE, plus a list of human-readable
    elimination narration lines.

    Math: a register CX q_p q_q does x[q] ^= x[p], i.e. left-multiply the state column-vector by
    the elementary matrix E_{q,p} = I + e_q e_p^T. SWAP q_a q_b is the permutation P_{a,b}.
    If elimination ops O_1, O_2, ..., O_t (applied to M's rows) reduce M to the identity, then as
    matrices E_t ... E_1 M = I, so M = E_1 ... E_t (every E is its own inverse over GF(2)).
    Applying gates g_1..g_k to x produces g_k ... g_1 x; choosing g_i = O_{t-i+1} makes the product
    g_k ... g_1 = E_1 ... E_t = M. Hence the gate list is the REVERSE of the elimination ops.

    compact=True emits terse one-token-per-step narration (for wide n, to stay in the char budget).
    """
    rows = list(M)
    elim: List[Tuple[str, int, int]] = []   # ('CX', src, dst) means rows[dst] ^= rows[src]
    narr: List[str] = []

    def rowadd(dst: int, src: int) -> None:
        rows[dst] ^= rows[src]
        elim.append(("CX", src, dst))
        if compact:
            narr.append(f"  row{dst} ^= row{src}")
        else:
            narr.append(f"  col {src}: row{dst} ^= row{src}  (eliminate the x{src} term from y{dst})")

    def rowswap(a: int, b: int) -> None:
        rows[a], rows[b] = rows[b], rows[a]
        elim.append(("SWAP", a, b))
        if compact:
            narr.append(f"  swap row{a},row{b}")
        else:
            narr.append(f"  col {a}: no pivot on the diagonal; SWAP row{a} <-> row{b} to bring a 1 onto the pivot")

    for col in range(n):
        piv = None
        for r in range(col, n):
            if (rows[r] >> col) & 1:
                piv = r
                break
        if piv is None:
            raise ValueError("M is singular over GF(2) — not invertible")
        if piv != col:
            rowswap(col, piv)
        for r in range(n):
            if r != col and ((rows[r] >> col) & 1):
                rowadd(r, col)

    # rows is now the identity. Gate sequence = reverse of the elimination ops.
    gates: List[Tuple[str, int, int]] = list(reversed(elim))
    return gates, narr


def _gates_to_text(gates: Sequence[Tuple[str, int, int]]) -> str:
    lines: List[str] = []
    for kind, a, b in gates:
        if kind == "CX":
            lines.append(f"CX q{a} q{b}")
        else:
            lines.append(f"SWAP q{a} q{b}")
    return "\n".join(lines)


def trace_gf2_linear(inst: tasks.Instance) -> Optional[Tuple[str, str]]:
    n = inst.params["n"]
    M = inst.params["M"]
    # restate the target XOR equations
    eqs: List[str] = []
    for i in range(n):
        ins = [f"x{j}" for j in range(n) if (M[i] >> j) & 1]
        eqs.append(f"y{i} = {' ^ '.join(ins) if ins else '0'}")

    compact = n >= 6
    try:
        gates, narr = decompose_gf2(M, n, compact=compact)
    except ValueError:
        return None
    if not gates:
        return None  # identity map — skip the degenerate no-op trace

    opstream = _gates_to_text(gates)

    # build the reasoning chain-of-thought
    R: List[str] = []
    R.append(f"GOAL: realize the GF(2)-linear map y = M x IN PLACE on the {n}-qubit register "
             f"q0..q{n-1}. The target output bits are exactly:")
    for e in eqs:
        R.append(f"  {e}")
    R.append("")
    R.append("METHOD (Gaussian elimination over GF(2)): the matrix M factors into elementary "
             "row operations. On the register, 'add row j into row i' is exactly CX qj qi "
             "(it does x_i ^= x_j), and a row swap is SWAP. I reduce M to the identity by "
             "column-pivoting; the gate sequence that COMPUTES Mx is the reverse of that "
             "elimination (each CX/SWAP is its own inverse over GF(2)).")
    R.append("")
    if not compact:
        R.append("Rows of M (bit j set means xj appears), row i = the bits of y_i:")
        for i in range(n):
            R.append(f"  row{i} = {_bits_str(M[i], n)}  -> {eqs[i]}")
        R.append("")
    R.append("Elimination steps (reduce M -> I), column by column:")
    R.extend(narr)
    R.append("")
    R.append("The reduction used the ordered ops above. Reversing them gives the gate sequence "
             "that applies M to the live register (so each x_i ends as y_i). Emitting:")
    R.append("")
    R.append(f"This is PURE CX/SWAP: 0 Toffoli, peak_width {n}. Cost = 0 (optimal for a linear map).")

    reasoning = "\n".join(R)
    return reasoning, opstream


# =================================================================================
# 2) const_add / reg_add — modular addition
# =================================================================================
def _cuccaro_narration(n: int, A_name: str, B_name: str) -> List[str]:
    """Describe the Cuccaro mod-2^n ripple-carry adder at the algorithm level."""
    return [
        f"Cuccaro ripple-carry adder, computing {B_name} += {A_name} (mod 2^{n}) in place "
        f"({A_name} preserved, one clean carry ancilla):",
        "  - MAJ chain (low bit to high bit): at each bit i compute the majority/carry into the "
        "next position using CX a b, CX a c, CCX c b a (c = incoming carry, b = sum bit, a = addend).",
        "  - drop the final carry-out to take the result mod 2^n.",
        "  - UMA chain (high bit back down to low bit): un-majority-and-add writes the sum bits into "
        f"{B_name} and restores {A_name} and the carry ancilla to their inputs (carry -> |0>).",
    ]


def trace_reg_add(inst: tasks.Instance) -> Optional[Tuple[str, str]]:
    n = inst.params["n"]
    m = inst.params["m"]

    if m == (1 << n):
        # self-build + verify our own Cuccaro (clean narratable algorithm).
        A = list(range(n))
        B = list(range(n, 2 * n))
        carry = 2 * n
        opstream = "\n".join(tasks.cuccaro_add(n, A, B, carry))
        R: List[str] = []
        R.append(f"GOAL: f(x,y) = (x + y) mod 2^{n}. Input x = q0..q{n-1}, y = q{n}..q{2*n-1}; "
                 f"x is preserved and the y register receives the sum. Carry ancilla q{carry} "
                 "starts and must end at |0>.")
        R.append("")
        R.append("METHOD: binary ripple-carry addition. Adding x into y bit-by-bit, the carry into "
                 f"position i+1 is the majority of (x_i, y_i, carry_i). Modulo 2^{n} we simply drop "
                 "the top carry-out. I use the in-place Cuccaro construction so x and the carry "
                 "ancilla are restored:")
        R.extend("  " + ln for ln in _cuccaro_narration(n, "x", "y"))
        R.append("")
        R.append(f"This emits {2*n} Toffoli-bearing positions? No — only the n CCX in the MAJ pass "
                 f"and n in the UMA pass touch each bit once; peak_width = {2*n+1}. Emitting the "
                 "MAJ-up then UMA-down op-stream:")
        return "\n".join(R), opstream

    # prime m: states >= m are fixed points -> a table permutation. Narrate the value-level
    # modular-add algorithm and emit the verified table-synthesized reference.
    ref = inst.reference_ops
    R = []
    R.append(f"GOAL: f(x,y) = (x + y) mod {m}, with x = q0..q{n-1}, y = q{n}..q{2*n-1} "
             f"(x preserved, sum into y). Only the in-range inputs x,y < {m} are constrained; "
             f"states >= {m} are fixed points to keep the map a clean bijection.")
    R.append("")
    R.append("METHOD (modular addition by ripple-carry + conditional reduction):")
    R.append(f"  1. Ripple-carry add x into y to get the raw sum s = x + y (0 <= s < {2*m-1}).")
    R.append(f"  2. Conditionally reduce: if s >= {m}, subtract {m} (s -= {m}); else leave it. This "
             f"is one comparison and one controlled constant-subtract, giving (x+y) mod {m}.")
    R.append(f"  3. x is untouched throughout; the comparison ancilla is uncomputed to |0>.")
    R.append("")
    R.append(f"I realize this permutation on the joint (x,y) space (it is fully determined by the "
             f"two rules above) with a reversible, ancilla-clean gate sequence. Emitting:")
    if len(_opstream_lines(ref)) == 0:
        return None
    return "\n".join(R), ref


def _build_const_add_2n(n: int, a: int) -> Tuple[str, dict]:
    """Self-built constant adder mod 2^n: x += a in place using a loaded Cuccaro.
    Returns (opstream, spec)."""
    in_q = list(range(n))
    out_q = list(range(n))
    A = list(range(n, 2 * n))   # scratch register that will hold the constant a
    B = list(range(n))          # the live x register
    carry = 2 * n
    lines: List[str] = []
    for i in range(n):
        if (a >> i) & 1:
            lines.append(f"X q{A[i]}")
    lines += tasks.cuccaro_add(n, A, B, carry)
    for i in range(n):
        if (a >> i) & 1:
            lines.append(f"X q{A[i]}")
    opstream = "\n".join(lines)

    def f(x: int, _a=a, _n=n) -> int:
        return (x + _a) % (1 << _n)

    spec = tasks._make_spec(n_in=n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=opstream)
    return opstream, spec


def trace_const_add(inst: tasks.Instance) -> Optional[Tuple[str, str]]:
    n = inst.params["n"]
    m = inst.params["m"]
    a = inst.params["a"] % m

    if m == (1 << n):
        # NOTE: this branch uses our self-built constant adder and its OWN spec; handled in
        # gen_trace via a dedicated path. Here we only narrate using the instance's spec.
        opstream, _spec = _build_const_add_2n(n, a)
        R: List[str] = []
        R.append(f"GOAL: f(x) = (x + {a}) mod 2^{n}, in place on q0..q{n-1}.")
        R.append("")
        R.append(f"METHOD: add the CONSTANT {a} by ripple-carry. Binary of {a} = "
                 f"{_bits_str(a, n)} (bit i set means add 2^i). I load {a} into a scratch register "
                 f"with free X gates, run an in-place Cuccaro adder x += scratch (mod 2^{n}), then "
                 "unload the scratch with the same X gates so it returns to |0>:")
        R.extend("  " + ln for ln in _cuccaro_narration(n, "the constant", "x"))
        R.append("")
        R.append("Emitting (X-load constant, Cuccaro add, X-unload):")
        return "\n".join(R), opstream

    # prime m: const add as a permutation of Z_m with fixed points >= m.
    ref = inst.reference_ops
    if len(_opstream_lines(ref)) == 0:
        return None
    R = []
    R.append(f"GOAL: f(x) = (x + {a}) mod {m}, in place on q0..q{n-1} "
             f"(a permutation of Z_{m}; states >= {m} are fixed points).")
    R.append("")
    R.append("METHOD (constant modular addition = add-then-conditionally-reduce):")
    R.append(f"  1. Add the constant {a} (binary {_bits_str(a, n)}) to x by ripple-carry, raw "
             f"sum s = x + {a}.")
    R.append(f"  2. If s >= {m}, subtract {m} to fold back into 0..{m-1}; otherwise keep s. That "
             f"yields (x + {a}) mod {m}.")
    R.append(f"  3. The {2**n - m} out-of-range states (x >= {m}) are left fixed so the whole map "
             "stays a bijection.")
    R.append("")
    R.append("Concretely the input/output pairs this must realize are:")
    pairs = ", ".join(f"{x}->{inst.f(x)}" for x in range(min(m, 8)))
    more = "" if m <= 8 else f", ... (and {m-8} more; states >= {m} fixed)"
    R.append(f"  {pairs}{more}")
    R.append("")
    R.append("I synthesize this fixed permutation as a reversible, ancilla-clean gate sequence. "
             "Emitting:")
    return "\n".join(R), ref


# =================================================================================
# 3) controlled_addsub — conditional modular add/sub
# =================================================================================
def trace_controlled_addsub(inst: tasks.Instance) -> Optional[Tuple[str, str]]:
    n = inst.params["n"]
    m = inst.params["m"]
    a = inst.params["a"] % m
    sub = inst.params["sub"]
    op = "subtract" if sub else "add"
    sign = "-" if sub else "+"
    ref = inst.reference_ops
    if len(_opstream_lines(ref)) == 0:
        return None

    R: List[str] = []
    R.append(f"GOAL: f(c, x) = c ? (x {sign} {a}) mod {m} : x. Control c = q0, x = q1..q{n} "
             f"(c preserved; x changes only when c = 1 and x < {m}).")
    R.append("")
    R.append("METHOD (controlled modular add/subtract):")
    R.append(f"  1. Everything is conditioned on the control qubit c (q0): when c = 0 the map is "
             "the identity, so every gate below is effectively controlled by c.")
    R.append(f"  2. When c = 1, {op} the constant {a} (binary {_bits_str(a, n)}) to x by "
             f"ripple-carry: raw value v = x {sign} {a}.")
    if sub:
        R.append(f"  3. Conditional reduction: if the subtraction underflowed (x < {a}), add {m} "
                 f"back so the result stays in 0..{m-1}: (x - {a}) mod {m}.")
    else:
        R.append(f"  3. Conditional reduction: if v >= {m}, subtract {m} so the result stays in "
                 f"0..{m-1}: (x + {a}) mod {m}.")
    R.append(f"  4. c is preserved and all comparison ancillas are uncomputed to |0>; out-of-range "
             f"x >= {m} are fixed points.")
    R.append("")
    R.append("Sample required pairs (c,x)->(c,x'):")
    samples = []
    for v in range(min(1 << (n + 1), 8)):
        samples.append(f"({v & 1},{v >> 1})->({inst.f(v) & 1},{inst.f(v) >> 1})")
    R.append("  " + ", ".join(samples))
    R.append("")
    R.append("I synthesize this controlled permutation as a reversible, ancilla-clean gate "
             "sequence. Emitting:")
    return "\n".join(R), ref


# =================================================================================
# top-level dispatch
# =================================================================================
def gen_trace(inst: tasks.Instance) -> Optional[Dict]:
    """Generate a VERIFIED reasoning trace for `inst`, or None if not applicable / fails.

    We build (reasoning, opstream, verifying-spec) cheaply, gate on the assistant char budget
    BEFORE the expensive basis-enumeration verify (oversized candidates are common at wide n),
    then run the full proxy_env.verify and keep only valid+correct+ancilla-clean traces.
    """
    fam = inst.family
    spec = inst.task_spec  # the spec the op-stream is verified against

    if fam == "gf2_linear":
        res = trace_gf2_linear(inst)
    elif fam == "reg_add":
        res = trace_reg_add(inst)
    elif fam == "const_add":
        n, m = inst.params["n"], inst.params["m"]
        if m == (1 << n):
            # self-built adder needs its OWN (mod 2^n) spec, not the instance's prime-mod spec.
            a = inst.params["a"] % m
            opstream, own_spec = _build_const_add_2n(n, a)
            tr = trace_const_add(inst)
            if tr is None:
                return None
            reasoning, _op = tr
            res = (reasoning, opstream)
            spec = own_spec
        else:
            res = trace_const_add(inst)
    elif fam == "controlled_addsub":
        res = trace_controlled_addsub(inst)
    else:
        return None

    if res is None:
        return None
    reasoning, opstream = res

    assistant = reasoning + "\n\nOP-STREAM:\n" + opstream
    if len(assistant) > MAX_ASSISTANT_CHARS:
        return None  # length gate BEFORE verifying (avoids enumerating doomed wide candidates)

    rep = _verify_keep(opstream, spec)
    if rep is None:
        return None

    # `spec_kind` records HOW to rebuild the verifying spec for independent re-validation:
    #   "instance"      -> spec == inst.task_spec (rebuild via tasks.build_<family>(**params))
    #   "const_add_2n"  -> spec == our self-built mod-2^n adder spec (_build_const_add_2n)
    spec_kind = "const_add_2n" if (fam == "const_add" and inst.params["m"] == (1 << inst.params["n"])) \
        else "instance"

    return {
        "family": fam,
        "band": inst.band,
        "params": dict(inst.params),
        "spec_kind": spec_kind,
        "reasoning": reasoning,
        "opstream": opstream,
        "assistant": assistant,
        "n_ops": len(_opstream_lines(opstream)),
        "chars": len(assistant),
        "verdict": {
            "valid": rep["valid"],
            "reason": rep["reason"],
            "toffoli": rep["toffoli"],
            "peak_width": rep["peak_width"],
            "cost": rep["cost"],
            "frac_correct": rep["frac_correct"],
            "frac_ancilla_clean": rep["frac_ancilla_clean"],
        },
    }


# =================================================================================
# dataset builder
# =================================================================================
# Families this generator handles, per band. (gf2_linear is the priority / bulk.)
_REASON_FAMILIES = {"gf2_linear", "const_add", "reg_add", "controlled_addsub"}

# Per-(family, band) quotas (how many VERIFIED traces to keep).
#
# Combinatorial reality of the distinct task space:
#   - gf2_linear has a HUGE distinct space at n>=4 (|GL(4,2)|=20160, |GL(5,2)|~1e7, n>=6 ~unbounded),
#     so it is the BULK of the dataset and the priority family the model fails today.
#   - the additive families are small: const_add/controlled_addsub distinct prompts per band =
#     sum_m (m-1) over the band's prime moduli (tens, not thousands); reg_add mod-2^n is a single
#     prompt and prime-m reg_add references are oversized. We take as many as exist (no-progress
#     break) rather than over-quota'ing them.
#   The emitted op-stream is always our pure-CX/SWAP derivation (0 Toffoli) regardless of band;
#   the per-instance build cost is dominated by the family builder's MMD REFERENCE (used only
#   for the honest cost-to-beat). MMD is cheap at n<=5 (the reference stays pure-CX there) but
#   expensive at n>=6 (it ignores linearity -> large Toffoli circuit), so the bulk is placed at
#   n=4/n=5 where distinct space is huge AND construction is cheap; n=6/7 get a smaller share.
_FB_QUOTA: Dict[Tuple[str, str], int] = {
    # gf2_linear (emitted PURE CX/SWAP) — the bulk and the priority family
    ("gf2_linear", "B1"): 150,    # n=3, |GL(3,2)|-1 = 167 reachable
    ("gf2_linear", "B2"): 1600,   # n=4  (|GL(4,2)| = 20160 distinct; cheap)
    ("gf2_linear", "B3"): 1600,   # n=5  (cheap; pure-CX MMD ref)
    ("gf2_linear", "B4"): 450,    # n=6  (MMD ref expensive -> smaller share)
    ("gf2_linear", "B5"): 250,    # n=6
    ("gf2_linear", "B6"): 120,    # n=7  (MMD ref very expensive -> small share)
    # const_add (modular add narration + verified ref)
    ("const_add", "B1"): 60,
    ("const_add", "B2"): 120,
    ("const_add", "B3"): 200,
    # controlled_addsub (conditional modular add/sub)
    ("controlled_addsub", "B1"): 60,
    ("controlled_addsub", "B4"): 60,
    # reg_add (Cuccaro mod-2^n + prime-m where short enough)
    ("reg_add", "B2"): 60,
}

# Stop sampling a slot after this many CONSECUTIVE samples add nothing new (space exhausted).
# Kept modest: the additive families saturate their small distinct spaces within a few hundred
# samples, and each miss for those builds an expensive MMD reference, so we don't want to grind
# thousands. gf2_linear (the bulk) hits its quota long before this triggers.
_NO_PROGRESS_LIMIT = 600

# Training seeds: DISJOINT from eval (>=60000) and from build_heldout_* (fixed seeds).
_SEED_LO = 1
_SEED_HI = 50000


def _fast_gf2_instance(n: int, band: str, rng: random.Random) -> Optional[tasks.Instance]:
    """Build a gf2_linear Instance at width n.

    Uses the family builder tasks.build_gf2_linear, which attaches the curriculum's MMD
    reference and its HONEST reference_cost (note: the MMD synthesizer treats the linear map
    as a generic permutation, so at n>=6 the reference is full of Toffolis and large — a real
    cost-to-beat that our pure-CX (cost 0) derivation crushes; that gap is exactly the lesson).

    Skips the degenerate identity matrix (where there is nothing to derive). `rng` is a
    training-namespace stream (seeds 1..50000) — DISJOINT from eval seeds >= 60000.
    """
    M = tasks._rand_invertible_gf2(n, rng)
    if M == [1 << i for i in range(n)]:
        return None  # identity -> nothing to derive, skip
    inst = tasks.build_gf2_linear(n, M)
    inst.band = band
    inst.prompt = tasks.render_prompt(inst)
    return inst


def build_dataset(verbose: bool = True) -> Dict:
    system_prompt = open(SYSTEM_PROMPT_PATH).read()
    os.makedirs(DATA_DIR, exist_ok=True)

    kept: List[Dict] = []                       # {messages, meta}
    per_fb: Dict[Tuple[str, str], int] = {}     # (family, band) -> count
    seen_user: set = set()                      # de-dup identical prompts

    def quota(fam: str, band: str) -> int:
        return _FB_QUOTA.get((fam, band), 0)

    def want(fam: str, band: str) -> bool:
        return per_fb.get((fam, band), 0) < quota(fam, band)

    def add(inst: tasks.Instance) -> bool:
        """Returns True iff a NEW verified trace was kept."""
        fam, band = inst.family, inst.band
        if fam not in _REASON_FAMILIES or not want(fam, band):
            return False
        if inst.prompt in seen_user:
            return False
        tr = gen_trace(inst)
        if tr is None:
            return False
        seen_user.add(inst.prompt)
        per_fb[(fam, band)] = per_fb.get((fam, band), 0) + 1
        kept.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": inst.prompt},
                {"role": "assistant", "content": tr["assistant"]},
            ],
            "meta": {
                "family": tr["family"], "band": band, "n_ops": tr["n_ops"],
                "chars": tr["chars"], "cost": tr["verdict"]["cost"],
                "peak_width": tr["verdict"]["peak_width"], "toffoli": tr["verdict"]["toffoli"],
                "params": tr["params"], "spec_kind": tr["spec_kind"],
            },
        })
        return True

    def fill_slot_via_curriculum(fam: str, band: str) -> None:
        """Fill one (family, band) slot by sampling the curriculum with training seeds, with a
        no-progress break once the reachable distinct space is exhausted."""
        no_progress = 0
        # deterministic per-slot starting seed (stable across runs; no PYTHONHASHSEED dependence)
        seed = _SEED_LO + (zlib.adler32(f"{fam}/{band}".encode()) % 997)
        while want(fam, band) and no_progress < _NO_PROGRESS_LIMIT and seed < _SEED_HI:
            rng = random.Random(seed)
            seed += 1
            try:
                inst = tasks.sample_instance(fam, band, rng)
            except Exception:
                no_progress += 1
                continue
            no_progress = 0 if add(inst) else no_progress + 1
        if verbose:
            print(f"  filled {fam}/{band}: {per_fb.get((fam, band), 0)}/{quota(fam, band)}")

    def fill_slot_wide_gf2(band: str, n: int, base_seed: int) -> None:
        """Fill a gf2_linear slot by directly sampling invertible matrices at width n
        (training-namespace RNG, disjoint from eval seeds >= 60000)."""
        wrng = random.Random(base_seed)
        no_progress = 0
        while want("gf2_linear", band) and no_progress < _NO_PROGRESS_LIMIT:
            inst = _fast_gf2_instance(n, band, wrng)
            no_progress = 0 if (inst is not None and add(inst)) else no_progress + 1
        if verbose:
            print(f"  filled gf2_linear/{band} (n={n}): "
                  f"{per_fb.get(('gf2_linear', band), 0)}/{quota('gf2_linear', band)}")

    # --- gf2_linear (the bulk): sample matrices directly at each band's width. ---
    gf2_width = {"B1": 3, "B2": 4, "B3": 5, "B4": 6, "B5": 6, "B6": 7}
    for i, band in enumerate(["B1", "B2", "B3", "B4", "B5", "B6"]):
        fill_slot_wide_gf2(band, gf2_width[band], base_seed=4200 + i)

    # --- additive families: sample via the curriculum (small reachable spaces). ---
    for (fam, band) in _FB_QUOTA:
        if fam == "gf2_linear":
            continue
        fill_slot_via_curriculum(fam, band)

    # write the dataset (chat format) + a companion meta file for independent re-validation.
    with open(OUT_PATH, "w") as fh:
        for row in kept:
            fh.write(json.dumps({"messages": row["messages"]}) + "\n")
    with open(META_PATH, "w") as fh:
        for row in kept:
            fh.write(json.dumps(row["meta"]) + "\n")

    return {
        "out_path": OUT_PATH,
        "meta_path": META_PATH,
        "total": len(kept),
        "per_family_band": {f"{f}/{b}": c for (f, b), c in sorted(per_fb.items())},
        "kept": kept,
    }


# =================================================================================
# validation + report
# =================================================================================
def _extract_opstream(assistant: str) -> str:
    """Pull the op-stream body out of an assistant turn (everything after 'OP-STREAM:')."""
    marker = "OP-STREAM:\n"
    idx = assistant.rfind(marker)
    if idx < 0:
        return ""
    return assistant[idx + len(marker):]


def _rebuild_spec(meta: dict) -> dict:
    """Reconstruct the EXACT verifying task_spec from a meta record (family + params)."""
    fam = meta["family"]
    p = meta["params"]
    if meta.get("spec_kind") == "const_add_2n":
        n = p["n"]
        a = p["a"] % p["m"]
        _op, spec = _build_const_add_2n(n, a)
        return spec
    if fam == "gf2_linear":
        return tasks.build_gf2_linear(p["n"], p["M"]).task_spec
    if fam == "const_add":
        return tasks.build_const_add(p["n"], p["m"], p["a"]).task_spec
    if fam == "reg_add":
        return tasks.build_reg_add(p["n"], p["m"]).task_spec
    if fam == "controlled_addsub":
        return tasks.build_controlled_addsub(p["n"], p["m"], p["a"], sub=p["sub"]).task_spec
    raise ValueError(f"cannot rebuild spec for family {fam}")


def revalidate(k: int = 200, seed: int = 13) -> dict:
    """Independently re-verify a random sample of k written traces: extract the op-stream from
    the JSONL assistant turn, rebuild the task_spec from the companion meta, and re-run
    proxy_env.verify. Returns pass/fail counts. This is the 'random 200' acceptance gate."""
    data_lines = open(OUT_PATH).read().splitlines()
    meta_lines = open(META_PATH).read().splitlines()
    assert len(data_lines) == len(meta_lines), "data/meta length mismatch"
    rng = random.Random(seed)
    idxs = rng.sample(range(len(data_lines)), min(k, len(data_lines)))
    passed = 0
    failures: List[dict] = []
    for i in idxs:
        row = json.loads(data_lines[i])
        meta = json.loads(meta_lines[i])
        assistant = row["messages"][-1]["content"]
        opstream = _extract_opstream(assistant)
        spec = _rebuild_spec(meta)
        rep = pe.verify(opstream, spec)
        ok = bool(rep["valid"] and rep["frac_correct"] == 1.0 and rep["frac_ancilla_clean"] == 1.0)
        if ok:
            passed += 1
        else:
            failures.append({"idx": i, "family": meta["family"], "band": meta["band"],
                             "reason": rep["reason"], "frac_correct": rep["frac_correct"]})
    return {"checked": len(idxs), "passed": passed,
            "fraction": passed / len(idxs) if idxs else 0.0, "failures": failures}


def one_full_sample(seed: int = 7) -> dict:
    """Return one complete (prompt, reasoning, op-stream, verdict) example for display.
    Prefers a gf2_linear example (the priority family the model fails today)."""
    data_lines = open(OUT_PATH).read().splitlines()
    meta_lines = open(META_PATH).read().splitlines()
    rng = random.Random(seed)
    order = list(range(len(data_lines)))
    rng.shuffle(order)
    pick = None
    for i in order:
        meta = json.loads(meta_lines[i])
        if meta["family"] == "gf2_linear" and meta["params"]["n"] in (3, 4):
            pick = i
            break
    if pick is None:
        pick = order[0]
    row = json.loads(data_lines[pick])
    meta = json.loads(meta_lines[pick])
    assistant = row["messages"][-1]["content"]
    opstream = _extract_opstream(assistant)
    spec = _rebuild_spec(meta)
    rep = pe.verify(opstream, spec)
    return {
        "prompt": row["messages"][1]["content"],
        "assistant": assistant,
        "verdict": {k2: rep[k2] for k2 in
                    ("valid", "reason", "toffoli", "peak_width", "cost",
                     "vs_ref_pct", "frac_correct", "frac_ancilla_clean")},
        "meta": meta,
    }


def report(result: Dict) -> None:
    print("=" * 76)
    print("REASONING-CoT DATASET BUILD REPORT  (data/sft_reason.jsonl)")
    print("=" * 76)
    print(f"file            : {result['out_path']}")
    print(f"meta            : {result['meta_path']}")
    print(f"total verified  : {result['total']}")
    print("per family/band :")
    by_fam: Dict[str, int] = {}
    by_band: Dict[str, int] = {}
    for key, c in result["per_family_band"].items():
        fam, band = key.split("/")
        by_fam[fam] = by_fam.get(fam, 0) + c
        by_band[band] = by_band.get(band, 0) + c
        print(f"    {key:28s} {c}")
    print("by family       :", by_fam)
    print("by band         :", by_band)

    # ---- independent re-validation of a random 200 ----
    print("-" * 76)
    rv = revalidate(k=200, seed=13)
    print(f"RE-VALIDATION (random {rv['checked']}): {rv['passed']}/{rv['checked']} pass "
          f"({100.0 * rv['fraction']:.1f}%)  [extract op-stream from JSONL -> rebuild spec -> "
          f"proxy_env.verify]")
    if rv["failures"]:
        print("  FAILURES:", rv["failures"][:10])

    # ---- one full sample trace ----
    print("=" * 76)
    print("ONE FULL SAMPLE TRACE (prompt + reasoning + op-stream + verifier verdict)")
    print("=" * 76)
    s = one_full_sample(seed=7)
    print("----- USER PROMPT -----")
    print(s["prompt"])
    print("\n----- ASSISTANT (reasoning CoT + OP-STREAM) -----")
    print(s["assistant"])
    print("\n----- VERIFIER VERDICT (proxy_env.verify) -----")
    print(json.dumps(s["verdict"], indent=2))


if __name__ == "__main__":
    res = build_dataset()
    report(res)
