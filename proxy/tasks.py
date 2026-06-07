"""
tasks.py — proxy curriculum TASK + REFERENCE-CIRCUIT layer for the ECDSA-fail proxy RL env.

This module sits on top of the verified verifier in `proxy_env.py`. For every task FAMILY
it provides:

  (1) the target function f in Python (a callable f(input_value) -> output_value),
  (2) a REFERENCE op-stream generator that emits a CORRECT, reversible, ancilla-clean,
      phase-0 circuit for that instance (references are baselines to BEAT — valid, not
      necessarily optimal),
  (3) an instance sampler per curriculum band (B0..B6, proxy_env_design.md §e),
  (4) a prompt renderer producing the model-facing PROMPT text (task spec, declared
      register layout as REGISTER/APPEND_TO_REGISTER lines, reference cost to beat, rules).

Families implemented (proxy_env_design.md §a):
  - const_add        f(x) = (x + a) mod m
  - reg_add          f(x,y) = (x + y) mod m
  - controlled_addsub f(c,x) = c ? (x ± a mod m) : x
  - mod_mult         f(x) = (k*x) mod m  (k a unit -> permutation of Z_m)
  - mod_inverse      f(x) = x^{-1} mod m  (f(0)=0), a permutation of Z_m
  - sbox             arbitrary fixed boolean permutation given by a truth table
  - gf2_linear       x -> M x over GF(2), M invertible

Two synthesis strategies are used for the reference circuits:
  * Constructive reversible arithmetic (Cuccaro ripple-carry adder) for the additive
    primitives that have a clean textbook construction.
  * A GENERIC transformation-based reversible synthesizer (Miller-Maslov-Dueck style) for
    the table-defined permutation families (S-box, modular multiply, modular inverse, GF(2)
    linear maps, modular add via reduction): iterate input patterns in increasing order,
    apply multi-controlled-NOT gates to fix each output, then decompose every
    multi-controlled-NOT into CCX with clean borrowed ancillas. This is guaranteed correct
    and ancilla-clean (possibly large — fine for a baseline to beat).

CONTRACT with proxy_env.verify(opstream_text, task_spec):
  task_spec carries: in_qubits, out_qubits, width, max_width, f (callable) and/or
  truth_table, reference_cost. The verifier enumerates ALL 2^enum_width basis states,
  checks full-state reversibility, correctness of f on the 2^n_in declared inputs (ancillas
  start |0>), phase==0, ancilla qubits return to |0>, and forward∘reverse identity.

Public API:
    FAMILIES                                  # list[str] of family names
    sample_instance(family, band, rng)        # -> Instance
    sample_band(band, rng, n_per_family=...)   # -> list[Instance]
    build_curriculum(rng, n_per_family=...)    # -> dict[band] -> list[Instance]
    render_prompt(instance) -> str             # model-facing prompt text
    Instance dataclass: .family .band .params .f .task_spec .reference_ops .prompt

    # reusable synthesis primitives
    mmd_synthesize_perm(perm, w)               # -> list of gate tuples
    emit_mcx(controls, target, anc)            # -> op-text lines
    synth_perm_ops(perm, n, in_q, out_q, anc)  # -> (op_text, peak_width, toffoli_avg)
    cuccaro_add(n, A, B, carry)                # -> op-text lines (B += A mod 2^n)
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import proxy_env as pe


FAMILIES = [
    "const_add",
    "reg_add",
    "controlled_addsub",
    "mod_mult",
    "mod_inverse",
    "sbox",
    "gf2_linear",
]

# Primes available per input width n (largest-prime-<=2^n style choices used by the design).
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]


def _primes_le(limit: int) -> List[int]:
    return [p for p in _PRIMES if p <= limit]


def _is_prime(x: int) -> bool:
    if x < 2:
        return False
    i = 2
    while i * i <= x:
        if x % i == 0:
            return False
        i += 1
    return True


# ===================================================================================
# Op-stream helpers
# ===================================================================================
def _peak_of(text: str) -> int:
    hi = -1
    for tok in re.findall(r"q(\d+)", text):
        hi = max(hi, int(tok))
    return hi + 1


def _toffoli_of(text: str) -> int:
    """Static unconditioned Toffoli count = number of CCX/CCZ lines (each fires on the
    full live batch for unconditioned references, so this equals avg toffoli)."""
    n = 0
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0] in ("CCX", "CCZ"):
            n += 1
    return n


# ===================================================================================
# Generic transformation-based reversible synthesizer (Miller-Maslov-Dueck style)
# ===================================================================================
def mmd_synthesize_perm(perm: Sequence[int], w: int) -> List[Tuple[int, int]]:
    """Synthesize a permutation `perm` of range(2**w) into a sequence of multi-controlled
    NOT gates. Returns a list of (controls_mask, target_bit) gates such that applying them
    in order to a basis state x yields perm[x]. A gate fires iff (x & controls_mask) ==
    controls_mask (i.e. all the set bits of controls_mask are 1), flipping bit `target_bit`.

    Algorithm (Maslov's basic transformation-based synthesis): walk i = 0..2^w-1; at each
    step turn the current output of i into i using two phases of multi-controlled NOTs
    chosen so already-fixed rows (< i) are never disturbed. The accumulated gate list maps
    perm -> identity when read on the output side; the REVERSE list (each MCX self-inverse)
    realizes perm on the input side.
    """
    n = 1 << w
    assert sorted(perm) == list(range(n)), "perm must be a permutation of range(2**w)"
    f = list(perm)  # f[x] = current image of input x
    gates: List[Tuple[int, int]] = []

    def apply_gate(controls_mask: int, target_bit: int) -> None:
        tb = 1 << target_bit
        for x in range(n):
            if (f[x] & controls_mask) == controls_mask:
                f[x] ^= tb

    for i in range(n):
        p = f[i]
        if p == i:
            continue
        cur = p
        # phase 1: bits set in i but not in current value -> turn 0->1, controlled by cur's set bits
        for j in range(w):
            if ((i >> j) & 1) and not ((cur >> j) & 1):
                controls_mask = cur
                gates.append((controls_mask, j))
                apply_gate(controls_mask, j)
                cur |= (1 << j)
        # phase 2: bits set in current value but not in i -> turn 1->0, controlled by i's set bits
        for j in range(w):
            if not ((i >> j) & 1) and ((cur >> j) & 1):
                controls_mask = i
                gates.append((controls_mask, j))
                apply_gate(controls_mask, j)
                cur &= ~(1 << j)
    assert f == list(range(n)), "MMD synthesis failed to reach identity"
    return list(reversed(gates))


def emit_mcx(controls: Sequence[int], target: int, anc: Sequence[int]) -> List[str]:
    """Emit op-text lines for a multi-controlled NOT: flip `target` iff every qubit in
    `controls` is 1. Uses CCX with clean ancillas; ancillas are returned to |0>.

    k = len(controls):
      0 -> X target
      1 -> CX c0 target
      2 -> CCX c0 c1 target
      k>=3 -> AND-tree compute into (k-2) clean ancillas, single CCX onto target, then
              uncompute the tree in reverse (ancillas restored to |0>).
    """
    controls = list(controls)
    k = len(controls)
    if k == 0:
        return [f"X q{target}"]
    if k == 1:
        return [f"CX q{controls[0]} q{target}"]
    if k == 2:
        return [f"CCX q{controls[0]} q{controls[1]} q{target}"]
    need = k - 2
    if len(anc) < need:
        raise ValueError(f"emit_mcx needs {need} ancillas for {k} controls, got {len(anc)}")
    a = list(anc)
    lines: List[str] = []
    # compute AND tree
    lines.append(f"CCX q{controls[0]} q{controls[1]} q{a[0]}")
    for j in range(2, k - 1):
        lines.append(f"CCX q{controls[j]} q{a[j - 2]} q{a[j - 1]}")
    # apply to target: target ^= controls[k-1] & a[k-3]
    lines.append(f"CCX q{controls[k - 1]} q{a[k - 3]} q{target}")
    # uncompute tree (reverse)
    for j in range(k - 2, 1, -1):
        lines.append(f"CCX q{controls[j]} q{a[j - 2]} q{a[j - 1]}")
    lines.append(f"CCX q{controls[0]} q{controls[1]} q{a[0]}")
    return lines


def synth_perm_ops(perm: Sequence[int], n: int) -> Tuple[str, int, List[int]]:
    """Synthesize an n-qubit permutation (in place on qubits 0..n-1) into an op-text stream
    using MMD + MCX decomposition. Borrowed-clean ancillas are placed at indices n, n+1, ...
    (as many as the widest MCX needs, at most n-2). Returns (op_text, peak_width,
    ancilla_qubits)."""
    gates = mmd_synthesize_perm(perm, n)
    # max controls in any gate
    max_ctrl = 0
    for cmask, _t in gates:
        max_ctrl = max(max_ctrl, bin(cmask).count("1"))
    n_anc = max(0, max_ctrl - 2)
    anc = list(range(n, n + n_anc))
    lines: List[str] = []
    for cmask, target in gates:
        controls = [b for b in range(n) if (cmask >> b) & 1]
        lines.extend(emit_mcx(controls, target, anc))
    text = "\n".join(lines)
    peak = max(n, _peak_of(text)) if lines else n
    return text, peak, anc


# ===================================================================================
# Constructive reversible arithmetic: Cuccaro ripple-carry adder
# ===================================================================================
def cuccaro_add(n: int, A: Sequence[int], B: Sequence[int], carry: int) -> List[str]:
    """In-place modular ripple-carry adder (Cuccaro et al., quant-ph/0410184) computing
    B += A  (mod 2^n): the sum lands in register B, register A is preserved, and `carry`
    is a clean ancilla returned to |0>. MAJ chain up, UMA chain down; the high carry-out is
    dropped to realize mod 2^n. Validated against proxy_env.verify."""
    A = list(A)
    B = list(B)
    carry_in = [carry] + A[: n - 1]
    lines: List[str] = []
    for i in range(n):
        c, b, a = carry_in[i], B[i], A[i]
        lines += [f"CX q{a} q{b}", f"CX q{a} q{c}", f"CCX q{c} q{b} q{a}"]
    for i in reversed(range(n)):
        c, b, a = carry_in[i], B[i], A[i]
        lines += [f"CCX q{c} q{b} q{a}", f"CX q{a} q{c}", f"CX q{c} q{b}"]
    return lines


# ===================================================================================
# Instance container
# ===================================================================================
@dataclass
class Instance:
    family: str
    band: str
    params: Dict
    f: Callable[[int], int]
    task_spec: Dict
    reference_ops: str
    prompt: str = ""
    desc: str = ""


# ---- low-level builder: wrap a function f over n_in input qubits into a task_spec ----
def _make_spec(
    *,
    n_in: int,
    in_qubits: List[int],
    out_qubits: List[int],
    f: Callable[[int], int],
    reference_ops: str,
    width_slack: int = 2,
) -> Dict:
    """Build a task_spec dict from an output function and its reference op stream. width is
    the enumeration width = max declared/used qubit index + 1; max_width = peak(ref)+slack so
    the policy has scratch room but cannot buy correctness with unbounded ancilla.

    reference_cost is computed by RUNNING proxy_env.verify() on the reference op-stream so it
    EXACTLY equals the cost the verifier measures (avg executed Toffoli x peak_width). Using
    the static CCX line count would be wrong whenever a CCX fires on only a subset of inputs
    (conditioned / data-dependent), making vs_ref_pct and the cost bonus inconsistent — a
    model re-emitting the reference must score a clean tie (vs_ref_pct == 1.0)."""
    ref_peak = max(_peak_of(reference_ops), (max(in_qubits + out_qubits) + 1))
    width = ref_peak
    out_mask = (1 << len(out_qubits)) - 1
    spec = {
        "in_qubits": list(in_qubits),
        "out_qubits": list(out_qubits),
        "n_in": n_in,
        "width": width,
        "max_width": ref_peak + width_slack,
        "f": (lambda x, _f=f, _m=out_mask: _f(x) & _m),
        "reference_cost": 1.0,  # placeholder; replaced below by the verifier-measured cost
    }
    # Measure the reference's TRUE cost via the verifier (placeholder ref_cost is unused for
    # the cost number itself — verify() reports cost = avg_toffoli * peak_width directly).
    rep = pe.verify(reference_ops, spec)
    if not rep["valid"]:
        raise ValueError(
            f"reference op-stream did not validate: reason={rep['reason']} "
            f"frac_correct={rep['frac_correct']} anc={rep['frac_ancilla_clean']}"
        )
    spec["reference_cost"] = float(rep["cost"])
    return spec


# ===================================================================================
# FAMILY BUILDERS — each returns an Instance (function f, reference ops, task_spec).
# Layout convention: input register on low qubits; ancillas above. Out-of-place families
# place the output register above the input register, with scratch above that.
# ===================================================================================

# ---------- const_add: f(x) = (x + a) mod m, in place (permutation of Z_m) -----------
def build_const_add(n: int, m: int, a: int) -> Instance:
    a = a % m
    in_q = list(range(n))
    out_q = list(range(n))

    def f(x: int) -> int:
        return (x + a) % m if x < m else x  # identity on out-of-range states (-> bijection)

    perm = [f(x) for x in range(1 << n)]
    ref, peak, anc = synth_perm_ops(perm, n)
    spec = _make_spec(n_in=n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    desc = (f"Constant modular add: f(x) = (x + {a}) mod {m} on the {n}-qubit input "
            f"register (states >= {m} are fixed points). In place.")
    return Instance("const_add", "", dict(n=n, m=m, a=a), f, spec, ref, desc=desc)


# ---------- reg_add: f(x,y) = (x + y) mod m, sum -> y register (x preserved) ----------
def build_reg_add(n: int, m: int) -> Instance:
    """Two n-qubit inputs x (qubits 0..n-1) and y (qubits n..2n-1). Output: x preserved,
    y register holds (x+y) mod m when both x<m and y<m. To keep it a clean full-state
    bijection we use mod 2^n addition via Cuccaro when m == 2**n, else a table-synthesized
    modular adder over the joint (x,y) space."""
    in_q = list(range(2 * n))
    out_q = list(range(2 * n))

    if m == (1 << n):
        # constructive Cuccaro: B(=y) += A(=x) mod 2^n, x preserved, 1 ancilla.
        A = list(range(n))
        B = list(range(n, 2 * n))
        carry = 2 * n
        ref = "\n".join(cuccaro_add(n, A, B, carry))

        def f(v: int) -> int:
            x = v & ((1 << n) - 1)
            y = (v >> n) & ((1 << n) - 1)
            s = (x + y) % m
            return x | (s << n)

        spec = _make_spec(n_in=2 * n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
        desc = (f"Register modular add (mod 2^{n}): inputs x=q0..q{n-1}, y=q{n}..q{2*n-1}; "
                f"output x preserved, y <- (x + y) mod {m}. Constructive Cuccaro ripple-carry.")
        return Instance("reg_add", "", dict(n=n, m=m), f, spec, ref, desc=desc)

    # prime/general m: synthesize the joint permutation on 2n qubits via MMD.
    def f(v: int) -> int:
        x = v & ((1 << n) - 1)
        y = (v >> n) & ((1 << n) - 1)
        if x < m and y < m:
            s = (x + y) % m
            return x | (s << n)
        return v  # out-of-range -> fixed point (bijection)

    perm = [f(v) for v in range(1 << (2 * n))]
    ref, peak, anc = synth_perm_ops(perm, 2 * n)
    spec = _make_spec(n_in=2 * n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    desc = (f"Register modular add (mod {m}): inputs x=q0..q{n-1}, y=q{n}..q{2*n-1}; "
            f"output x preserved, y <- (x + y) mod {m} when both in range.")
    return Instance("reg_add", "", dict(n=n, m=m), f, spec, ref, desc=desc)


# ---------- controlled_addsub: f(c,x) = c ? (x +/- a mod m) : x -----------------------
def build_controlled_addsub(n: int, m: int, a: int, sub: bool = False) -> Instance:
    """Control qubit c = q0, x register = q1..qn. If c==1 apply (x +/- a) mod m, else
    identity; c preserved. Synthesized over the full (c,x) permutation via MMD."""
    a = a % m
    cbit = 0
    xq = list(range(1, n + 1))
    in_q = [cbit] + xq
    out_q = [cbit] + xq
    width = n + 1

    def f(v: int) -> int:
        c = v & 1
        x = (v >> 1) & ((1 << n) - 1)
        if c and x < m:
            x = (x - a) % m if sub else (x + a) % m
        return (v & 1) | (x << 1)

    perm = [f(v) for v in range(1 << width)]
    ref, peak, anc = synth_perm_ops(perm, width)
    spec = _make_spec(n_in=width, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    op = "subtract" if sub else "add"
    desc = (f"Controlled modular {op}: control c=q0, x=q1..q{n}. If c=1, x <- (x {'-' if sub else '+'} {a}) "
            f"mod {m}; else x unchanged. c preserved.")
    return Instance("controlled_addsub", "", dict(n=n, m=m, a=a, sub=sub), f, spec, ref, desc=desc)


# ---------- mod_mult: f(x) = (k*x) mod m, k a unit (permutation of Z_m) ---------------
def build_mod_mult(n: int, m: int, k: int) -> Instance:
    assert math.gcd(k, m) == 1, "k must be a unit mod m"
    in_q = list(range(n))
    out_q = list(range(n))

    def f(x: int) -> int:
        return (k * x) % m if x < m else x

    perm = [f(x) for x in range(1 << n)]
    ref, peak, anc = synth_perm_ops(perm, n)
    spec = _make_spec(n_in=n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    desc = (f"Small modular multiply by a unit: f(x) = ({k} * x) mod {m} on the {n}-qubit "
            f"register (a permutation of Z_{m}; states >= {m} fixed). In place.")
    return Instance("mod_mult", "", dict(n=n, m=m, k=k), f, spec, ref, desc=desc)


# ---------- mod_inverse: f(x) = x^{-1} mod m, f(0)=0 ----------------------------------
def build_mod_inverse(n: int, m: int) -> Instance:
    assert _is_prime(m), "modular inverse permutation needs prime m so every x!=0 is a unit"
    in_q = list(range(n))
    out_q = list(range(n))

    def inv(x: int) -> int:
        if x % m == 0:
            return 0
        return pow(x % m, m - 2, m)  # Fermat; m prime

    def f(x: int) -> int:
        return inv(x) if x < m else x

    perm = [f(x) for x in range(1 << n)]
    ref, peak, anc = synth_perm_ops(perm, n)
    spec = _make_spec(n_in=n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    desc = (f"Small modular inverse: f(x) = x^(-1) mod {m} with f(0)=0 on the {n}-qubit "
            f"register (a permutation of Z_{m}). In place.")
    return Instance("mod_inverse", "", dict(n=n, m=m), f, spec, ref, desc=desc)


# ---------- sbox: arbitrary fixed boolean permutation (truth table) -------------------
def build_sbox(n: int, table: Sequence[int]) -> Instance:
    assert sorted(table) == list(range(1 << n)), "S-box must be a permutation of all 2^n values"
    in_q = list(range(n))
    out_q = list(range(n))

    def f(x: int) -> int:
        return table[x]

    perm = list(table)
    ref, peak, anc = synth_perm_ops(perm, n)
    spec = _make_spec(n_in=n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    # FULLY specify the target: the model must be told the exact permutation (the table),
    # else the task is underspecified and same-n S-boxes collapse to one prompt.
    desc = (f"Fixed boolean permutation (S-box) on {n} bits. The target is the EXACT permutation "
            f"given by this truth table, f(x) for x=0..{(1 << n) - 1} (output qubit i holds bit i "
            f"of f(x)):\n  f = {list(table)}\nIn place.")
    inst = Instance("sbox", "", dict(n=n, table=list(table)), f, spec, ref, desc=desc)
    return inst


# ---------- gf2_linear: x -> M x over GF(2), M invertible -----------------------------
def build_gf2_linear(n: int, M: Sequence[int]) -> Instance:
    """M is given as a list of n integers; row i is M[i] (a bitmask over n input bits).
    Output bit i = XOR of input bits j where bit j of M[i] is set. M must be invertible
    over GF(2). Reference: synthesized via MMD (correct for any invertible M)."""
    in_q = list(range(n))
    out_q = list(range(n))

    def f(x: int) -> int:
        y = 0
        for i in range(n):
            if bin(M[i] & x).count("1") & 1:
                y |= (1 << i)
        return y

    perm = [f(x) for x in range(1 << n)]
    assert sorted(perm) == list(range(1 << n)), "M is not invertible over GF(2)"
    ref, peak, anc = synth_perm_ops(perm, n)
    spec = _make_spec(n_in=n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    # FULLY specify the linear map as explicit XOR equations (else underspecified / collapses).
    eqs = []
    for i in range(n):
        ins = [f"x{j}" for j in range(n) if (M[i] >> j) & 1]
        eqs.append(f"y{i} = {' ^ '.join(ins) if ins else '0'}")
    desc = (f"GF(2)-linear map (M invertible) on {n} bits. The target output bits are EXACTLY:\n  "
            + "; ".join(eqs) + "\n(output qubit i holds y_i). In place.")
    return Instance("gf2_linear", "", dict(n=n, M=list(M)), f, spec, ref, desc=desc)


# ===================================================================================
# Random helpers for samplers
# ===================================================================================
def _rand_unit(m: int, rng: random.Random) -> int:
    units = [k for k in range(1, m) if math.gcd(k, m) == 1]
    return rng.choice(units) if units else 1


def _rand_perm(n: int, rng: random.Random) -> List[int]:
    p = list(range(1 << n))
    rng.shuffle(p)
    return p


def _rand_invertible_gf2(n: int, rng: random.Random) -> List[int]:
    """Sample an invertible GF(2) matrix as n row-bitmasks by rejection."""
    while True:
        M = [rng.randrange(1 << n) for _ in range(n)]
        # check rank n over GF(2)
        rows = list(M)
        rank = 0
        for col in range(n):
            piv = None
            for r in range(rank, n):
                if (rows[r] >> col) & 1:
                    piv = r
                    break
            if piv is None:
                continue
            rows[rank], rows[piv] = rows[piv], rows[rank]
            for r in range(n):
                if r != rank and ((rows[r] >> col) & 1):
                    rows[r] ^= rows[rank]
            rank += 1
        if rank == n:
            return M


# ===================================================================================
# CURRICULUM BANDS (proxy_env_design.md §e). Each band -> which families at which n/m.
# ===================================================================================
# Band -> spec describing what to sample. We keep enumeration width small (W <= ~12).
_BAND_PLAN: Dict[str, Dict] = {
    "B0": {  # n=2: const-add mod 3, 2-bit boolean perms
        "n_in": 2,
        "families": ["const_add", "sbox", "gf2_linear"],
        "moduli": {"const_add": [3]},
    },
    "B1": {  # n=3: const-add/sub mod {5,7}, controlled-add, bit-reversal-like sbox
        "n_in": 3,
        "families": ["const_add", "controlled_addsub", "sbox", "gf2_linear"],
        "moduli": {"const_add": [5, 7], "controlled_addsub": [5, 7]},
    },
    "B2": {  # n=4: reg-add mod {11,13} and mod 2^4, GF(2) linear, mod_mult small
        "n_in": 4,
        "families": ["reg_add", "gf2_linear", "mod_mult", "const_add"],
        "moduli": {"reg_add": [11, 13, 16], "mod_mult": [11, 13], "const_add": [11, 13]},
    },
    "B3": {  # n=5: modular multiply by unit k (in-place permutation), sbox
        "n_in": 5,
        "families": ["mod_mult", "sbox", "const_add"],
        "moduli": {"mod_mult": [29, 31], "const_add": [29, 31]},
    },
    "B4": {  # n=6: controlled add/sub, larger mod_mult, gf2 linear
        "n_in": 6,
        "families": ["controlled_addsub", "mod_mult", "gf2_linear"],
        "moduli": {"controlled_addsub": [59, 61], "mod_mult": [59, 61]},
    },
    "B5": {  # n=6..7: modular inverse x^{-1} mod m (the real challenge's top cost center)
        "n_in": 6,
        "families": ["mod_inverse", "mod_mult"],
        "moduli": {"mod_inverse": [59, 61], "mod_mult": [59, 61]},
    },
    "B6": {  # stretch: same families, larger; (R/HMR optional, not used here — unitary refs)
        "n_in": 7,
        "families": ["mod_inverse", "mod_mult", "sbox"],
        "moduli": {"mod_inverse": [127], "mod_mult": [113, 127]},
    },
}

BANDS = list(_BAND_PLAN.keys())


def sample_instance(family: str, band: str, rng: random.Random) -> Instance:
    """Sample one concrete Instance of `family` for `band`."""
    plan = _BAND_PLAN[band]
    n = plan["n_in"]
    moduli = plan.get("moduli", {})

    if family == "const_add":
        ms = moduli.get("const_add", _primes_le((1 << n)) or [(1 << n)])
        m = rng.choice(ms)
        a = rng.randrange(1, max(2, m))
        inst = build_const_add(n, m, a)
    elif family == "reg_add":
        ms = moduli.get("reg_add", [(1 << n)])
        m = rng.choice(ms)
        inst = build_reg_add(n, m)
    elif family == "controlled_addsub":
        ms = moduli.get("controlled_addsub", _primes_le((1 << n)) or [(1 << n)])
        m = rng.choice(ms)
        a = rng.randrange(1, max(2, m))
        sub = bool(rng.getrandbits(1))
        inst = build_controlled_addsub(n, m, a, sub=sub)
    elif family == "mod_mult":
        ms = moduli.get("mod_mult", [p for p in _primes_le((1 << n)) if p > 2] or [(1 << n)])
        m = rng.choice(ms)
        k = _rand_unit(m, rng)
        if k == 1:
            k = _rand_unit(m, rng)
        inst = build_mod_mult(n, m, k)
    elif family == "mod_inverse":
        ms = moduli.get("mod_inverse", [p for p in _primes_le((1 << n)) if p > 2])
        m = rng.choice(ms)
        inst = build_mod_inverse(n, m)
    elif family == "sbox":
        table = _rand_perm(n, rng)
        inst = build_sbox(n, table)
    elif family == "gf2_linear":
        M = _rand_invertible_gf2(n, rng)
        inst = build_gf2_linear(n, M)
    else:
        raise ValueError(f"unknown family {family}")

    inst.band = band
    inst.prompt = render_prompt(inst)
    return inst


def sample_band(band: str, rng: random.Random, n_per_family: int = 1) -> List[Instance]:
    out: List[Instance] = []
    for fam in _BAND_PLAN[band]["families"]:
        for _ in range(n_per_family):
            out.append(sample_instance(fam, band, rng))
    return out


def build_curriculum(rng: Optional[random.Random] = None,
                     n_per_family: int = 1) -> Dict[str, List[Instance]]:
    rng = rng or random.Random(0)
    return {band: sample_band(band, rng, n_per_family) for band in BANDS}


# ===================================================================================
# Prompt renderer — model-facing PROMPT text.
# ===================================================================================
RULES = (
    "OP-STREAM RULES (harness DSL, one op per line):\n"
    "  X qT                 # NOT (FREE)\n"
    "  CX qC qT             # CNOT (clifford, free of Toffoli cost)\n"
    "  CCX qC1 qC2 qT       # Toffoli  (THE cost lever: +1 per CCX)\n"
    "  SWAP qA qB           # exchange (clifford)\n"
    "  Z/CZ/CCZ, BIT_* , ... if bM   # phase / classical-bit / conditioned ops\n"
    "Cost = (avg Toffoli over inputs) x peak_width, where peak_width = max qubit index + 1.\n"
    "X/CX/SWAP/Z are free; only CCX/CCZ cost. Lower cost is better; a strict improvement\n"
    "over the reference cost wins (a tie does not). The circuit MUST: compute the target\n"
    "function on the input register for every input, be reversible (a full-state bijection),\n"
    "keep phase = 0, and return every ancilla qubit to |0>. No operand may alias\n"
    "(target != control1 != control2). Emit ONLY op lines, no commentary."
)


def _register_layout_lines(inst: Instance) -> List[str]:
    """Render the declared register I/O contract as REGISTER / APPEND_TO_REGISTER lines,
    mirroring the real harness register contract."""
    spec = inst.task_spec
    in_q = spec["in_qubits"]
    out_q = spec["out_qubits"]
    width = spec["width"]
    in_set, out_set = set(in_q), set(out_q)
    anc = [q for q in range(width) if q not in in_set and q not in out_set]
    lines = ["REGISTER r0   # INPUT register"]
    for q in in_q:
        lines.append(f"APPEND_TO_REGISTER q{q} r0")
    if out_q != in_q:
        lines.append("REGISTER r1   # OUTPUT register")
        for q in out_q:
            lines.append(f"APPEND_TO_REGISTER q{q} r1")
    else:
        lines.append("# (output register == input register; in-place task)")
    if anc:
        lines.append(f"# ancilla qubits (start and MUST end |0>): {anc}")
    return lines


def render_prompt(inst: Instance) -> str:
    spec = inst.task_spec
    in_q = spec["in_qubits"]
    out_q = spec["out_qubits"]
    width = spec["width"]
    n_in = spec["n_in"]
    layout = _register_layout_lines(inst)
    header = (
        f"TASK family={inst.family} band={inst.band or '-'} n_in={n_in}\n"
        f"{inst.desc}\n"
    )
    layout_block = (
        f"\nREGISTER LAYOUT (declared by the environment; qubit indices):\n"
        f"  total width W = {width} qubits (indices 0..{width - 1})\n"
        f"  input  register qubits : {in_q}\n"
        f"  output register qubits : {out_q}\n"
        f"  hard width cap         : {spec['max_width']} (peak_width must not exceed)\n"
        + "\n".join("  " + ln for ln in layout)
        + "\n"
    )
    ref_block = (
        f"\nREFERENCE COST TO BEAT: {spec['reference_cost']:.0f}  "
        f"(= reference Toffoli x peak_width). Strictly cheaper wins; a tie does not.\n"
    )
    return header + layout_block + ref_block + "\n" + RULES + "\n\nEmit the op-stream body now:"


# ===================================================================================
# Held-out generalization tasks (proxy_env_design.md §f) — NEVER used in training.
#   1) fused multiply-accumulate (x*y + c) mod m at an UNSEEN width/modulus.
#   2) an arbitrary fixed S-box on n=6 (truth table only).
# ===================================================================================
def build_heldout_fma(n: int = 2, m: int = 3) -> Instance:
    """Fused multiply-accumulate: f(x,y,c) = (x*y + c) mod m, OUT OF PLACE.

    Layout: x = q0..q(n-1), y = qn..q(2n-1), c = q(2n)..q(3n-1), result register
    r = q(3n)..q(4n-1) (starts |0>, holds the answer). The inputs x,y,c are PRESERVED.

    Verifier contract note: the verifier requires every qubit NOT in out_qubits to end at
    |0>. So a preserved input must be DECLARED as part of the output. We therefore set
    out_qubits = [x,y,c, result] and let f return the full (inputs || result) layout; the
    only true ancillas are the MMD scratch qubits, which are uncomputed to |0>. This is a
    genuine out-of-place fused step (result computed into a fresh |0> register, inputs intact)
    that never appears as a single training task — it composes B4 multiply + B2 modular add +
    cross-block uncompute. The whole joint permutation is realized via the generic MMD
    synthesizer, deliberately a DIFFERENT (table-driven) decomposition than the training
    references. Default (n=2, m=3) keeps the enumeration width <= 16 after MCX ancillas.
    """
    in_q = list(range(3 * n))                 # x, y, c
    res_q = list(range(3 * n, 4 * n))         # result register (fresh |0>)
    out_q = in_q + res_q                       # inputs PRESERVED + result -> all are "output"
    width = 4 * n

    def f_full(v: int) -> int:
        mask = (1 << n) - 1
        x = v & mask
        y = (v >> n) & mask
        c = (v >> (2 * n)) & mask
        r = (v >> (3 * n)) & mask              # result reg (0 on declared inputs)
        if x < m and y < m and c < m:
            r = r ^ ((x * y + c) % m)          # XOR answer onto fresh result register
        return (v & ((1 << (3 * n)) - 1)) | (r << (3 * n))

    perm = [f_full(v) for v in range(1 << width)]
    ref, peak, anc = synth_perm_ops(perm, width)

    # f over the declared n_in = 3n input qubits, output = out_q (= inputs || result).
    # Declared inputs always have the result register = |0>, so the full input value to f
    # equals the input-register value (low 3n bits); higher bits (result) are 0.
    def f(inp: int) -> int:
        mask = (1 << n) - 1
        x = inp & mask
        y = (inp >> n) & mask
        c = (inp >> (2 * n)) & mask
        result = ((x * y + c) % m) if (x < m and y < m and c < m) else 0
        # out_qubits order = in_q (preserved) then res_q (result):
        #   low 3n bits = preserved inputs == inp ; next n bits = result
        return (inp & ((1 << (3 * n)) - 1)) | (result << (3 * n))

    spec = _make_spec(n_in=3 * n, in_qubits=in_q, out_qubits=out_q, f=f, reference_ops=ref)
    desc = (f"[HELD-OUT] Fused multiply-accumulate (out of place): "
            f"f(x,y,c) = (x*y + c) mod {m}, with x=q0..q{n-1}, y=q{n}..q{2*n-1}, "
            f"c=q{2*n}..q{3*n-1}, result -> q{3*n}..q{4*n-1} (starts |0>; inputs preserved).")
    inst = Instance("heldout_fma", "HELDOUT", dict(n=n, m=m), f, spec, ref, desc=desc)
    inst.prompt = render_prompt(inst)
    return inst


def build_heldout_sbox6(seed: int = 20240606) -> Instance:
    """An arbitrary fixed S-box permutation on n=6 bits (truth table only, no structure)."""
    rng = random.Random(seed)
    n = 6
    table = list(range(1 << n))
    rng.shuffle(table)
    inst = build_sbox(n, table)
    inst.family = "heldout_sbox6"
    inst.band = "HELDOUT"
    inst.desc = ("[HELD-OUT] Arbitrary fixed S-box permutation on n=6 bits given ONLY by its "
                 "truth table (no arithmetic structure). In place.")
    inst.prompt = render_prompt(inst)
    return inst


def heldout_tasks() -> List[Instance]:
    return [build_heldout_fma(n=2, m=3), build_heldout_sbox6()]


# ===================================================================================
# Self-test / manifest generation when run directly is handled by build scripts.
# ===================================================================================
if __name__ == "__main__":
    rng = random.Random(0)
    cur = build_curriculum(rng, n_per_family=1)
    for band, insts in cur.items():
        for inst in insts:
            rep = pe.verify(inst.reference_ops, inst.task_spec)
            print(f"[{band}] {inst.family:20s} valid={rep['valid']} "
                  f"reason={rep['reason']:20s} tof={rep['toffoli']} pw={rep['peak_width']} "
                  f"cost={rep['cost']}")
