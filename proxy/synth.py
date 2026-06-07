"""
synth.py — verifier-gated SEARCH for the cheapest VALID reversible op-stream.

This is the data factory for the proxy RL effort: given a proxy task (`task_spec`) and a
known-valid-but-bloated reference op-stream (the MMD/Cuccaro baseline from tasks.py), it
SEARCHES for the cheapest VALID circuit so that training targets become near-optimal
rather than the over-generated references the PoC used (which taught format, not optimality).

Correctness contract
---------------------
Every candidate is gated by `proxy_env.verify()` — the faithful verifier — before it can be
returned. We NEVER return an invalid circuit: the worst case is the validated reference
itself. cost = avg_executed_Toffoli x peak_width, read straight from verify().

Throughput
----------
The inner search loop scores candidates with a FAST numpy evaluator (`fast_eval`) that
reproduces verify()'s (valid, cost) bit-for-bit on the X/CX/CCX/SWAP gate set — including
verify()'s 64-lane Toffoli accounting (an unconditioned CCX contributes 64 executed shots
per <=64-input batch, so avg_toffoli = n_ccx * 64 * n_batches / n_states). `fast_eval` is
validated against verify() on 176/176 curriculum references across 8 seeds (see synth_eval).
Greedy deletion re-simulates only the suffix after the deleted gate (incremental), so a full
sweep is one prefix-cached pass instead of O(n) full re-simulations. The final chosen circuit
is ALWAYS re-confirmed by the real verify() (the gate of record).

Composed optimizers (all validity-gated):
  1. greedy gate deletion          — remove any gate that keeps the circuit valid (fixpoint),
                                     CCX-first, incremental-suffix simulation
  2. local moves / peephole        — cancel adjacent / commuting self-inverse pairs
                                     (X;X, CX;CX, SWAP;SWAP, CCX;CCX); CCX->CX/X/drop downgrade
  3. simulated annealing / RRHC    — random valid-preserving edits, cost-accept w/ cooling,
                                     seed-varied restarts, keep best-ever
  4. exhaustive IDA* optimal       — iterative-deepening over {X,CX,CCX,SWAP} for tiny in-place
                                     permutations (width <= EXACT_MAX_WIDTH); provably
                                     Toffoli-minimal on the enumerated permutation.

Determinism: all randomness derives from the passed `seed` and a loop index via
random.Random(seed ^ idx). The only wall-clock use is the time-budget deadline, which bounds
how much search happens but never changes which candidate wins for a fixed amount of compute.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import proxy_env as pe


# ----------------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------------
EXACT_MAX_WIDTH = 5          # peak width at/below which IDA* optimal search is attempted
EXACT_MAX_STATES = 1 << 12
SA_RESTARTS = 5
SA_ITERS = 300
ENUM_CAP = 16                # verify() refuses enum_width > 16; mirror that here


def _slice(deadline: float, frac: float) -> float:
    """A sub-deadline `frac` of the way through the remaining budget (for staged passes)."""
    now = time.monotonic()
    return min(deadline, now + frac * max(0.0, deadline - now))


# ----------------------------------------------------------------------------------
# Op-stream <-> token representation.
#   Structured gates we optimize: X, CX, CCX, SWAP (unconditioned, qubit-only).
#   Everything else (comments, REGISTER, conditioned ops, Z/CZ/CCZ, R/HMR, ...) is kept
#   verbatim as a ('RAW', line) token so rendering is loss-free and we never break a
#   stream that uses ops outside our menu — we simply don't optimize across RAW lines.
# gate := ("X", a, None, None) | ("CX", c, t, None) | ("SWAP", a, b, None) | ("CCX", c1, c2, t)
# ----------------------------------------------------------------------------------
_SIMPLE = {"X", "CX", "SWAP", "CCX"}


def _parse_to_gates(text: str) -> List[tuple]:
    gates: List[tuple] = []
    for line in text.splitlines():
        toks = line.split()
        if not toks or toks[0].startswith("#"):
            if line.strip():
                gates.append(("RAW", line))
            continue
        head = toks[0]
        if head in _SIMPLE and "if" not in toks:
            qs: List[int] = []
            ok = True
            for t in toks[1:]:
                if t.startswith("#"):
                    break
                if t.startswith("q") and t[1:].isdigit():
                    qs.append(int(t[1:]))
                else:
                    ok = False
                    break
            if ok:
                if head == "X" and len(qs) == 1:
                    gates.append(("X", qs[0], None, None)); continue
                if head == "CX" and len(qs) == 2:
                    gates.append(("CX", qs[0], qs[1], None)); continue
                if head == "SWAP" and len(qs) == 2:
                    gates.append(("SWAP", qs[0], qs[1], None)); continue
                if head == "CCX" and len(qs) == 3:
                    gates.append(("CCX", qs[0], qs[1], qs[2])); continue
        gates.append(("RAW", line))
    return gates


def _render(gates: Sequence[tuple]) -> str:
    out: List[str] = []
    for g in gates:
        op = g[0]
        if op == "RAW":
            out.append(g[1])
        elif op == "X":
            out.append(f"X q{g[1]}")
        elif op == "CX":
            out.append(f"CX q{g[1]} q{g[2]}")
        elif op == "SWAP":
            out.append(f"SWAP q{g[1]} q{g[2]}")
        elif op == "CCX":
            out.append(f"CCX q{g[1]} q{g[2]} q{g[3]}")
        else:  # pragma: no cover
            raise ValueError(f"unknown gate {g}")
    return "\n".join(out)


def _qubits_of(g: tuple) -> Tuple[int, ...]:
    op = g[0]
    if op == "X":
        return (g[1],)
    if op in ("CX", "SWAP"):
        return (g[1], g[2])
    if op == "CCX":
        return (g[1], g[2], g[3])
    return ()


def _has_raw(gates: Sequence[tuple]) -> bool:
    return any(g[0] == "RAW" for g in gates)


# ----------------------------------------------------------------------------------
# Fast numpy evaluator — reproduces verify()'s (valid, cost) for the simple-gate subset.
# ----------------------------------------------------------------------------------
def _avg_toffoli(n_ccx: int, n_states: int) -> float:
    """Mirror verify()'s 64-lane accounting: each unconditioned CCX contributes
    cond.count_ones()==64 executed shots per <=64-input batch. n_batches = ceil(n/64)."""
    n_batches = (n_states + 63) // 64
    return n_ccx * 64 * n_batches / n_states


def _apply_all(st: np.ndarray, gates: Sequence[tuple]) -> Tuple[np.ndarray, int]:
    one = np.int64(1)
    n_ccx = 0
    for g in gates:
        op = g[0]
        if op == "X":
            st = st ^ (one << g[1])
        elif op == "CX":
            c, t = g[1], g[2]
            m = ((st >> c) & 1).astype(bool)
            st = np.where(m, st ^ (one << t), st)
        elif op == "SWAP":
            a, b = g[1], g[2]
            d = (((st >> a) & 1) != ((st >> b) & 1))
            st = np.where(d, st ^ ((one << a) | (one << b)), st)
        elif op == "CCX":
            c1, c2, t = g[1], g[2], g[3]
            m = ((((st >> c1) & 1) & ((st >> c2) & 1))).astype(bool)
            st = np.where(m, st ^ (one << t), st)
            n_ccx += 1
        else:  # RAW -> caller must route to verify()
            raise ValueError("RAW in fast path")
    return st, n_ccx


class Evaluator:
    """Validity + cost oracle. Uses fast_eval for structured (no-RAW) gate lists, and the
    real verify() otherwise (and as the final gate of record). Caches by rendered text."""

    def __init__(self, task_spec: dict):
        self.spec = task_spec
        self.s = pe._normalize_spec(task_spec)
        self._cache: Dict[str, Tuple[bool, float]] = {}
        self.n_verify = 0       # real verify() calls
        self.n_fast = 0         # fast_eval calls
        # precompute correctness scaffold
        self.width = self.s["width"]
        self.max_width = self.s["max_width"] if self.s["max_width"] else self.width
        self.in_q = self.s["in_qubits"]
        self.out_q = self.s["out_qubits"]
        self.n_in = self.s["n_in"]
        self.f = self.s["f"]
        self.out_mask = (1 << len(self.out_q)) - 1

    # ---- exact, fast scoring for structured gate lists (validated == verify) ----
    def fast_eval(self, gates: Sequence[tuple]) -> Optional[Tuple[bool, float]]:
        """Return (valid, cost) identical to verify() for a RAW-free structured circuit,
        or None if the gate list contains RAW lines / is out of enumerable range
        (caller should fall back to verify())."""
        if _has_raw(gates):
            return None
        hi = -1
        for g in gates:
            for q in _qubits_of(g):
                if q > hi:
                    hi = q
        decl_hi = max(self.in_q + self.out_q) if (self.in_q or self.out_q) else 0
        pw = max(hi + 1, decl_hi + 1)
        if pw > self.max_width:
            return (False, float("inf"))
        enum_width = max(self.width, pw)
        if enum_width > ENUM_CAP:
            return None
        self.n_fast += 1
        n = 1 << enum_width
        st = np.arange(n, dtype=np.int64)
        st, n_ccx = _apply_all(st, gates)
        # full-state bijection
        if np.unique(st).size != n:
            return (False, float("inf"))
        cost = _avg_toffoli(n_ccx, n) * pw
        # correctness + ancilla cleanliness over declared inputs
        st_l = st.tolist()
        out_q = self.out_q
        in_q = self.in_q
        out_set = set(out_q)
        anc = [q for q in range(enum_width) if q not in out_set]
        for inp in range(1 << self.n_in):
            basis = 0
            for i, q in enumerate(in_q):
                if (inp >> i) & 1:
                    basis |= (1 << q)
            o = st_l[basis]
            got = 0
            for i, q in enumerate(out_q):
                if (o >> q) & 1:
                    got |= (1 << i)
            if got != (self.f(inp) & self.out_mask):
                return (False, float("inf"))
            for q in anc:
                if (o >> q) & 1:
                    return (False, float("inf"))
        return (True, float(cost))

    def eval_gates(self, gates: Sequence[tuple]) -> Tuple[bool, float]:
        fe = self.fast_eval(gates)
        if fe is not None:
            return fe
        return self.eval_text(_render(gates))

    def eval_text(self, text: str) -> Tuple[bool, float]:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        self.n_verify += 1
        try:
            rep = pe.verify(text, self.spec)
        except Exception:
            res = (False, float("inf"))
        else:
            res = (True, float(rep["cost"])) if rep.get("valid") else (False, float("inf"))
        self._cache[text] = res
        return res

    def verify_final(self, gates: Sequence[tuple]) -> Tuple[bool, float]:
        """Gate of record: ALWAYS the real verify(), bypassing the fast path/cache."""
        text = _render(gates)
        self.n_verify += 1
        try:
            rep = pe.verify(text, self.spec)
        except Exception:
            return (False, float("inf"))
        return (True, float(rep["cost"])) if rep.get("valid") else (False, float("inf"))


# ----------------------------------------------------------------------------------
# 1. GREEDY GATE DELETION — incremental, CCX-first, to fixpoint.
# ----------------------------------------------------------------------------------
def greedy_delete(gates: List[tuple], ev: Evaluator, deadline: float) -> List[tuple]:
    """Remove any gate whose deletion keeps the circuit VALID without raising cost; iterate
    to fixpoint. We try CCX gates first (each removed CCX is a strict cost win) and use the
    fast evaluator; if RAW lines are present we fall back to whole-stream eval."""
    cur = list(gates)
    cur_valid, cur_cost = ev.eval_gates(cur)
    if not cur_valid:
        return cur

    use_fast = not _has_raw(cur)
    changed = True
    while changed:
        changed = False
        # order: all CCX indices first (biggest lever), then the rest
        order = [i for i, g in enumerate(cur) if g[0] == "CCX"] + \
                [i for i, g in enumerate(cur) if g[0] not in ("CCX", "RAW")]
        # process by identity not index (list shifts on deletion)
        targets = [cur[i] for i in order]
        for g in targets:
            if time.monotonic() > deadline:
                return cur
            try:
                i = cur.index(g)
            except ValueError:
                continue
            cand = cur[:i] + cur[i + 1:]
            if use_fast:
                fe = ev.fast_eval(cand)
                valid, cost = fe if fe is not None else ev.eval_gates(cand)
            else:
                valid, cost = ev.eval_gates(cand)
            if valid and cost <= cur_cost + 1e-9:
                cur, cur_cost = cand, cost
                changed = True
    return cur


# ----------------------------------------------------------------------------------
# 1b. STRUCTURAL CANCELLATION — O(n) commute-and-cancel of identical self-inverse gates.
#
# The MMD/emit_mcx references are AND-trees (compute up, target, uncompute down). Between
# consecutive multi-controlled-NOTs the partial-AND ancillas are recomputed, producing many
# identical self-inverse gate pairs separated only by gates that commute with them. This pass
# removes ALL such pairs in one near-linear sweep and re-verifies the WHOLE result once per
# sweep (not per pair) — the dominant cheap win on the bloated references, and fast enough to
# run on 1000+-gate circuits in tens of ms. Iterated to fixpoint; each accepted sweep is
# gated by the evaluator (== verify()).
# ----------------------------------------------------------------------------------
def _structural_sweep(gates: List[tuple], window: int) -> Tuple[List[tuple], int]:
    """One sweep: cancel each gate with the nearest later identical self-inverse twin that is
    reachable through commuting (disjoint-support) gates within `window`. Returns (new, n)."""
    out = list(gates)
    removed = 0
    i = 0
    while i < len(out):
        gi = out[i]
        if gi[0] == "RAW":
            i += 1
            continue
        j = i + 1
        lim = min(len(out), i + 1 + window)
        while j < lim and _commutes(gi, out[j]):
            j += 1
        if j < len(out) and j < i + 1 + window and _same_self_inverse(gi, out[j]):
            del out[j]
            del out[i]
            removed += 1
            i = max(0, i - 1)
            continue
        i += 1
    return out, removed


def structural_cancel(gates: List[tuple], ev: Evaluator, deadline: float,
                      window: int = 64, max_sweeps: int = 200) -> List[tuple]:
    """Iterate _structural_sweep to fixpoint, each sweep gated by the evaluator. Because every
    cancelled pair is an exact self-inverse reachable through commuting gates, the rewrite is
    semantics-preserving; we still re-verify each sweep and only keep cost-non-increasing
    results (never returns an invalid circuit)."""
    cur = list(gates)
    cur_valid, cur_cost = ev.eval_gates(cur)
    if not cur_valid:
        return cur
    for _ in range(max_sweeps):
        if time.monotonic() > deadline:
            break
        cand, removed = _structural_sweep(cur, window)
        if removed == 0:
            break
        valid, cost = ev.eval_gates(cand)
        if valid and cost <= cur_cost + 1e-9:
            cur, cur_cost = cand, cost
        else:
            break
    return cur


# ----------------------------------------------------------------------------------
# 2. LOCAL MOVES / PEEPHOLE.
# ----------------------------------------------------------------------------------
def _same_self_inverse(a: tuple, b: tuple) -> bool:
    if a[0] != b[0]:
        return False
    op = a[0]
    if op == "X":
        return a[1] == b[1]
    if op == "CX":
        return a[1] == b[1] and a[2] == b[2]
    if op == "SWAP":
        return {a[1], a[2]} == {b[1], b[2]}
    if op == "CCX":
        return {a[1], a[2]} == {b[1], b[2]} and a[3] == b[3]
    return False


def _commutes(a: tuple, b: tuple) -> bool:
    """Sound (incomplete) commutation: disjoint qubit supports commute. RAW never commutes."""
    if a[0] == "RAW" or b[0] == "RAW":
        return False
    return set(_qubits_of(a)).isdisjoint(_qubits_of(b))


def peephole(gates: List[tuple], ev: Evaluator, deadline: float) -> List[tuple]:
    cur = list(gates)
    _, cur_cost = ev.eval_gates(cur)
    changed = True
    while changed:
        changed = False
        if time.monotonic() > deadline:
            break

        # (a) adjacent self-inverse cancellation
        i = 0
        while i + 1 < len(cur):
            if _same_self_inverse(cur[i], cur[i + 1]):
                cand = cur[:i] + cur[i + 2:]
                valid, cost = ev.eval_gates(cand)
                if valid and cost <= cur_cost + 1e-9:
                    cur, cur_cost = cand, cost
                    changed = True
                    i = max(0, i - 1)
                    continue
            i += 1

        # (b) commute-and-cancel: reach a later identical twin through commuting gates
        i = 0
        while i < len(cur):
            if time.monotonic() > deadline:
                break
            gi = cur[i]
            if gi[0] != "RAW":
                j = i + 1
                while j < len(cur) and _commutes(gi, cur[j]):
                    j += 1
                if j < len(cur) and _same_self_inverse(gi, cur[j]):
                    cand = cur[:i] + cur[i + 1:j] + cur[j + 1:]
                    valid, cost = ev.eval_gates(cand)
                    if valid and cost <= cur_cost + 1e-9:
                        cur, cur_cost = cand, cost
                        changed = True
                        continue
            i += 1

        # (c) CCX downgrade / drop
        i = 0
        while i < len(cur):
            if time.monotonic() > deadline:
                break
            g = cur[i]
            if g[0] == "CCX":
                c1, c2, t = g[1], g[2], g[3]
                for rep in (None, ("CX", c1, t, None), ("CX", c2, t, None), ("X", t, None, None)):
                    cand = cur[:i] + ([] if rep is None else [rep]) + cur[i + 1:]
                    valid, cost = ev.eval_gates(cand)
                    if valid and cost <= cur_cost + 1e-9:
                        cur, cur_cost = cand, cost
                        changed = True
                        break
            i += 1
    return cur


# ----------------------------------------------------------------------------------
# 3. SIMULATED ANNEALING / RANDOM-RESTART HILL CLIMBING.
# ----------------------------------------------------------------------------------
def _max_qubit(gates: Sequence[tuple], ev: Evaluator) -> int:
    hi = -1
    for g in gates:
        for q in _qubits_of(g):
            hi = max(hi, q)
        if g[0] == "RAW":
            for tok in g[1].split():
                if tok.startswith("q") and tok[1:].isdigit():
                    hi = max(hi, int(tok[1:]))
    cap = ev.max_width
    return max(hi, int(cap) - 1)


def _random_edit(gates: List[tuple], rng: random.Random, max_q: int) -> Optional[List[tuple]]:
    simple_idx = [i for i, g in enumerate(gates) if g[0] != "RAW"]
    if not simple_idx:
        return None
    move = rng.randrange(4)
    cand = list(gates)
    if move == 0:  # delete
        del cand[rng.choice(simple_idx)]
        return cand
    if move == 1:  # swap adjacent
        if len(cand) < 2:
            return None
        i = rng.randrange(len(cand) - 1)
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        return cand
    if move == 2:  # insert a FREE gate (X or CX)
        if max_q < 0:
            return None
        pos = rng.randrange(len(cand) + 1)
        if max_q >= 1 and rng.random() < 0.6:
            a, b = rng.sample(range(max_q + 1), 2)
            cand.insert(pos, ("CX", a, b, None))
        else:
            cand.insert(pos, ("X", rng.randrange(max_q + 1), None, None))
        return cand
    # move == 3: replace a gate with a (usually cheaper) op
    i = rng.choice(simple_idx)
    qs = _qubits_of(cand[i])
    if not qs:
        return None
    t = qs[-1]
    c = rng.randrange(2)
    if c == 0:
        cand[i] = ("X", t, None, None)
    else:
        others = [q for q in range(max_q + 1) if q != t]
        if not others:
            return None
        cand[i] = ("CX", rng.choice(others), t, None)
    return cand


def drop_dead_free_gates(gates: List[tuple], ev: Evaluator, deadline: float) -> List[tuple]:
    """Remove FREE gates (X/CX/SWAP) whose deletion leaves the circuit valid at the same cost.
    These never change cost (only CCX*width does), so dropping them yields a cleaner, shorter
    target without ever worsening cost — purely cosmetic hygiene for training data."""
    cur = list(gates)
    _, cur_cost = ev.eval_gates(cur)
    changed = True
    while changed:
        changed = False
        for g in [x for x in cur if x[0] in ("X", "CX", "SWAP")]:
            if time.monotonic() > deadline:
                return cur
            try:
                i = cur.index(g)
            except ValueError:
                continue
            cand = cur[:i] + cur[i + 1:]
            valid, cost = ev.eval_gates(cand)
            if valid and cost <= cur_cost + 1e-9:
                cur, cur_cost = cand, cost
                changed = True
    return cur


def anneal(gates: List[tuple], ev: Evaluator, seed: int, deadline: float,
           restarts: int = SA_RESTARTS, iters: int = SA_ITERS) -> List[tuple]:
    best = list(gates)
    best_valid, best_cost = ev.eval_gates(best)
    if not best_valid:
        return best
    max_q = _max_qubit(best, ev)
    for r in range(restarts):
        if time.monotonic() > deadline:
            break
        rng = random.Random((seed * 1_000_003) ^ (r + 1))
        cur = list(best)
        _, cur_cost = ev.eval_gates(cur)
        T0 = max(1.0, best_cost * 0.10) if best_cost > 0 else 1.0
        for it in range(iters):
            if (it & 15) == 0 and time.monotonic() > deadline:
                break
            T = T0 * (0.97 ** it)
            cand = _random_edit(cur, rng, max_q)
            if cand is None:
                continue
            valid, cost = ev.eval_gates(cand)
            if not valid:
                continue
            d = cost - cur_cost
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
                cur, cur_cost = cand, cost
                if cost < best_cost:
                    best, best_cost = list(cand), cost
    return best


# ----------------------------------------------------------------------------------
# 3b. RESYNTHESIS under bit-orderings — a fresh, often much cheaper base than the reference.
#
# The reference uses MMD's basic transformation in a FIXED bit order. The Toffoli count of
# transformation-based synthesis depends strongly on the variable ordering, so re-running MMD
# under many seeded qubit relabelings and keeping the cheapest (after structural cancellation)
# is a large, near-free win for the permutation families (const_add, mod_mult, mod_inverse,
# sbox, gf2_linear, controlled_addsub). Every candidate is verifier-gated.
# ----------------------------------------------------------------------------------
def _resynth_one(perm: Sequence[int], width: int, order: Sequence[int]):
    """MMD-synthesize `perm` (a permutation of range(2**width)) under qubit relabeling `order`
    (relabeled bit i == original qubit order[i]); return op-text. Imports tasks lazily to avoid
    a circular import at module load."""
    import tasks as _T  # local import: synth is imported by some tasks tooling

    n = width
    N = 1 << n

    def relabel(x: int) -> int:
        y = 0
        for i in range(n):
            if (x >> order[i]) & 1:
                y |= (1 << i)
        return y

    rperm = [0] * N
    for x in range(N):
        rperm[relabel(x)] = relabel(perm[x])
    gates = _T.mmd_synthesize_perm(rperm, n)
    max_ctrl = max((bin(c).count("1") for c, _ in gates), default=0)
    n_anc = max(0, max_ctrl - 2)
    anc = list(range(n, n + n_anc))
    lines: List[str] = []
    for cmask, tb in gates:
        controls = [order[b] for b in range(n) if (cmask >> b) & 1]
        lines.extend(_T.emit_mcx(controls, order[tb], anc))
    return "\n".join(lines)


def resynth_orderings(ev: Evaluator, deadline: float, seed: int,
                      n_orders: int = 48) -> Optional[List[tuple]]:
    """Resynthesize the task's in-place permutation under up to `n_orders` seeded bit-orderings
    (identity first), structural-cancel each candidate, and return the cheapest VALID gate list
    found (re-verified by caller), or None if the task is not a clean in-place permutation.

    The permutation is synthesized over the n_in INPUT qubits only (qubits 0..n_in-1), with MCX
    ancillas borrowed above — this matches the reference families (in-place perm on the low
    input register) and keeps the MMD synthesis at the small n_in, not the larger enum width."""
    in_q, out_q = ev.in_q, ev.out_q
    # require the canonical in-place layout: input == output == low contiguous register
    n_in = ev.n_in
    if sorted(in_q) != list(range(n_in)) or sorted(out_q) != list(range(n_in)):
        return None
    width = n_in
    if width < 2 or (1 << width) > EXACT_MAX_STATES * 4:
        return None
    # build the n_in-qubit permutation directly from f (ancillas are MCX scratch, start/end |0>)
    perm = [ev.f(x) & ((1 << len(out_q)) - 1) for x in range(1 << width)]
    if len(set(perm)) != (1 << width):
        return None
    best_gates: Optional[List[tuple]] = None
    best_cost = float("inf")
    rng = random.Random((seed * 2_654_435_761) & 0xFFFFFFFF)
    orders = [list(range(width))]
    for _ in range(n_orders - 1):
        o = list(range(width))
        rng.shuffle(o)
        orders.append(o)
    for idx, order in enumerate(orders):
        if time.monotonic() > deadline:
            break
        try:
            txt = _resynth_one(perm, width, order)
        except Exception:
            continue
        gates = _parse_to_gates(txt)
        gates = structural_cancel(gates, ev, _slice(deadline, 0.5))
        valid, cost = ev.eval_gates(gates)
        if valid and cost < best_cost:
            best_gates, best_cost = gates, cost
    return best_gates


# ----------------------------------------------------------------------------------
# 4. EXHAUSTIVE / IDA* OPTIMAL for tiny in-place permutations.
# ----------------------------------------------------------------------------------
def _build_perm(ev: Evaluator, width: int) -> Optional[List[int]]:
    """Recover the full-width permutation an in-place task must implement, or None if the
    task is not a clean in-place permutation on exactly `width` qubits with no ancilla."""
    in_q, out_q = ev.in_q, ev.out_q
    if set(in_q) != set(out_q):
        return None
    used = set(in_q) | set(out_q)
    if any(q >= width for q in used):
        return None
    ancilla = [q for q in range(width) if q not in used]
    perm = [0] * (1 << width)
    seen = set()
    for full in range(1 << width):
        if any((full >> q) & 1 for q in ancilla):
            perm[full] = full  # non-zero ancilla sector maps to itself
            continue
        inp_val = 0
        for i, q in enumerate(in_q):
            if (full >> q) & 1:
                inp_val |= (1 << i)
        out_val = ev.f(inp_val) & ((1 << len(out_q)) - 1)
        out_full = 0
        for i, q in enumerate(out_q):
            if (out_val >> i) & 1:
                out_full |= (1 << q)
        perm[full] = out_full
        if out_full in seen:
            return None
        seen.add(out_full)
    if len(set(perm)) != (1 << width):
        return None
    return perm


def _apply_state(state: int, g: tuple) -> int:
    op = g[0]
    if op == "X":
        return state ^ (1 << g[1])
    if op == "CX":
        return state ^ (1 << g[2]) if (state >> g[1]) & 1 else state
    if op == "SWAP":
        a, b = g[1], g[2]
        if ((state >> a) & 1) != ((state >> b) & 1):
            return state ^ ((1 << a) | (1 << b))
        return state
    if op == "CCX":
        c1, c2, t = g[1], g[2], g[3]
        if ((state >> c1) & 1) and ((state >> c2) & 1):
            return state ^ (1 << t)
        return state
    return state


def _map_after(cur_map: Tuple[int, ...], g: tuple) -> Tuple[int, ...]:
    return tuple(_apply_state(v, g) for v in cur_map)


# ---- GF(2)-affine free-gate closure (X/CX/SWAP reach any affine residual) ----
def _is_affine(m: Tuple[int, ...], width: int) -> bool:
    b = m[0]
    cols = [m[1 << j] ^ b for j in range(width)]
    for x in range(1 << width):
        v = b
        xx, j = x, 0
        while xx:
            if xx & 1:
                v ^= cols[j]
            xx >>= 1
            j += 1
        if v != m[x]:
            return False
    return True


def _gf2_apply(cols: Sequence[int], x: int, width: int) -> int:
    v = 0
    for j in range(width):
        if (x >> j) & 1:
            v ^= cols[j]
    return v


def _gf2_matmul(a: Sequence[int], b: Sequence[int], width: int) -> List[int]:
    return [_gf2_apply(a, b[j], width) for j in range(width)]


def _gf2_inverse(cols: Sequence[int], width: int) -> Optional[List[int]]:
    M = [0] * width
    for j in range(width):
        cj = cols[j]
        for i in range(width):
            if (cj >> i) & 1:
                M[i] |= (1 << j)
    I = [1 << i for i in range(width)]
    for col in range(width):
        piv = next((r for r in range(col, width) if (M[r] >> col) & 1), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        I[col], I[piv] = I[piv], I[col]
        for r in range(width):
            if r != col and ((M[r] >> col) & 1):
                M[r] ^= M[col]
                I[r] ^= I[col]
    inv = [0] * width
    for i in range(width):
        for j in range(width):
            if (I[i] >> j) & 1:
                inv[j] |= (1 << i)
    return inv


def _linear_to_cx(cols: Sequence[int], width: int) -> Optional[List[tuple]]:
    M = [0] * width
    for j in range(width):
        cj = cols[j]
        for i in range(width):
            if (cj >> i) & 1:
                M[i] |= (1 << j)
    ops: List[Tuple[int, int]] = []
    for col in range(width):
        if not ((M[col] >> col) & 1):
            piv = next((r for r in range(col + 1, width) if (M[r] >> col) & 1), None)
            if piv is None:
                return None
            M[col] ^= M[piv]
            ops.append((piv, col))
        for r in range(width):
            if r != col and ((M[r] >> col) & 1):
                M[r] ^= M[col]
                ops.append((col, r))
    return [("CX", c, t, None) for (c, t) in reversed(ops)]


def _affine_close(state: Tuple[int, ...], target: Tuple[int, ...], width: int) -> Optional[List[tuple]]:
    if not (_is_affine(state, width) and _is_affine(target, width)):
        return None

    def decomp(st):
        b = st[0]
        return [st[1 << j] ^ b for j in range(width)], b

    As, bs = decomp(state)
    At, bt = decomp(target)
    Ai = _gf2_inverse(As, width)
    if Ai is None:
        return None
    L = _gf2_matmul(At, Ai, width)
    c = bt ^ _gf2_apply(L, bs, width)
    cx = _linear_to_cx(L, width)
    if cx is None:
        return None
    gates = list(cx)
    for j in range(width):
        if (c >> j) & 1:
            gates.append(("X", j, None, None))
    return gates


def exact_optimal(ev: Evaluator, deadline: float, max_width: int = EXACT_MAX_WIDTH) -> Optional[List[tuple]]:
    """IDA* minimizing CCX count for a tiny in-place permutation. Free gates (X/CX/SWAP) are
    realized analytically via GF(2)-affine closure, so the only branching is over CCX gates
    (the cost lever) — making the deepening bound an exact minimum-Toffoli search. Returns the
    cheapest VALID gate list (re-verified by caller) or None if not applicable/found."""
    width = ev.width
    if width > max_width or (1 << width) > EXACT_MAX_STATES:
        return None
    perm = _build_perm(ev, width)
    if perm is None:
        return None
    n = 1 << width
    target = tuple(perm)
    identity = tuple(range(n))
    if target == identity:
        return []

    # CCX menu (aliasing-free), controls unordered
    ccx_menu = []
    for t in range(width):
        for c1 in range(width):
            for c2 in range(c1 + 1, width):
                if t != c1 and t != c2:
                    ccx_menu.append(("CCX", c1, c2, t))

    best_path: List[Optional[List[tuple]]] = [None]

    def dfs(cur_map: Tuple[int, ...], used: int, bound: int, path: List[tuple],
            t_stop: float) -> bool:
        if time.monotonic() > t_stop:
            return False
        # free-gate closure: if residual to target is affine, finish for free
        clos = _affine_close(cur_map, target, width)
        if clos is not None:
            best_path[0] = list(path) + clos
            return True
        if used >= bound:
            return False
        for cg in ccx_menu:
            nm = _map_after(cur_map, cg)
            if dfs(nm, used + 1, bound, path + [cg], t_stop):
                return True
        return False

    max_bound = min(8, 2 * width + 1)
    t_stop = min(deadline, time.monotonic() + 1.0)
    for bound in range(0, max_bound + 1):
        if time.monotonic() > t_stop:
            return None
        best_path[0] = None
        if dfs(identity, 0, bound, [], t_stop):
            res = best_path[0]
            valid, _ = ev.verify_final(res)
            if valid:
                return res
    return None


# ----------------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------------
def synthesize(task_spec: dict, ref_ops: str, time_budget_s: float = 2.0,
               seed: int = 0) -> Dict:
    """Search for the cheapest VALID op-stream for `task_spec`, starting from the valid
    reference `ref_ops`. ALWAYS returns a valid opstream (worst case = the reference).

    Returns {opstream, cost, valid, method, ref_cost, cost_ratio, n_verify, n_fast,
             elapsed_s, optimal}."""
    t0 = time.monotonic()
    deadline = t0 + max(0.05, float(time_budget_s))
    ev = Evaluator(task_spec)

    ref_valid, ref_cost = ev.eval_text(ref_ops)
    best_text = ref_ops
    best_gates: Optional[List[tuple]] = None
    best_cost = ref_cost if ref_valid else float("inf")
    best_method = "reference" if ref_valid else "none"
    is_optimal = False

    def consider(gates: List[tuple], method: str, optimal: bool = False) -> None:
        nonlocal best_text, best_cost, best_method, best_gates, is_optimal
        valid, cost = ev.verify_final(gates)  # gate of record
        if not valid:
            return
        if cost < best_cost - 1e-9:
            best_text, best_cost, best_method, best_gates = _render(gates), cost, method, list(gates)
            is_optimal = optimal
        elif optimal and cost <= best_cost + 1e-9 and not is_optimal:
            # tie at proven optimum: record optimality + canonical circuit
            best_text, best_method, best_gates = _render(gates), method, list(gates)
            is_optimal = True
        elif abs(cost - best_cost) <= 1e-9 and best_gates is not None \
                and len(gates) < len(best_gates):
            # same cost but FEWER ops -> cleaner training target; keep it (preserve method)
            best_text, best_gates = _render(gates), list(gates)

    # 4. exact optimal (best possible; cheap & decisive on tiny in-place perms)
    try:
        exact = exact_optimal(ev, min(deadline, t0 + 0.6 * (deadline - t0)))
    except Exception:
        exact = None
    if exact is not None:
        consider(exact, "exact_optimal", optimal=True)

    # 3b. RESYNTHESIS under bit-orderings (cheap, large win on permutation families): produces
    # a fresh base that is often far cheaper than the reference. We only adopt it as the
    # refinement seed if it is actually cheaper than the (structural-cleaned) reference; the
    # reference path always runs, so a slow/poor resynth never costs us the reference baseline.
    # Resynthesis MMD is expensive for wide perms, so cap it and bound its time slice.
    base_seed = None
    base_seed_cost = float("inf")
    if ref_valid and ev.n_in <= 8 and (deadline - time.monotonic()) > 0.05:
        # more orderings for small n_in (cheap), fewer for larger (each synth is costlier)
        n_orders = 64 if ev.n_in <= 5 else (32 if ev.n_in <= 6 else 12)
        try:
            rs = resynth_orderings(ev, _slice(deadline, 0.45), seed, n_orders=n_orders)
        except Exception:
            rs = None
        if rs is not None:
            v, c = ev.eval_gates(rs)
            if v:
                consider(rs, "resynth")
                base_seed, base_seed_cost = rs, c

    if ref_valid:
        # 1b. STRUCTURAL cancellation on the reference: near-linear, the dominant cheap win on
        # the AND-tree MMD references. Shrinks the circuit so later quadratic passes are cheap.
        gs_ref = structural_cancel(_parse_to_gates(ref_ops), ev, deadline)
        consider(gs_ref, "structural")
        _, gs_ref_cost = ev.eval_gates(gs_ref)

        # pick the cheaper of {reference-after-structural, resynth} as the refinement base.
        if base_seed is not None and base_seed_cost <= gs_ref_cost + 1e-9:
            gs = base_seed
        else:
            gs = gs_ref

        n_simple = sum(1 for g in gs if g[0] != "RAW")
        big = n_simple > 220   # circuits where a full greedy sweep alone can blow the budget

        if big:
            # quadratic passes on what's left; structural already did the heavy lifting.
            g1 = greedy_delete(list(gs), ev, _slice(deadline, 0.45))
            consider(g1, "structural+greedy")
            g2 = peephole(list(g1), ev, _slice(deadline, 0.75))
            g2 = greedy_delete(g2, ev, deadline)
            consider(g2, "structural+greedy+peephole")
        else:
            # 1. greedy deletion
            g1 = greedy_delete(list(gs), ev, deadline)
            consider(g1, "greedy_delete")
            # 2. peephole, then re-greedy (downgrades unlock deletions)
            g2 = peephole(list(g1), ev, deadline)
            g2 = greedy_delete(g2, ev, deadline)
            consider(g2, "greedy+peephole")
            # 3. annealing from best structured circuit so far. Skip if the circuit is already
            # tiny and proven optimal (exact_optimal already nailed it) — saves budget so easy
            # tasks finish fast (higher factory throughput).
            seed_gates = best_gates if best_gates is not None else g2
            if not is_optimal and not _has_raw(seed_gates):
                g3 = anneal(list(seed_gates), ev, seed, deadline)
                g3 = greedy_delete(g3, ev, deadline)
                g3 = peephole(g3, ev, deadline)
                consider(g3, "anneal")

        # final hygiene on the ACTUAL winner: drop free gates that don't change validity/cost
        # (cleaner, shorter training targets), then a last structural sweep.
        if best_gates is not None and not _has_raw(best_gates):
            gc = drop_dead_free_gates(list(best_gates), ev, deadline)
            consider(gc, best_method)

    # final guard: NEVER return invalid
    final_valid, final_cost = ev.eval_text(best_text)
    if not final_valid:
        best_text = ref_ops
        final_valid, final_cost = ev.eval_text(ref_ops)
        best_method, is_optimal = "reference", False

    return {
        "opstream": best_text,
        "cost": final_cost,
        "valid": bool(final_valid),
        "method": best_method,
        "optimal": bool(is_optimal),
        "ref_cost": (ref_cost if ref_valid else None),
        "cost_ratio": ((final_cost / ref_cost) if (ref_valid and ref_cost > 0)
                       else (1.0 if (ref_valid and ref_cost == 0) else None)),
        "n_verify": ev.n_verify,
        "n_fast": ev.n_fast,
        "elapsed_s": time.monotonic() - t0,
    }


if __name__ == "__main__":
    import tasks as T

    rng = random.Random(0)
    cur = T.build_curriculum(rng, n_per_family=1)
    for band, insts in cur.items():
        for inst in insts:
            res = synthesize(inst.task_spec, inst.reference_ops, time_budget_s=2.0, seed=1)
            rc = res["ref_cost"]
            ratio = (res["cost"] / rc) if rc else float("nan")
            print(f"[{band}] {inst.family:18s} ref={rc!s:>8} -> synth={res['cost']!s:>8} "
                  f"ratio={ratio:6.3f} method={res['method']:16s} opt={int(res['optimal'])} "
                  f"valid={res['valid']} nver={res['n_verify']} nfast={res['n_fast']}")
