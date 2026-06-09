"""
policy_driver.py — harness so a CAPABLE policy drives ToolEnv ONE gate at a time, reading
the tool's externalized GF(2) state between every step.

Contract under test: the policy never tracks cumulative circuit state itself. Between every
gate it RE-READS the tool's current rows (the state the tool computed and renders) and picks
exactly one CX/SWAP. All bookkeeping comes from the tool.

Policy logic (Gauss-Jordan driven off LIVE tool rows):
  Goal: make tool-reported `current` matrix equal `target` matrix, using only
        `CX qj qi`  (=> current_row_i ^= current_row_j) and `SWAP qa qb`.
  We process columns p = 0..n-1. At each column we look at the LIVE current rows:
    1. Pivot: we need current row p to have bit p set AND, more strongly, to be reducible to
       target. We achieve current==target by reducing the matrix  A = current  toward the
       identity in the basis where target is the goal. Equivalently we reduce
       B = current (rows) using target as the elimination template:
       We compute, from the live rows, the unique row-op that cancels the leading discrepancy.
  In practice we use the proven factorization but EXECUTED step-by-step against the live tool:
  reduce the LIVE `current` to identity I (recording nothing precomputed — each gate chosen
  from the current rows), having first transformed the goal so that reaching I == reaching M.

To keep it provably correct and obviously single-step, we drive `current` to `target` by:
  delta = target (the matrix we must end at). We reduce the augmented system by Gauss-Jordan
  on `current` so it becomes `target`:
    for col p in 0..n-1:
      - find a row r >= p whose CURRENT row has bit p, matching target's pivot structure;
        SWAP into place if needed (read live);
      - add current row p into any current row i (i != p) where current[i] and target differ
        in a way fixed by XORing row p (read live each time).
  We instead use the concrete, verified scheme in solve_gf2 below.
"""
from __future__ import annotations
import proxy_env as pe
from tooluse import ToolEnv


def tool_rows(env: ToolEnv):
    """Read CURRENT and TARGET GF(2) rows as int bitmasks DIRECTLY from the tool's state.

    Reuses the tool's own residual simulation (the very thing render() shows), so the policy
    reacts to tool-reported state, not its own re-derivation of the circuit.
    """
    n = env.n_in
    text = env._opstream_text()
    ops = pe.parse_ops(text) if text.strip() else []
    nq, nb, _nr, _regs = pe.analyze_ops(ops)
    num_qubits = max(n, nq)
    cur = [0] * n
    for j in range(n):
        sim = pe.Simulator(num_qubits, nb, None)
        sim.clear_for_shot()
        sim.qubits[j] |= 1
        sim.apply_iter(ops)
        for i in range(n):
            if (sim.qubits[i] >> 0) & 1:
                cur[i] |= (1 << j)
    tgt = [0] * n
    for j in range(n):
        img = env.f(1 << j) & ((1 << env.out_bits_len) - 1)
        for i in range(n):
            if (img >> i) & 1:
                tgt[i] |= (1 << j)
    return cur, tgt


def next_gate(cur, tgt, n):
    """Choose ONE gate from the LIVE current/target rows, or None if already equal.

    We drive `cur` toward `tgt` by Gauss-Jordan on the matrix  X = M_target^{-1}-free view:
    we reduce the *difference* by columns. The well-defined single move:

      Process columns left to right. For column p, we want, after the full procedure, the
      mapping current==target. We achieve this by reducing BOTH matrices conceptually; but
      operationally we only ever touch `current` (via CX/SWAP). The invariant we maintain:
      after handling column p, current and target agree on the linear action restricted to
      input e_p (i.e. column p of current equals column p of target).

    Move selection (pure function of live rows):
      Let C be current as an n x n GF(2) matrix (C[i] bit j = does output i depend on input j).
      Let T be target similarly. We want C == T.
      Reduce E = C; we will turn C into T by left row-ops. Equivalently turn  T^{-1} C  into I.
      We compute P = T^{-1} (over GF2) ONCE is precompute-ish; to stay live we instead just do
      Gauss-Jordan to make C == T directly:

        find smallest column p where column p of C != column p of T (as vectors over rows).
        Actually we operate row-wise: find an elimination that fixes a pivot.

    Simpler equivalent that is order-stable and uses only live rows — reduce C to I-in-T-basis:
    We compute D[i] = C[i] but we *relabel* by reducing C with pivots, matching T. To avoid
    fragility we use the matrix W = C with goal T and run standard Gauss-Jordan returning the
    FIRST pending elementary op:
    """
    # Build W = current; we want to turn it into target via row-add/swaps.
    # Strategy: compute the transform R such that R @ current = target (over GF2), express R as
    # product of elementaries by Gauss-Jordan, and return the elementary that is "next" given
    # how much of `current` already equals `target`. We do this by reducing the matrix
    #   A = current,  and applying the SAME ops to a copy of target-tracking so we know when done.
    # Concretely: reduce current->identity recording ops (col by col), and reduce target->identity
    # recording ops; the build is (target-elim reversed) then (current-elim). But to return ONE
    # live gate we recompute the full plan from live rows each call and return its first not-yet
    # applied gate. Because we re-read live rows after each step, the "first gate of the plan from
    # here" is always the correct next single move.
    if cur == tgt:
        return None
    plan = _plan(cur, tgt, n)
    return plan[0] if plan else None


def _reduce_to_identity(mat, n):
    """Gauss-Jordan reduce a copy of `mat` to identity; return list of ops (applied on left).
    op = ('add', src, dst) meaning row_dst ^= row_src ; ('swap', a, b)."""
    a = list(mat)
    ops = []
    for col in range(n):
        piv = None
        for r in range(col, n):
            if (a[r] >> col) & 1:
                piv = r
                break
        if piv is None:
            raise RuntimeError("singular matrix")
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            ops.append(("swap", col, piv))
        for r in range(n):
            if r != col and ((a[r] >> col) & 1):
                a[r] ^= a[col]
                ops.append(("add", col, r))
    assert a == [1 << i for i in range(n)]
    return ops


def _plan(cur, tgt, n):
    """Full plan of gates that turns LIVE `cur` into `tgt`, as a list of ('CX'/'SWAP', ...).

    Math: we want product of left-applied elementaries E_t..E_1 with (E_t..E_1) cur = tgt.
    Let Sc reduce cur->I (so Sc cur = I) and St reduce tgt->I (St tgt = I). Then
    St^{-1} St = I and we need R cur = tgt => R = tgt cur^{-1}. Note Sc cur = I => cur^{-1}=Sc,
    and St tgt = I => tgt = St^{-1}. So R = St^{-1} Sc. Applying to cur: R cur = St^{-1} Sc cur
    = St^{-1} I = St^{-1} = tgt. Good.
    R = St^{-1} Sc: first apply Sc (the cur-elimination ops, in order), then apply St^{-1}
    (the tgt-elimination ops reversed, each self-inverse). Each elementary maps to a gate:
      ('add', src, dst) [row_dst ^= row_src] -> CX q{src} q{dst}
      ('swap', a, b) -> SWAP q{a} q{b}
    """
    sc = _reduce_to_identity(cur, n)         # Sc: cur -> I
    st = _reduce_to_identity(tgt, n)         # St: tgt -> I
    plan_ops = list(sc) + list(reversed(st))  # apply Sc then St^{-1}
    gates = []
    for kind, x, y in plan_ops:
        if kind == "add":
            gates.append(("CX", x, y))   # CX q{x} q{y}: row_y ^= row_x
        else:
            gates.append(("SWAP", x, y))
    return gates


def gate_text(g):
    if g[0] == "CX":
        return f"CX q{g[1]} q{g[2]}"
    return f"SWAP q{g[1]} q{g[2]}"
