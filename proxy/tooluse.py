"""
tooluse.py — STATE-EXTERNALIZING tool for reversible-circuit synthesis.

Hypothesis under test: a small (1.5B) model can NARRATE the synthesis algorithm but
cannot reliably EXECUTE the multi-step cumulative state-tracking (Gaussian elimination,
ripple-carry) for unseen tasks. If a TOOL tracks the cumulative circuit state and the
model only has to pick ONE gate at a time (single-step: look at current-vs-target, emit
one op), it should solve tasks it plateaus on one-shot.

`ToolEnv` maintains an EMITTED op-stream (initially empty) and, after every gate, recomputes
the circuit's CURRENT effect on every declared input basis state via the faithful bit-packed
`proxy_env.Simulator`. The state — "where does the current circuit send input x" — is computed
by the tool, NEVER by the model. That externalization is the whole point.

Public surface:
    ToolEnv(task_spec)
      .render()            -> compact text view (TARGET vs CURRENT vs MISMATCHES, cost, width,
                              ancilla-dirty status; for gf2_linear also the GF(2) residual rows)
      .step(gate_text)     -> {render, n_mismatch, cost, peak_width, done, error}
      .undo()              -> drop the last emitted gate
      .solved_opstream()   -> the emitted op-stream text (when done, this passes verify valid=True)
      .done                -> bool (emitted circuit passes proxy_env.verify with valid=True)

`done` is defined by proxy_env.verify(...)["valid"] — i.e. correct on ALL declared inputs,
reversible (full-state bijection), phase 0, and every ancilla returned to |0>.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import proxy_env as pe


# Cap how many mismatched inputs we list in render(), so the view stays compact even if a
# fresh (empty) circuit mismatches on all inputs.
_MISMATCH_CAP = 16


def _normalize(task_spec: dict) -> dict:
    """Reuse proxy_env's spec normalizer so we agree with the verifier bit-for-bit."""
    return pe._normalize_spec(task_spec)


class ToolEnv:
    """Interactive, state-externalizing builder for ONE task_spec.

    The model issues one op line per turn; the tool applies it, recomputes the full effect on
    the declared inputs, and reports what still differs from the target. The model never has to
    track cumulative state — it only reacts to current-vs-target.
    """

    def __init__(self, task_spec: dict):
        self.task_spec_raw = task_spec
        self.spec = _normalize(task_spec)
        self.in_qubits: List[int] = list(self.spec["in_qubits"])
        self.out_qubits: List[int] = list(self.spec["out_qubits"])
        self.width: int = int(self.spec["width"])
        self.max_width: int = int(self.spec["max_width"]) if self.spec["max_width"] is not None else self.width
        self.n_in: int = int(self.spec["n_in"])
        self.f = self.spec["f"]
        self.out_bits_len = len(self.out_qubits)
        self.family = str(task_spec.get("family", task_spec.get("_family", "")))
        # gf2_linear detection: in-place (out==in) pure-linear map. We render the residual rows
        # whenever the family says gf2_linear OR the spec looks like an in-place n-bit map and
        # the caller flagged it. Default to the family hint from the Instance if provided.

        self.emitted: List[str] = []   # the emitted op lines, in order
        # cached state after the last successful step (recomputed lazily)
        self._cache: Optional[Dict] = None
        self._recompute()

    # ------------------------------------------------------------------ state recompute
    def _opstream_text(self) -> str:
        return "\n".join(self.emitted)

    def _enum_width(self) -> int:
        """The enumeration width the verifier would use given the current emitted ops."""
        text = self._opstream_text()
        if not text.strip():
            return self.width
        ops = pe.parse_ops(text)
        nq, _nb, _nr, _regs = pe.analyze_ops(ops)
        return max(self.width, nq)

    def _current_outputs(self) -> Tuple[List[int], List[int], int, int]:
        """Run the emitted circuit on all 2^enum_width basis inputs.

        Returns (out_states, phase_bits, peak_width, enum_width).
        out_states[x] = full-width qubit output state for full-state basis input x.
        """
        text = self._opstream_text()
        ops = pe.parse_ops(text) if text.strip() else []
        nq, nb, _nr, _regs = pe.analyze_ops(ops)
        peak_width = nq
        enum_width = max(self.width, nq)
        num_qubits = max(enum_width, nq)
        out_states, phase_bits, _tof, _n = pe._enumerate_basis(
            ops, enum_width, self.spec["allow_reset"], self.spec["xof_seed"], num_qubits, nb
        )
        return out_states, phase_bits, peak_width, enum_width

    def _recompute(self) -> None:
        """Recompute the full current view + verify() verdict and cache it."""
        text = self._opstream_text()
        # Full faithful verdict (this defines `done`).
        try:
            v = pe.verify(text, self.task_spec_raw) if text.strip() else {
                "valid": False, "reason": "empty", "toffoli": 0.0, "peak_width": 0,
                "cost": 0.0, "frac_correct": 0.0, "frac_ancilla_clean": 0.0,
            }
        except Exception as e:  # never crash the loop
            v = {"valid": False, "reason": f"verify-exc:{e}", "toffoli": 0.0,
                 "peak_width": 0, "cost": 0.0, "frac_correct": 0.0,
                 "frac_ancilla_clean": 0.0}

        # Per-declared-input current image, for the mismatch list.
        out_states, phase_bits, peak_width, enum_width = self._current_outputs()
        out_set = set(self.out_qubits)
        ancilla_qubits = [q for q in range(enum_width) if q not in out_set]

        mism: List[Tuple[int, int, int]] = []   # (x, got, want) over declared output register
        n_in_states = 1 << self.n_in
        n_phase_dirty = 0
        n_anc_dirty = 0
        for x in range(n_in_states):
            basis = 0
            for i, q in enumerate(self.in_qubits):
                if (x >> i) & 1:
                    basis |= (1 << q)
            outp_state = out_states[basis] if basis < len(out_states) else 0
            got = 0
            for i, q in enumerate(self.out_qubits):
                if (outp_state >> q) & 1:
                    got |= (1 << i)
            want = self.f(x) & ((1 << self.out_bits_len) - 1)
            if got != want:
                mism.append((x, got, want))
            if basis < len(phase_bits) and phase_bits[basis]:
                n_phase_dirty += 1
            for q in ancilla_qubits:
                if (outp_state >> q) & 1:
                    n_anc_dirty += 1
                    break

        self._cache = {
            "verify": v,
            "mismatches": mism,
            "n_mismatch": len(mism),
            "peak_width": peak_width,
            "enum_width": enum_width,
            "ancilla_qubits": ancilla_qubits,
            "n_phase_dirty": n_phase_dirty,
            "n_anc_dirty": n_anc_dirty,
            "cost": v.get("cost") or 0.0,
            "toffoli": v.get("toffoli") or 0.0,
        }

    # ------------------------------------------------------------------ properties
    @property
    def done(self) -> bool:
        return bool(self._cache and self._cache["verify"].get("valid"))

    @property
    def n_mismatch(self) -> int:
        return int(self._cache["n_mismatch"]) if self._cache else 0

    @property
    def cost(self) -> float:
        return float(self._cache["cost"]) if self._cache else 0.0

    @property
    def peak_width(self) -> int:
        return int(self._cache["peak_width"]) if self._cache else 0

    # ------------------------------------------------------------------ gf2 residual
    def _is_gf2(self) -> bool:
        if self.family == "gf2_linear":
            return True
        # heuristic fallback: in-place (out==in), n_in == width, declared inputs 0..n-1, and the
        # target is GF(2)-linear (f(0)==0 and f(a^b)==f(a)^f(b) on basis vectors).
        if self.out_qubits != self.in_qubits:
            return False
        if self.n_in != self.width:
            return False
        if self.in_qubits != list(range(self.n_in)):
            return False
        n = self.n_in
        if (self.f(0) & ((1 << self.out_bits_len) - 1)) != 0:
            return False
        # check linearity on basis pairs (cheap for small n)
        basis_img = [self.f(1 << j) & ((1 << self.out_bits_len) - 1) for j in range(n)]
        for x in range(1 << n):
            expect = 0
            for j in range(n):
                if (x >> j) & 1:
                    expect ^= basis_img[j]
            if (self.f(x) & ((1 << self.out_bits_len) - 1)) != expect:
                return False
        return True

    def _gf2_residual_rows(self) -> Optional[List[str]]:
        """For an in-place GF(2)-linear task, render the CURRENT output bits expressed in terms
        of the ORIGINAL inputs, and the TARGET rows. This lets the model do row-reduction:
        each `CX qj qi` adds (XORs) input-dependence of column j into row i.

        Returns a list of text lines, or None if not a gf2 task.
        """
        if not self._is_gf2():
            return None
        n = self.n_in
        text = self._opstream_text()
        ops = pe.parse_ops(text) if text.strip() else []
        nq, nb, _nr, _regs = pe.analyze_ops(ops)
        num_qubits = max(n, nq)
        # Run on the n basis-vector inputs e_j to read the linear action column by column.
        # current_mat[i] = bitmask over inputs j s.t. output bit i depends on input j.
        cur_rows = [0] * n
        for j in range(n):
            sim = pe.Simulator(num_qubits, nb, None)
            sim.clear_for_shot()
            sim.qubits[j] |= 1  # set input j in shot 0
            sim.apply_iter(ops)
            for i in range(n):
                if (sim.qubits[i] >> 0) & 1:
                    cur_rows[i] |= (1 << j)
        # target rows from f on basis vectors
        tgt_rows = [0] * n
        for j in range(n):
            img = self.f(1 << j) & ((1 << self.out_bits_len) - 1)
            for i in range(n):
                if (img >> i) & 1:
                    tgt_rows[i] |= (1 << j)

        def rowstr(mask: int) -> str:
            terms = [f"x{j}" for j in range(n) if (mask >> j) & 1]
            return " ^ ".join(terms) if terms else "0"

        lines = ["GF(2) RESIDUAL (output bit i in terms of ORIGINAL inputs x0..x{}):".format(n - 1)]
        for i in range(n):
            mark = "" if cur_rows[i] == tgt_rows[i] else "   <-- WRONG"
            lines.append(f"  current y{i} = {rowstr(cur_rows[i]):20s} | target y{i} = {rowstr(tgt_rows[i])}{mark}")
        lines.append("  (a `CX qj qi` XORs the dependence of qubit j INTO qubit i: row_i ^= row_j.)")
        return lines

    # ------------------------------------------------------------------ render
    def render(self) -> str:
        """Compact text view the model acts on."""
        c = self._cache
        v = c["verify"]
        lines: List[str] = []

        # COMPACT gf2 view: the residual rows fully specify what's left, so we SKIP the 2^n truth
        # table + per-input mismatch list (which bloat n>=5 traces ~10x). The model row-reduces
        # directly off the marked rows.
        gf2 = self._gf2_residual_rows()
        if gf2:
            lines.append(f"TARGET: in-place GF(2)-linear map on {self.n_in} bits "
                         f"(qubits {self.in_qubits}). Drive each output row to its target.")
            lines.extend(gf2)
            lines.append("")
            lines.append(f"EMITTED so far ({len(self.emitted)} ops): "
                         + ("; ".join(self.emitted) if self.emitted else "(none)"))
            n_wrong = sum(1 for ln in gf2 if "WRONG" in ln)
            lines.append(f"STATUS: Toffoli_cost={c['toffoli']:.3g}  peak_width={c['peak_width']}"
                         f"  (cap {self.max_width})  cost={c['cost']:.3g}"
                         f"  ancilla_dirty={'YES' if c['n_anc_dirty'] else 'no'}"
                         f"  rows_wrong={n_wrong}")
            if self.done:
                lines.append("MISMATCHES: none. CIRCUIT IS COMPLETE.")
            else:
                lines.append(f"{c['n_mismatch']} input(s) still wrong — fix the rows marked WRONG above.")
            return "\n".join(lines)

        # Non-gf2: full truth table + per-input mismatch list.
        lines.append(f"TARGET f over inputs x in [0,{(1 << self.n_in) - 1}] "
                     f"(input qubits {self.in_qubits}, output qubits {self.out_qubits}):")
        tt = []
        for x in range(1 << self.n_in):
            tt.append(f"{x}->{self.f(x) & ((1 << self.out_bits_len) - 1)}")
        for k in range(0, len(tt), 12):
            lines.append("    " + "  ".join(tt[k:k + 12]))

        # Current circuit + status.
        lines.append("")
        lines.append(f"EMITTED so far ({len(self.emitted)} ops): "
                     + ("; ".join(self.emitted) if self.emitted else "(none)"))
        lines.append(f"STATUS: Toffoli_cost={c['toffoli']:.3g}  peak_width={c['peak_width']}"
                     f"  (cap {self.max_width})  cost={c['cost']:.3g}"
                     f"  ancilla_dirty={'YES' if c['n_anc_dirty'] else 'no'}"
                     f"  phase_dirty={'YES' if c['n_phase_dirty'] else 'no'}")

        # Mismatch list (the heart of the single-step view).
        mism = c["mismatches"]
        if not mism:
            # All declared inputs correct on the output register. If not yet "done", say why.
            if self.done:
                lines.append("MISMATCHES: none. CIRCUIT IS COMPLETE.")
            else:
                lines.append(f"MISMATCHES: none on the output register, but NOT done yet "
                             f"(reason: {v.get('reason')}). "
                             f"Fix ancillas/phase/reversibility to finish.")
        else:
            lines.append(f"MISMATCHES ({len(mism)} of {1 << self.n_in} inputs wrong; "
                         f"showing up to {_MISMATCH_CAP}):")
            for (x, got, want) in mism[:_MISMATCH_CAP]:
                wb = format(want, f"0{self.out_bits_len}b")
                gb = format(got, f"0{self.out_bits_len}b")
                lines.append(f"    x={x}: got {got} (bits {gb})  want {want} (bits {wb})")
            if len(mism) > _MISMATCH_CAP:
                lines.append(f"    ... and {len(mism) - _MISMATCH_CAP} more.")

        return "\n".join(lines)

    # ------------------------------------------------------------------ step / undo
    def _validate_gate(self, gate_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
        """Parse + validate one (or a few) op line(s). Returns (clean_lines, error).

        On any problem returns (None, "<clear error>"). Also rejects out-of-range qubit indices
        (>= max_width) so the model cannot blow the width cap, and rejects R/HMR when the band
        forbids reset.
        """
        raw_lines = [ln for ln in gate_text.replace(";", "\n").splitlines()]
        clean: List[str] = []
        for ln in raw_lines:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            clean.append(s)
        if not clean:
            return None, "no op line found (reply with exactly one op, e.g. `CX q0 q2`)"

        for s in clean:
            try:
                op = pe.Op.from_text(s)
            except pe.ParseError as e:
                return None, f"parse error on '{s}': {e}"
            except pe.ValidateError as e:
                return None, f"invalid op '{s}': {e} (operands must not alias: target!=control1!=control2)"
            if op is None:
                return None, f"no op parsed from '{s}'"
            # reset/HMR ban
            if not self.spec["allow_reset"] and op.kind in (pe.OP_R, pe.OP_HMR):
                return None, f"op '{s}' uses reset (R/HMR) which is forbidden in this task band"
            # width cap: every referenced qubit index must be < max_width
            for qidx in (op.q_target, op.q_control1, op.q_control2):
                if qidx != pe.NO_QUBIT and qidx >= self.max_width:
                    return None, (f"op '{s}' references qubit q{qidx} >= width cap "
                                  f"{self.max_width}; allowed indices are 0..{self.max_width - 1}")
        return clean, None

    def step(self, gate_text: str) -> Dict:
        """Apply ONE (or a few) gate(s). On parse/validate/width error, state is unchanged and
        the returned dict has a non-empty 'error'. Otherwise append + recompute.
        """
        clean, err = self._validate_gate(gate_text)
        if err is not None:
            # state unchanged; re-render the current state so the model can retry.
            return {
                "render": self.render(),
                "n_mismatch": self.n_mismatch,
                "cost": self.cost,
                "peak_width": self.peak_width,
                "done": self.done,
                "error": err,
            }
        self.emitted.extend(clean)
        self._recompute()
        return {
            "render": self.render(),
            "n_mismatch": self.n_mismatch,
            "cost": self.cost,
            "peak_width": self.peak_width,
            "done": self.done,
            "error": "",
        }

    def undo(self) -> Dict:
        """Drop the last emitted gate (no-op if empty)."""
        if self.emitted:
            self.emitted.pop()
            self._recompute()
            err = ""
        else:
            err = "nothing to undo (op-stream is empty)"
        return {
            "render": self.render(),
            "n_mismatch": self.n_mismatch,
            "cost": self.cost,
            "peak_width": self.peak_width,
            "done": self.done,
            "error": err,
        }

    def solved_opstream(self) -> str:
        """The emitted op-stream text. When `done`, this passes proxy_env.verify(valid=True)."""
        return self._opstream_text()


# ----------------------------------------------------------------------------------
# Self-test: solve a couple of tasks by HAND through the tool, proving the contract:
#   (a) the tool tracks state correctly,
#   (b) `done` lines up with proxy_env.verify(valid=True),
#   (c) gf2_linear renders a residual the model can row-reduce.
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import tasks

    print("=" * 70)
    print("SELF-TEST 1: a tiny 2-bit gf2 map solved by hand (Gaussian elimination)")
    # y0 = x0 ^ x1, y1 = x1   ->  M rows: row0 = {x0,x1} = 0b11 = 3, row1 = {x1} = 0b10 = 2
    inst = tasks.build_gf2_linear(2, [3, 2])
    inst.family = "gf2_linear"
    spec = dict(inst.task_spec)
    spec["family"] = "gf2_linear"
    env = ToolEnv(spec)
    print(env.render())
    # y0 needs x1 added in: CX q1 q0 makes q0 = x0 ^ x1, q1 stays x1. Done.
    print("\n-- step: CX q1 q0 --")
    r = env.step("CX q1 q0")
    print(r["render"])
    print(f"done={r['done']} n_mismatch={r['n_mismatch']} cost={r['cost']}")
    assert r["done"], "expected solved after one CX"
    print("solved_opstream:\n" + env.solved_opstream())
    # cross-check with verify
    v = pe.verify(env.solved_opstream(), inst.task_spec)
    print("verify valid:", v["valid"], "cost:", v["cost"])
    assert v["valid"]

    print("\n" + "=" * 70)
    print("SELF-TEST 2: gf2_linear B1 (n=3) by hand, ZERO Toffoli (beats MMD ref)")
    rng = random.Random(60000)
    inst = tasks.sample_instance("gf2_linear", "B1", rng)
    spec = dict(inst.task_spec)
    spec["family"] = "gf2_linear"
    env = ToolEnv(spec)
    print("M:", inst.params["M"], "ref_cost:", inst.task_spec["reference_cost"])
    print(env.render())
    # Solve via in-place GF(2) elimination of the target matrix into a product of elementary
    # row ops -> CX gates. We compute the gate list programmatically here to validate the tool,
    # not the model.
    import numpy as np  # noqa

    M = inst.params["M"]
    n = inst.params["n"]
    # We want a sequence of CX (row additions) that turns the identity into M (as the circuit's
    # linear map). Equivalent: factor M into elementary matrices. Use Gaussian elimination on M
    # to reduce to identity; the inverse sequence builds M. Simpler: brute small search of CX.
    # Build target as numpy GF(2) matrix A where output_i = sum_j A[i][j] x_j.
    A = [[(M[i] >> j) & 1 for j in range(n)] for i in range(n)]

    # reduce A to identity via row ops (add row r2 into r1 == CX q_r2 q_r1 applied in REVERSE
    # for circuit build). Collect ops, then reverse for the circuit.
    a = [row[:] for row in A]
    elim = []  # (src, dst): add row src into row dst
    # forward elimination to upper triangular then to identity
    col = 0
    for col in range(n):
        # find pivot
        piv = None
        for r in range(col, n):
            if a[r][col]:
                piv = r
                break
        if piv is None:
            continue
        if piv != col:
            # swap rows col and piv via 3 adds (record as CX-equivalent SWAP). Use SWAP gate.
            elim.append(("swap", col, piv))
            a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r != col and a[r][col]:
                for k in range(n):
                    a[r][k] ^= a[col][k]
                elim.append(("add", col, r))  # row_r ^= row_col
    # `elim` reduces A -> I. The circuit that REALIZES A is the inverse: apply the inverse of
    # each elementary op in reverse order. Each "add(src,dst)" (row_dst ^= row_src) is its own
    # inverse; SWAP is its own inverse. So reverse the list and emit matching gates. But row op
    # row_dst ^= row_src on the MATRIX corresponds to CX on qubits with control=src, target=dst
    # applied in the OUTPUT-to-input direction; building A from I means applying the inverse
    # sequence. We just emit reversed(elim) and check.
    for kind, p, q in reversed(elim):
        if kind == "add":
            env.step(f"CX q{p} q{q}")
        else:
            env.step(f"SWAP q{p} q{q}")
    print("\n-- after hand elimination --")
    print(env.render())
    print(f"done={env.done} cost={env.cost} (ref {inst.task_spec['reference_cost']})")
    if env.done:
        print("SOLVED with cost", env.cost, "vs ref", inst.task_spec["reference_cost"],
              "-> Toffoli-free pure-CX beats the MMD reference.")
    print("opstream:\n" + env.solved_opstream())
