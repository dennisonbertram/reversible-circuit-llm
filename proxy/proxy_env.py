"""
proxy_env.py — fast proxy RL environment + verifier for reversible-circuit optimization,
with semantics IDENTICAL to the real ECDSA-fail simulator.

This is the most correctness-critical artifact in the project. The simulator core is an
EXACT mirror of the challenge code:

  * ~/develop/quantum-project/ecdsafail-challenge/src/sim.rs    (Simulator::apply_iter)
  * ~/develop/quantum-project/ecdsafail-challenge/src/circuit.rs (Op / OperationType /
        Op::from_text / Op::validate / analyze_ops)

The challenge simulator is a bit-packed CLASSICAL reversible simulator: each qubit/bit
holds up to 64 shots in the bit-lanes of a 64-bit word (here a Python int masked to 64
bits, so the twiddling is exact). We mirror `apply_iter` op-by-op:

    CCX:  q_target ^= cond & q_c1 & q_c2          (toffoli += cond.count_ones())
    CX:   q_target ^= cond & q_c1                 (clifford)
    X:    q_target ^= cond                         (FREE)
    SWAP: conditional 3-XOR swap                   (clifford)
    Z:    phase ^= cond & q_target                 (FREE)
    CZ:   phase ^= cond & q_target & q_c1          (clifford)
    CCZ:  phase ^= cond & q_target & q_c1 & q_c2   (toffoli += cond.count_ones())
    NEG:  phase ^= cond                            (FREE)
    R:    phase ^= q_target & rng & cond ; q_target &= !cond   (clifford)
    HMR:  bit write + phase ^= q_target&rng&cond + zero target (clifford)
    BIT_INVERT/BIT_STORE0/BIT_STORE1: classical bit writes (free)
    PUSH_CONDITION/POP_CONDITION: nested classical control (free)
    REGISTER/APPEND_TO_REGISTER/DEBUG_PRINT: no state effect (free)

Toffoli accounting (the optimization axis): each executed CCX/CCZ adds cond.count_ones()
(number of live/conditioned shots) to toffoli_gates. For an UNCONDITIONED op over a live
batch that is the batch's live-shot count. X and Z are FREE. CX/CZ/SWAP/R/HMR are
Clifford (not in the optimization cost). peak_width = max referenced qubit index + 1
(mirror analyze_ops).

Reset semantics: for the deterministic-core proxy, R is modelled as a clean reset
(rng=0 -> no phase injection; target zeroed). The phase^=target&rng punishment is only
active in the optional STRETCH band with a seeded SHAKE256 XOF (allow_reset=True), matching
the real sim's dirty-free punishment.

------------------------------------------------------------------------------------
Public API
------------------------------------------------------------------------------------
Exact-semantics core (the deliverable contract):
    parse_ops(text) -> list[Op]
    analyze_ops(ops) -> (num_qubits, num_bits, num_registers, registers)
    Simulator(num_qubits, num_bits, xof=None)        # bit-packed, mirrors sim.rs
    verify(opstream_text, task_spec) -> dict          # full 4-gate verifier
    reward(opstream_text, task_spec) -> float          # layered shaping
    format_reward(text) -> float

Backward-compatible GRPO/TRL surface (used by the trainer):
    proxy_verify(op_text, task_spec_dict) -> Report
    reward(prompts, completions, **kwargs) -> list[float]      # dispatches on arg types
    format_reward(prompts, completions, **kwargs) -> list[float]
    reward_from_report(rep) -> float
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

MASK64 = (1 << 64) - 1
U64_MAX = (1 << 64) - 1
NO_QUBIT = U64_MAX
NO_BIT = U64_MAX
NO_REG = U64_MAX

# ----------------------------------------------------------------------------------
# OperationType — discriminants match circuit.rs exactly.
# ----------------------------------------------------------------------------------
OP_NEG = 0
OP_REGISTER = 1
OP_APPEND_TO_REGISTER = 2
OP_BIT_INVERT = 3
OP_BIT_STORE0 = 4
OP_BIT_STORE1 = 5
OP_X = 6
OP_Z = 7
OP_CX = 8
OP_CZ = 9
OP_SWAP = 10
OP_R = 11
OP_HMR = 12
OP_CCX = 13
OP_CCZ = 14
OP_PUSH_CONDITION = 15
OP_POP_CONDITION = 16
OP_DEBUG_PRINT = 17

NAME_TO_KIND: Dict[str, int] = {
    "NEG": OP_NEG,
    "REGISTER": OP_REGISTER,
    "APPEND_TO_REGISTER": OP_APPEND_TO_REGISTER,
    "BIT_INVERT": OP_BIT_INVERT,
    "BIT_STORE0": OP_BIT_STORE0,
    "BIT_STORE1": OP_BIT_STORE1,
    "X": OP_X,
    "Z": OP_Z,
    "CX": OP_CX,
    "CZ": OP_CZ,
    "SWAP": OP_SWAP,
    "R": OP_R,
    "HMR": OP_HMR,
    "CCX": OP_CCX,
    "CCZ": OP_CCZ,
    "PUSH_CONDITION": OP_PUSH_CONDITION,
    "POP_CONDITION": OP_POP_CONDITION,
    "DEBUG_PRINT": OP_DEBUG_PRINT,
}
KIND_TO_NAME = {v: k for k, v in NAME_TO_KIND.items()}

# flag values for validate()
_BANNED = 0
_ALLOWED = 1
_REQUIRED = 2


class ParseError(Exception):
    """Op line cannot be parsed (mirror of a Rust panic in from_text)."""


class ValidateError(Exception):
    """Op fails Op::validate (aliasing / arity)."""


def _popcount(x: int) -> int:
    return bin(x & MASK64).count("1")


# ----------------------------------------------------------------------------------
# Op — mirror of circuit.rs Op struct + from_text + validate.
# ----------------------------------------------------------------------------------
@dataclass
class Op:
    kind: int
    q_control2: int = NO_QUBIT
    q_control1: int = NO_QUBIT
    q_target: int = NO_QUBIT
    c_target: int = NO_BIT
    c_condition: int = NO_BIT
    r_target: int = NO_REG

    def validate(self) -> None:
        # operand aliasing
        if self.q_target == self.q_control1 and self.q_target != NO_QUBIT:
            raise ValidateError(
                f"kind={KIND_TO_NAME[self.kind]} and q_target==q_control1==q{self.q_target}"
            )
        if self.q_target == self.q_control2 and self.q_target != NO_QUBIT:
            raise ValidateError(
                f"kind={KIND_TO_NAME[self.kind]} and q_target==q_control2==q{self.q_target}"
            )
        if self.q_control1 == self.q_control2 and self.q_control1 != NO_QUBIT:
            raise ValidateError(
                f"kind={KIND_TO_NAME[self.kind]} and q_control1==q_control2==q{self.q_control1}"
            )

        q_target_flag = _BANNED
        q_control1_flag = _BANNED
        q_control2_flag = _BANNED
        c_target_flag = _BANNED
        r_target_flag = _BANNED
        c_condition_flag = _BANNED

        k = self.kind
        if k == OP_DEBUG_PRINT:
            return
        elif k == OP_REGISTER:
            r_target_flag = _REQUIRED
        elif k == OP_APPEND_TO_REGISTER:
            if (self.q_target == NO_QUBIT) == (self.c_target == NO_BIT):
                raise ValidateError(
                    f"kind={KIND_TO_NAME[k]} needs exactly one qubit target or bit target"
                )
            c_target_flag = _ALLOWED
            q_target_flag = _ALLOWED
            r_target_flag = _REQUIRED
        elif k in (OP_CCX, OP_CCZ):
            c_condition_flag = _ALLOWED
            q_target_flag = _REQUIRED
            q_control1_flag = _REQUIRED
            q_control2_flag = _REQUIRED
        elif k in (OP_CX, OP_CZ, OP_SWAP):
            c_condition_flag = _ALLOWED
            q_target_flag = _REQUIRED
            q_control1_flag = _REQUIRED
        elif k in (OP_X, OP_Z, OP_R):
            c_condition_flag = _ALLOWED
            q_target_flag = _REQUIRED
        elif k == OP_NEG:
            c_condition_flag = _ALLOWED
        elif k == OP_HMR:
            c_condition_flag = _ALLOWED
            q_target_flag = _REQUIRED
            c_target_flag = _REQUIRED
        elif k in (OP_BIT_INVERT, OP_BIT_STORE0, OP_BIT_STORE1):
            c_condition_flag = _ALLOWED
            c_target_flag = _REQUIRED
        elif k == OP_PUSH_CONDITION:
            c_condition_flag = _REQUIRED
        elif k == OP_POP_CONDITION:
            pass

        if c_condition_flag == _REQUIRED and self.c_condition == NO_BIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but c_condition == NO_BIT")
        elif c_condition_flag == _BANNED and self.c_condition != NO_BIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but c_condition != NO_BIT")

        if q_target_flag == _REQUIRED and self.q_target == NO_QUBIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but q_target == NO_QUBIT")
        elif q_target_flag == _BANNED and self.q_target != NO_QUBIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but q_target != NO_QUBIT")

        if q_control1_flag == _REQUIRED and self.q_control1 == NO_QUBIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but q_control1 == NO_QUBIT")
        elif q_control1_flag == _BANNED and self.q_control1 != NO_QUBIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but q_control1 != NO_QUBIT")

        if q_control2_flag == _REQUIRED and self.q_control2 == NO_QUBIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but q_control2 == NO_QUBIT")
        elif q_control2_flag == _BANNED and self.q_control2 != NO_QUBIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but q_control2 != NO_QUBIT")

        if c_target_flag == _REQUIRED and self.c_target == NO_BIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but c_target == NO_BIT")
        elif c_target_flag == _BANNED and self.c_target != NO_BIT:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but c_target != NO_BIT")

        if r_target_flag == _REQUIRED and self.r_target == NO_REG:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but r_target == NO_REG")
        elif r_target_flag == _BANNED and self.r_target != NO_REG:
            raise ValidateError(f"kind={KIND_TO_NAME[k]} but r_target != NO_REG")

    @staticmethod
    def from_text(line: str) -> Optional["Op"]:
        words = line.split()
        if not words or words[0].startswith("#"):
            return None

        kind = NAME_TO_KIND.get(words[0])
        if kind is None:
            raise ParseError(f"Unrecognized operation type '{words[0]}'")

        out = Op(kind=kind)
        cur = 1

        def parse_u64_below_max(s: str) -> int:
            if not s.isdigit():
                raise ParseError(f"bad integer '{s}'")
            v = int(s)
            if v == U64_MAX:
                raise ParseError("value == u64::MAX")
            if v >= (1 << 64):
                raise ParseError("value overflows u64")
            return v

        if cur < len(words) and words[cur].startswith("q"):
            out.q_target = parse_u64_below_max(words[cur][1:])
            cur += 1
            if cur < len(words) and words[cur].startswith("q"):
                out.q_control1 = out.q_target
                out.q_target = parse_u64_below_max(words[cur][1:])
                cur += 1
            if cur < len(words) and words[cur].startswith("q"):
                out.q_control2 = out.q_control1
                out.q_control1 = out.q_target
                out.q_target = parse_u64_below_max(words[cur][1:])
                cur += 1

        if cur < len(words) and words[cur].startswith("b"):
            out.c_target = parse_u64_below_max(words[cur][1:])
            cur += 1
        if cur < len(words) and words[cur].startswith("r"):
            out.r_target = parse_u64_below_max(words[cur][1:])
            cur += 1
        if (
            cur + 1 < len(words)
            and words[cur] == "if"
            and words[cur + 1].startswith("b")
        ):
            out.c_condition = parse_u64_below_max(words[cur + 1][1:])
            cur += 2

        if cur < len(words) and words[cur].startswith("#"):
            pass  # trailing comment
        elif cur != len(words):
            raise ParseError(f"Failed to parse line '{line}'")

        out.validate()
        return out


def parse_ops(text: str) -> List[Op]:
    """Parse a whole op-stream text into a list of Ops (mirror Circuit::from_text)."""
    ops: List[Op] = []
    for line in text.splitlines():
        op = Op.from_text(line)
        if op is not None:
            ops.append(op)
    return ops


# ----------------------------------------------------------------------------------
# analyze_ops — mirror of circuit.rs analyze_ops.
#   registers[r] = list of ("q", idx) / ("b", idx) in little-endian append order.
# ----------------------------------------------------------------------------------
def analyze_ops(ops: Sequence[Op]) -> Tuple[int, int, int, List[List[Tuple[str, int]]]]:
    registers: List[List[Tuple[str, int]]] = []
    num_qubits = 0
    num_bits = 0
    num_registers = 0

    for op in ops:
        if op.q_control2 != NO_QUBIT:
            num_qubits = max(num_qubits, op.q_control2 + 1)
        if op.q_control1 != NO_QUBIT:
            num_qubits = max(num_qubits, op.q_control1 + 1)
        if op.q_target != NO_QUBIT:
            num_qubits = max(num_qubits, op.q_target + 1)
        if op.c_target != NO_BIT:
            num_bits = max(num_bits, op.c_target + 1)
        if op.c_condition != NO_BIT:
            num_bits = max(num_bits, op.c_condition + 1)
        if op.r_target != NO_REG:
            num_registers = max(num_registers, op.r_target + 1)
            while len(registers) <= op.r_target:
                registers.append([])
        if op.kind == OP_APPEND_TO_REGISTER:
            if op.q_target != NO_QUBIT:
                registers[op.r_target].append(("q", op.q_target))
            if op.c_target != NO_BIT:
                registers[op.r_target].append(("b", op.c_target))

    return num_qubits, num_bits, num_registers, registers


# ----------------------------------------------------------------------------------
# SHAKE256 XOF reader matching the Rust sha3 crate byte stream (stretch band only).
# ----------------------------------------------------------------------------------
class XofShake256:
    """SHAKE256 XOF reader. Reads little-endian u64s exactly like sha3's XofReader.

    sha3's `finalize_xof().read(buf)` returns successive bytes of the SHAKE256 squeeze
    of the absorbed input. hashlib.shake_256().digest(n) returns the first n bytes of
    the same (prefix-stable) stream, so reading 8 bytes at a time via growing-length
    digests reproduces the Rust byte stream exactly.
    """

    def __init__(self, seed: bytes):
        import hashlib

        self._shake = hashlib.shake_256()
        self._shake.update(seed)
        self._consumed = 0

    def read_u64(self) -> int:
        end = self._consumed + 8
        out = self._shake.digest(end)
        chunk = out[self._consumed:end]
        self._consumed = end
        return int.from_bytes(chunk, "little")


# ----------------------------------------------------------------------------------
# Simulator — exact mirror of sim.rs Simulator. 64 shots packed into bit-lanes of a
# Python int (masked to 64 bits).
# ----------------------------------------------------------------------------------
class Simulator:
    def __init__(self, num_qubits: int, num_bits: int, xof: Optional[XofShake256] = None):
        self.num_qubits = num_qubits
        self.num_bits = num_bits
        self.qubits = [0] * num_qubits
        self.bits = [0] * num_bits
        self.phase = 0
        self.xof = xof
        self.toffoli_gates = 0
        self.clifford_gates = 0

    def clear_for_shot(self) -> None:
        for i in range(self.num_qubits):
            self.qubits[i] = 0
        for i in range(self.num_bits):
            self.bits[i] = 0
        self.phase = 0

    def apply_iter(self, ops: Sequence[Op]) -> None:
        condition_stack: List[int] = []
        current_base_condition = MASK64

        for op in ops:
            cond = current_base_condition
            if op.c_condition != NO_BIT:
                cond &= self.bits[op.c_condition]
            cond &= MASK64

            executed_shots = _popcount(cond)

            k = op.kind
            # cost accounting (mirror apply_iter)
            if k in (OP_CCZ, OP_CCX):
                self.toffoli_gates += executed_shots
            elif k in (OP_CX, OP_CZ, OP_SWAP, OP_R, OP_HMR):
                self.clifford_gates += executed_shots
            # X and Z are not counted.

            if k == OP_CCX:
                v = cond & self.qubits[op.q_control1] & self.qubits[op.q_control2]
                self.qubits[op.q_target] = (self.qubits[op.q_target] ^ v) & MASK64
            elif k == OP_CX:
                v = cond & self.qubits[op.q_control1]
                self.qubits[op.q_target] = (self.qubits[op.q_target] ^ v) & MASK64
            elif k == OP_SWAP:
                q_c1 = self.qubits[op.q_control1]
                q_t = self.qubits[op.q_target]
                q_c1 ^= q_t
                q_t ^= (cond & q_c1) & MASK64
                q_c1 ^= q_t
                self.qubits[op.q_control1] = q_c1 & MASK64
                self.qubits[op.q_target] = q_t & MASK64
            elif k == OP_X:
                self.qubits[op.q_target] = (self.qubits[op.q_target] ^ cond) & MASK64
            elif k == OP_CCZ:
                v = (
                    cond
                    & self.qubits[op.q_target]
                    & self.qubits[op.q_control1]
                    & self.qubits[op.q_control2]
                )
                self.phase = (self.phase ^ v) & MASK64
            elif k == OP_CZ:
                v = cond & self.qubits[op.q_target] & self.qubits[op.q_control1]
                self.phase = (self.phase ^ v) & MASK64
            elif k == OP_Z:
                v = cond & self.qubits[op.q_target]
                self.phase = (self.phase ^ v) & MASK64
            elif k == OP_NEG:
                self.phase = (self.phase ^ cond) & MASK64
            elif k == OP_HMR:
                rng_val = self._next_rng()
                self.bits[op.c_target] &= (~cond) & MASK64
                self.bits[op.c_target] = (self.bits[op.c_target] ^ (rng_val & cond)) & MASK64
                self.phase = (self.phase ^ (self.qubits[op.q_target] & rng_val & cond)) & MASK64
                self.qubits[op.q_target] &= (~cond) & MASK64
            elif k == OP_R:
                rng_val = self._next_rng()
                self.phase = (self.phase ^ (self.qubits[op.q_target] & rng_val & cond)) & MASK64
                self.qubits[op.q_target] &= (~cond) & MASK64
            elif k == OP_BIT_INVERT:
                self.bits[op.c_target] = (self.bits[op.c_target] ^ cond) & MASK64
            elif k == OP_BIT_STORE0:
                self.bits[op.c_target] &= (~cond) & MASK64
            elif k == OP_BIT_STORE1:
                self.bits[op.c_target] = (self.bits[op.c_target] | cond) & MASK64
            elif k in (OP_APPEND_TO_REGISTER, OP_REGISTER, OP_DEBUG_PRINT):
                pass
            elif k == OP_PUSH_CONDITION:
                condition_stack.append(current_base_condition)
                current_base_condition = (current_base_condition & self.bits[op.c_condition]) & MASK64
            elif k == OP_POP_CONDITION:
                if condition_stack:
                    current_base_condition = condition_stack.pop()

    def _next_rng(self) -> int:
        if self.xof is None:
            # Deterministic-core proxy: clean reset (rng=0 => no phase injection,
            # target zeroed). The dirty-free punishment is only modelled in the
            # stretch band via the seeded XOF.
            return 0
        return self.xof.read_u64()

    # set_register / get_register: little-endian load/read, mirror sim.rs.
    def set_register(self, reg: Sequence[Tuple[str, int]], val: int, shot_idx: int) -> None:
        for i, (kind, idx) in enumerate(reg):
            bit_val = (val >> i) & 1
            if kind == "q":
                if bit_val:
                    self.qubits[idx] |= (1 << shot_idx)
                else:
                    self.qubits[idx] &= ~(1 << shot_idx) & MASK64
            else:
                if bit_val:
                    self.bits[idx] |= (1 << shot_idx)
                else:
                    self.bits[idx] &= ~(1 << shot_idx) & MASK64

    def get_register(self, reg: Sequence[Tuple[str, int]], shot_idx: int) -> int:
        v = 0
        for i, (kind, idx) in enumerate(reg):
            if kind == "q":
                b = (self.qubits[idx] >> shot_idx) & 1
            else:
                b = (self.bits[idx] >> shot_idx) & 1
            if b:
                v |= (1 << i)
        return v


# ----------------------------------------------------------------------------------
# Backward-compatible Report (used by the TRL trainer).
# ----------------------------------------------------------------------------------
@dataclass
class Report:
    valid: bool = False
    reason: str = "parse"
    toffoli: float = 0.0
    peak_width: int = 0
    cost: float = 0.0
    ref_cost: float = 0.0
    vs_ref_pct: float = 0.0
    n_ops: int = 0
    parse_progress: float = 0.0
    frac_correct: float = 0.0
    frac_ancilla_clean: float = 0.0
    error: str = ""
    witness: object = None


# ----------------------------------------------------------------------------------
# Verification internals
# ----------------------------------------------------------------------------------
def _parse_progress(text: str) -> Tuple[int, int]:
    """(n_ok, n_total) op lines that parse+validate (for dense reward credit)."""
    n_ok = 0
    n_total = 0
    for line in text.splitlines():
        toks = line.split()
        if not toks or toks[0].startswith("#"):
            continue
        n_total += 1
        try:
            Op.from_text(line)
            n_ok += 1
        except (ParseError, ValidateError):
            # stop counting at first failure to mirror the Rust panic-at-first-bad-line
            break
    return n_ok, n_total


def _enumerate_basis(ops: List[Op], width: int, allow_reset: bool, xof_seed: bytes,
                     num_qubits: int, num_bits: int):
    """Simulate the op stream on ALL 2^width basis input states, 64 inputs/batch.

    Returns (out_states, phase_bits, toffoli_total, n_states).
      out_states[i] = full `width`-qubit output state for basis input i
      phase_bits[i] = phase lane for input i
      toffoli_total = sum of executed Toffolis over all enumerated inputs
    """
    n_states = 1 << width
    out_states = [0] * n_states
    phase_bits = [0] * n_states
    toffoli_total = 0

    idx = 0
    while idx < n_states:
        batch = min(64, n_states - idx)
        xof = XofShake256(xof_seed) if allow_reset else None
        sim = Simulator(num_qubits, num_bits, xof)
        sim.clear_for_shot()
        for s in range(batch):
            inp = idx + s
            for q in range(width):
                if (inp >> q) & 1:
                    sim.qubits[q] |= (1 << s)
        sim.apply_iter(ops)
        toffoli_total += sim.toffoli_gates
        for s in range(batch):
            inp = idx + s
            st = 0
            for q in range(width):
                if (sim.qubits[q] >> s) & 1:
                    st |= (1 << q)
            out_states[inp] = st
            phase_bits[inp] = (sim.phase >> s) & 1
        idx += batch

    return out_states, phase_bits, toffoli_total, n_states


def simulate_basis_states(opstream_text: str, width: int,
                          allow_reset: bool = False, xof_seed: bytes = b"proxy-seed"):
    """Ground-truth-parity helper: parse `opstream_text`, run the bit-packed simulator over all
    2^width basis input states (full state enumeration, 64 inputs/batch), and return exactly the
    tuple the Rust `proxy_verify` driver prints:

        (toffoli_total, peak_width, final_states)

    where `final_states[i]` is the full-width output state for basis input i and `toffoli_total`
    is the total executed Toffoli count summed over all batches. Batch boundaries match the Rust
    driver (and verify()) so the two agree bit-for-bit.
    """
    ops = parse_ops(opstream_text)
    nq, nb, _nr, _regs = analyze_ops(ops)
    num_qubits = max(width, nq)
    num_bits = nb
    out_states, _phase, toffoli_total, _n = _enumerate_basis(
        ops, width, allow_reset, xof_seed, num_qubits, num_bits
    )
    return toffoli_total, nq, out_states


def simulate_full(opstream_text: str, width: int,
                  allow_reset: bool = False, xof_seed: bytes = b"proxy-seed"):
    """Full-parity helper for the broad equivalence sweep (covers phase + classical bits).

    Returns (toffoli_total, peak_width, final_states, phase_bits, bit_states) exactly matching
    the Rust `proxy_verify` driver's JSON, where for each basis input i:
      final_states[i] = full-width qubit state, phase_bits[i] = phase lane (0/1),
      bit_states[i]   = final classical-bit register value (over num_bits bits).
    """
    ops = parse_ops(opstream_text)
    nq, nb, _nr, _regs = analyze_ops(ops)
    num_qubits = max(width, nq)
    num_bits = nb

    n_states = 1 << width
    final_states = [0] * n_states
    phase_bits = [0] * n_states
    bit_states = [0] * n_states
    toffoli_total = 0

    idx = 0
    while idx < n_states:
        batch = min(64, n_states - idx)
        xof = XofShake256(xof_seed) if allow_reset else None
        sim = Simulator(num_qubits, num_bits, xof)
        sim.clear_for_shot()
        for s in range(batch):
            inp = idx + s
            for q in range(width):
                if (inp >> q) & 1:
                    sim.qubits[q] |= (1 << s)
        sim.apply_iter(ops)
        toffoli_total += sim.toffoli_gates
        for s in range(batch):
            inp = idx + s
            st = 0
            for q in range(width):
                if (sim.qubits[q] >> s) & 1:
                    st |= (1 << q)
            final_states[inp] = st
            phase_bits[inp] = (sim.phase >> s) & 1
            bst = 0
            for b in range(num_bits):
                if (sim.bits[b] >> s) & 1:
                    bst |= (1 << b)
            bit_states[inp] = bst
        idx += batch

    return toffoli_total, nq, final_states, phase_bits, bit_states


def _check_forward_reverse_identity(ops: List[Op], width: int,
                                    num_qubits: int, num_bits: int) -> bool:
    """Apply ops then the gate-reversed inverse; require identity on all basis states.
    The unitary gate set (X,CX,CCX,SWAP,Z,CZ,CCZ,NEG) + classical bit ops are involutions,
    so the reversed program is the inverse."""
    rev = list(reversed(ops))
    full = list(ops) + rev
    nq, nb, _, _ = analyze_ops(full)
    nq = max(num_qubits, nq, width)
    nb = max(num_bits, nb)

    n_states = 1 << width
    idx = 0
    while idx < n_states:
        batch = min(64, n_states - idx)
        sim = Simulator(nq, nb, None)
        sim.clear_for_shot()
        for s in range(batch):
            inp = idx + s
            for q in range(width):
                if (inp >> q) & 1:
                    sim.qubits[q] |= (1 << s)
        sim.apply_iter(full)
        for s in range(batch):
            inp = idx + s
            st = 0
            for q in range(width):
                if (sim.qubits[q] >> s) & 1:
                    st |= (1 << q)
            if st != inp:
                return False
        idx += batch
    return True


# ----------------------------------------------------------------------------------
# task_spec normalization. Accepts either the rich TaskSpec-style dict
#   {width, n_in, in_qubits, out_qubits, f, reference_cost, allow_reset, xof_seed, max_width}
# or the legacy dict {in_bits, out_bits, width, width_cap, truth_table, ref_cost}.
# ----------------------------------------------------------------------------------
def _normalize_spec(task_spec: dict) -> dict:
    s = dict(task_spec)
    # field aliases
    in_qubits = s.get("in_qubits", s.get("in_bits"))
    out_qubits = s.get("out_qubits", s.get("out_bits"))
    if in_qubits is None or out_qubits is None:
        raise KeyError("task_spec needs in_qubits/in_bits and out_qubits/out_bits")
    in_qubits = list(in_qubits)
    out_qubits = list(out_qubits)

    width = int(s.get("width", max(in_qubits + out_qubits) + 1 if (in_qubits or out_qubits) else 1))
    n_in = int(s.get("n_in", len(in_qubits)))

    # reference function f(x)->y
    if "f" in s and callable(s["f"]):
        f = s["f"]
    elif "truth_table" in s:
        tt = s["truth_table"]
        f = (lambda x, _tt=tt: _tt[x]) if isinstance(tt, dict) else tt
    else:
        raise KeyError("task_spec needs f or truth_table")

    ref_cost = float(s.get("reference_cost", s.get("ref_cost", 1.0)))
    allow_reset = bool(s.get("allow_reset", False))
    xof_seed = s.get("xof_seed", b"proxy-seed")
    if isinstance(xof_seed, str):
        xof_seed = xof_seed.encode()
    max_width = s.get("max_width", s.get("width_cap", None))
    if max_width is not None:
        max_width = int(max_width)

    return {
        "width": width,
        "n_in": n_in,
        "in_qubits": in_qubits,
        "out_qubits": out_qubits,
        "f": f,
        "reference_cost": ref_cost,
        "allow_reset": allow_reset,
        "xof_seed": xof_seed,
        "max_width": max_width,
    }


# ----------------------------------------------------------------------------------
# verify() — the full 4-gate verifier, returns a dict.
# ----------------------------------------------------------------------------------
def verify(opstream_text: str, task_spec: dict) -> Dict:
    result = {
        "valid": False,
        "toffoli": None,
        "peak_width": None,
        "cost": None,
        "vs_ref_pct": None,
        "reason": "",
        "witness": None,
        "frac_correct": 0.0,
        "frac_ancilla_clean": 0.0,
        "parse_progress": 0.0,
    }

    n_ok, n_total = _parse_progress(opstream_text)
    result["parse_progress"] = (n_ok / n_total) if n_total else 0.0

    try:
        spec = _normalize_spec(task_spec)
    except Exception as e:
        result["reason"] = f"spec: {e}"
        return result

    # 1. parse + validate (catch_unwind analog: any failure -> clean reject)
    try:
        ops = parse_ops(opstream_text)
    except Exception as e:
        result["reason"] = f"parse/validate: {e}"
        return result

    if not spec["allow_reset"]:
        for op in ops:
            if op.kind in (OP_R, OP_HMR):
                result["reason"] = "reset/HMR not allowed in this task band"
                return result

    # 2. peak width
    try:
        nq, nb, nr, regs = analyze_ops(ops)
    except Exception as e:
        result["reason"] = f"analyze: {e}"
        return result
    peak_width = nq
    result["peak_width"] = peak_width

    max_width = spec["max_width"] if spec["max_width"] is not None else spec["width"]
    if peak_width > max_width:
        result["reason"] = f"peak_width {peak_width} exceeds cap {max_width}"
        return result

    enum_width = max(spec["width"], peak_width)
    if enum_width > 16:
        result["reason"] = f"enumeration width {enum_width} too large (>16)"
        return result

    num_qubits = max(enum_width, nq)
    num_bits = nb

    # simulate all basis states
    try:
        out_states, phase_bits, toffoli_total, n_states = _enumerate_basis(
            ops, enum_width, spec["allow_reset"], spec["xof_seed"], num_qubits, num_bits
        )
    except Exception as e:
        result["reason"] = f"simulate: {e}\n{traceback.format_exc()}"
        return result

    avg_toffoli = toffoli_total / n_states
    result["toffoli"] = avg_toffoli
    result["cost"] = avg_toffoli * peak_width
    if spec["reference_cost"]:
        result["vs_ref_pct"] = result["cost"] / spec["reference_cost"]

    # 3. reversibility: induced map on the full W-qubit state must be a bijection.
    seen: Dict[int, int] = {}
    for inp in range(n_states):
        outp = out_states[inp]
        if outp in seen:
            result["reason"] = "not reversible (collision)"
            result["witness"] = {"input_a": seen[outp], "input_b": inp, "output": outp}
            return result
        seen[outp] = inp

    # 4. correctness + phase + ancilla over declared-input enumeration.
    in_qubits = spec["in_qubits"]
    out_qubits = spec["out_qubits"]
    n_in = spec["n_in"]
    n_in_states = 1 << n_in
    out_set = set(out_qubits)
    ancilla_qubits = [q for q in range(enum_width) if q not in out_set]

    n_correct = 0
    n_phase_clean = 0
    n_ancilla_clean = 0
    first_mismatch = None
    out_bits_len = len(out_qubits)

    for inp_val in range(n_in_states):
        basis = 0
        for i, q in enumerate(in_qubits):
            if (inp_val >> i) & 1:
                basis |= (1 << q)
        outp_state = out_states[basis]
        got = 0
        for i, q in enumerate(out_qubits):
            if (outp_state >> q) & 1:
                got |= (1 << i)
        expected = spec["f"](inp_val) & ((1 << out_bits_len) - 1)
        if got == expected:
            n_correct += 1
        elif first_mismatch is None:
            first_mismatch = {"input": inp_val, "got": got, "expected": expected}

        if phase_bits[basis] == 0:
            n_phase_clean += 1

        clean = True
        for q in ancilla_qubits:
            if (outp_state >> q) & 1:
                clean = False
                break
        if clean:
            n_ancilla_clean += 1

    result["frac_correct"] = n_correct / n_in_states
    result["frac_ancilla_clean"] = n_ancilla_clean / n_in_states

    if n_correct != n_in_states:
        result["reason"] = "classical_mismatch"
        result["witness"] = first_mismatch
        return result

    if n_phase_clean != n_in_states:
        result["reason"] = "phase_garbage"
        return result

    if n_ancilla_clean != n_in_states:
        result["reason"] = "ancilla_garbage"
        return result

    # 5. forward-then-reversed-inverse identity (skip if irreversible R/HMR present).
    has_irreversible = any(op.kind in (OP_R, OP_HMR) for op in ops)
    if not has_irreversible:
        if not _check_forward_reverse_identity(ops, enum_width, num_qubits, num_bits):
            result["reason"] = "forward_reverse_identity_failed"
            return result

    result["valid"] = True
    result["reason"] = "ok"
    return result


# ----------------------------------------------------------------------------------
# proxy_verify — backward-compatible Report wrapper around verify().
# ----------------------------------------------------------------------------------
def proxy_verify(op_text: str, task_spec: dict) -> Report:
    rep = Report()
    lines = [ln for ln in op_text.splitlines() if ln.split("#", 1)[0].strip()]
    rep.n_ops = len(lines)
    try:
        res = verify(op_text, task_spec)
    except Exception as e:  # never crash the RL loop
        rep.valid = False
        rep.reason = "validate"
        rep.error = f"{e}\n{traceback.format_exc()}"
        return rep

    rep.parse_progress = res["parse_progress"]
    rep.frac_correct = res["frac_correct"]
    rep.frac_ancilla_clean = res["frac_ancilla_clean"]
    rep.toffoli = res["toffoli"] or 0.0
    rep.peak_width = res["peak_width"] or 0
    rep.cost = res["cost"] or 0.0
    try:
        spec = _normalize_spec(task_spec)
        rep.ref_cost = spec["reference_cost"]
    except Exception:
        rep.ref_cost = float(task_spec.get("ref_cost", task_spec.get("reference_cost", 0.0)))
    rep.vs_ref_pct = res["vs_ref_pct"] or 0.0
    rep.witness = res["witness"]
    rep.valid = res["valid"]

    # map verify() reason strings to the legacy Report.reason vocabulary
    reason = res["reason"]
    if rep.valid:
        rep.reason = "ok"
    elif reason.startswith("parse/validate") or reason.startswith("spec") \
            or reason.startswith("analyze"):
        rep.reason = "parse"
        rep.error = reason
    elif reason.startswith("peak_width") or reason.startswith("enumeration") \
            or reason.startswith("reset/HMR"):
        rep.reason = "width_cap"
        rep.error = reason
    elif reason == "not reversible (collision)":
        rep.reason = "not_reversible"
        rep.error = reason
    elif reason == "classical_mismatch":
        rep.reason = "classical_mismatch"
        rep.error = reason
    elif reason in ("phase_garbage", "ancilla_garbage", "forward_reverse_identity_failed"):
        rep.reason = "ancilla_dirty"
        rep.error = reason
    else:
        rep.reason = "validate"
        rep.error = reason
    return rep


# ----------------------------------------------------------------------------------
# Reward shaping (proxy_env_design.md §d).
# ----------------------------------------------------------------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def reward_from_report(rep: Report) -> float:
    if rep.reason in ("parse", "validate", "width_cap"):
        return -1.0 + 0.05 * rep.parse_progress
    if rep.reason == "not_reversible":
        return 0.0 + 0.15 * rep.frac_correct
    if rep.reason == "classical_mismatch":
        return 0.0 + 0.15 * rep.frac_correct
    if rep.reason == "ancilla_dirty":
        return 0.3 + 0.10 * rep.frac_ancilla_clean
    # fully valid
    if rep.ref_cost > 0:
        cost_bonus = _clip((rep.ref_cost - rep.cost) / rep.ref_cost, -0.3, 1.0) * 1.5
    else:
        cost_bonus = 0.0
    return 1.0 + cost_bonus


def _reward_single(opstream_text: str, task_spec: dict) -> float:
    """Layered reward for one op stream against one task spec."""
    rep = proxy_verify(opstream_text, task_spec)
    return reward_from_report(rep)


# ----------------------------------------------------------------------------------
# TRL helpers.
# ----------------------------------------------------------------------------------
def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)
    return str(completion)


def _task_specs_from_kwargs(n: int, **kwargs) -> List[dict]:
    specs = kwargs.get("task_spec")
    if specs is None:
        specs = [
            {
                "in_bits": [0],
                "out_bits": [0],
                "width": 1,
                "width_cap": 1,
                "truth_table": {0: 0, 1: 1},
                "ref_cost": 1,
            }
        ] * n
    return list(specs)


# ----------------------------------------------------------------------------------
# reward() — dispatches on argument types:
#   reward(opstream_text: str, task_spec: dict) -> float        (exact-semantics single)
#   reward(prompts, completions, **kwargs) -> list[float]       (TRL batch)
# ----------------------------------------------------------------------------------
def reward(prompts=None, completions=None, **kwargs):
    # single-text form: reward(opstream_text, task_spec)
    if isinstance(prompts, str) and isinstance(completions, dict):
        return _reward_single(prompts, completions)
    # TRL batch form
    completions = completions or []
    specs = _task_specs_from_kwargs(len(completions), **kwargs)
    out: List[float] = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        spec = specs[i] if i < len(specs) else specs[-1]
        try:
            out.append(_reward_single(text, spec))
        except Exception:
            out.append(-1.0)
    return out


# ----------------------------------------------------------------------------------
# format_reward() — dispatches:
#   format_reward(text: str) -> float                            (single)
#   format_reward(prompts, completions, **kwargs) -> list[float] (TRL batch)
# ----------------------------------------------------------------------------------
def _format_reward_single(text: str) -> float:
    n_ok, n_total = _parse_progress(text)
    if n_total == 0:
        return 0.0
    return n_ok / n_total


def format_reward(prompts=None, completions=None, **kwargs):
    # single-text form: format_reward(text)
    if isinstance(prompts, str) and completions is None:
        return _format_reward_single(prompts)
    completions = completions or []
    out: List[float] = []
    for comp in completions:
        text = _extract_text(comp)
        out.append(_format_reward_single(text))
    return out


# Convenience export expected by some trainers.
proxy_cost_reward = reward


if __name__ == "__main__":
    # tiny self-test: a NOT on a 1-qubit register computing f(x)=1-x
    spec = {
        "in_qubits": [0],
        "out_qubits": [0],
        "width": 1,
        "max_width": 2,
        "f": lambda x: 1 - x,
        "reference_cost": 1.0,  # pretend ref needs 1 toffoli so a free X beats it
    }
    print("X q0 ->", verify("X q0\n", spec))
    print("reward:", reward("X q0\n", spec))
    print("format_reward:", format_reward("X q0\nCCX q0 q1 q2\n"))
    print("FOO q0 ->", verify("FOO q0\n", spec)["reason"])
