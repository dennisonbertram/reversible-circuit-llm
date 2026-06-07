# Proxy RL + Eval Environment — Design (`proxy_env_design.md`)

**Status:** DESIGN ONLY (no implementation in this doc). Authoritative source-of-truth
for the op/gate model is the ECDSA-fail harness:
`~/develop/quantum-project/ecdsafail-challenge/src/circuit.rs`,
`~/develop/quantum-project/ecdsafail-challenge/src/sim.rs`,
`~/develop/quantum-project/ecdsafail-challenge/src/bin/eval_circuit.rs`.

**Purpose.** A *fast, deterministic* verifier environment for RL post-training (GRPO) and
eval whose structure is identical to the real challenge — **minimize
`cost = Toffoli_count × peak_qubit_width` for a REVERSIBLE circuit under hard
correctness + reversibility + phase constraints** — but on circuits small enough to
verify by **exhaustive enumeration of all 2^n inputs** at **>1000 rollouts/sec on one CPU
core**. Doing well here is direct evidence of the target skill because the op set, the cost
metric, the validity gates, and the op-stream text DSL are a faithful small-`n` mirror of the
secp256k1 harness.

---

## 0. What we are mirroring from the real harness (the contract)

Pulled directly from the challenge code so the skill transfers 1:1:

- **Op set (`OperationType` in `circuit.rs`):** `NEG, REGISTER, APPEND_TO_REGISTER,
  BIT_INVERT, BIT_STORE0, BIT_STORE1, X, Z, CX, CZ, SWAP, R, HMR, CCX, CCZ,
  PUSH_CONDITION, POP_CONDITION, DEBUG_PRINT`.
- **Cost accounting (`sim.rs` `apply_iter`):** `CCX` and `CCZ` each add to `toffoli_gates`;
  `CX, CZ, SWAP, R, HMR` add to `clifford_gates`; **`X` and `Z` are FREE** (classically
  tracked); register/bit/condition/debug ops are free. **Only `CCX`/`CCZ` count toward the
  Toffoli cost.** This is the single most important transferable fact: *the optimizer's job
  is to realize a target permutation with the fewest Toffoli (`CCX`) gates and the smallest
  peak width, spending free `X`/`CX`/`SWAP` liberally.*
- **Peak width (`analyze_ops` in `circuit.rs`):** `num_qubits = max(referenced qubit index)+1`.
  Width is the **highest qubit index touched anywhere in the stream + 1** — exactly the proxy's
  `peak_qubit_width`. There is no dynamic alloc/free; an ancilla "costs width" iff its index is
  the high-water mark. (Mirror this exactly: do NOT model qubit liveness intervals for cost;
  cost = top index + 1, just like the real harness.)
- **Score (`eval_circuit.rs` `write_score`):** `score = round(avg_Toffoli) × peak_qubits`,
  lower better.
- **Register layout contract (`eval_circuit.rs main`):** real harness requires exactly 4
  registers, each 256 wide; regs 0,1 are qubits (`target_x,target_y`), regs 2,3 are classical
  bits (`offset_x,offset_y`). The proxy generalizes this to a small declared I/O register
  layout (see §2) but keeps the same `REGISTER`/`APPEND_TO_REGISTER` mechanism so the model
  practices declaring its I/O registers.
- **Four validity gates (`eval_circuit.rs run_tests`):**
  1. **Classical correctness** — output registers equal the reference function on every tested
     input (real harness: 9024 Fiat-Shamir shots; proxy: ALL 2^n inputs for small n).
  2. **Phase cleanliness** — `sim.phase` must be 0 across all live shots (no kickback from
     sloppy uncompute).
  3. **Ancilla cleanup** — after zeroing the output registers, every other qubit must be `|0>`
     on every live shot (ancillas uncomputed to 0 before free).
  4. **Forward∘reverse identity** (README contract) — circuit then its gate-reversed inverse
     restores the input on every qubit. (`X→X, CX→CX, CCX→CCX, SWAP→SWAP, Z→Z, CZ→CZ,
     CCZ→CCZ` are self-inverse; the proxy enforces this by re-simulating reversed.)
- **Validation rules (`Op::validate`):** no operand aliasing (`q_target != q_control1 !=
  q_control2`), per-kind arity (e.g. `CCX` requires 2 controls + target; `X` requires only a
  target). The proxy parser reuses these exact rules so the model learns the legal op shapes.
- **Text DSL (`Op::from_text`):** `KIND [q.. ] [bN] [rN] [if bM]  # comment`. Qubit tokens are
  positional, **last `q` token is the target**, earlier ones are controls (`q_control1` then
  `q_control2`). This is the exact stream a model emits and the verifier parses. The proxy
  adopts this text format verbatim so emitted skills transfer to the real `.kmx` op stream.
- **Determinism / Fiat-Shamir:** real harness seeds test inputs + RNG from `SHAKE256` over the
  op stream (`"quantum_ecc-fiat-shamir-v2"`), so you can't tune to the test set. The proxy is
  deterministic by *full enumeration* (no sampling for small n → nothing to overfit), and for
  the larger curriculum bands that can't be fully enumerated it reuses the **same Fiat-Shamir
  trick** (seed sampled inputs from a hash of the op stream) so the anti-overfit property holds.

> Implementation language note (for when we build it): a single-file Rust crate that *reuses*
> the challenge's `circuit.rs` + `sim.rs` verbatim (the `u64`-packed 64-shots-in-parallel
> simulator). For small n the 2^n enumeration fits in a handful of 64-wide batches, so one
> `apply_iter` call verifies 64 inputs at once — this is what buys >1000 rollouts/sec.

---

## (a) Task distribution — small reversible blocks

Every task is: **"emit an op stream that maps declared input register(s) to declared output
register(s) computing function `f`, reversibly, at minimum `Toffoli × peak_width`."** All `f`
are bijections on the full state (in-place) or are made bijective via Bennett-style ancilla +
uncompute (out-of-place). Parameters: bit-width `n`, modulus `m` (small prime or `2^n`),
direction (in-place / out-of-place), and a per-task constant where relevant.

Task families (each parameterized by `n`, all chosen so the **truth table fits exhaustive
enumeration**, `n ≤ 12`):

1. **Modular adder mod small prime** `f(x) = (x + a) mod m` and `f(x,y)=(x+y) mod m`.
   - Constant-add (`a` baked in) and register-add (two quantum inputs).
   - `m` ∈ {small primes near `2^n`: 3,5,7,11,13, … up to the largest prime `< 2^n`}.
   - This is the closest small analog of the challenge's modular field arithmetic.
2. **Controlled add/sub** `f(c,x) = c ? (x±a mod m) : x`. Practices `CCX`-gated arithmetic and
   the control discipline that dominates the real circuit's conditional ops.
3. **Small modular multiply** `f(x) = (k·x) mod m` for fixed unit `k` (a permutation of
   `Z_m`), and `f(x,y) = (x·y) mod m` out-of-place into an ancilla output register. Multiply is
   the Toffoli-heavy primitive — the proxy's main "cost to be minimized" pressure, mirroring
   the field-mult/inverse cost centers that dominate the real score.
4. **Small modular inverse / modular-reduction step** `f(x) = x^{-1} mod m` (for `x≠0`; define
   `f(0)=0` to stay a bijection) and a single conditional-subtract reduction step `x ↦ x-m if
   x≥m`. These are tiny analogs of the GCD/safegcd inverse that is the real challenge's biggest
   cost center (memory: cswap/jump-GCD ≈ 25% of T) — so practicing cheap reversible
   conditional-subtract + swap chains transfers directly.
5. **Small boolean permutations** — arbitrary fixed bijections on `n` bits specified by a
   truth table: bit-reversal, cyclic rotate, a fixed S-box-like permutation, linear maps
   `x ↦ Mx` over GF(2) (`M` invertible). Tests realizing an *arbitrary* target permutation
   from the gate set, which is the general skill underneath all the arithmetic.
6. **In-place vs out-of-place variants** of (1)–(3):
   - *In-place*: output register == input register (must be a true bijection on `n` qubits;
     forces clean uncompute of any scratch — directly trains gate 3 of the validity check).
   - *Out-of-place*: inputs preserved, result written to a fresh output register; any ancilla
     used must be returned to `|0>` (Bennett compute-copy-uncompute). This is where
     **uncompute discipline** (the skill the README's "no skipping uncompute" rule rewards) is
     learned, and where the `peak_width` term creates the compute-vs-space tradeoff.

Each concrete task instance is a tuple
`(family, n, m, direction, constant, io_layout, reference_circuit)`. `reference_circuit` is a
known-correct baseline op stream (textbook construction) whose `(Toffoli, peak_width)` defines
the cost the model must beat (see §d, §h).

---

## (b) Op set + compact circuit representation the model emits

**Op set the model may emit** (subset of the real harness — same names, same semantics):

| Token | Cost | Meaning (from `sim.rs`) |
|---|---|---|
| `X qT` | free | NOT |
| `CX qC qT` | clifford | CNOT |
| `CCX qC1 qC2 qT` | **Toffoli (+1)** | Toffoli (the only cost lever besides width) |
| `SWAP qA qT` | clifford | exchange two qubits |
| `Z qT` / `CZ qC qT` / `CCZ qC1 qC2 qT` | free / clifford / **Toffoli** | phase ops (mostly for uncompute symmetry; CCZ also counts) |
| `BIT_INVERT bN` / `BIT_STORE0 bN` / `BIT_STORE1 bN` | free | classical-bit writes |
| `REGISTER rN` | free | declare a register |
| `APPEND_TO_REGISTER (qN\|bN) rN` | free | bind a qubit/bit into a register (I/O contract) |
| `... if bM` | — | classical-conditioned execution (mirrors `c_condition`) |
| `PUSH_CONDITION bM` / `POP_CONDITION` | free | scoped conditioning |

`R`/`HMR` (measurement-based uncompute) are **allowed but optional** in an advanced
curriculum band; the core bands use unitary-only `X/CX/CCX/SWAP` so the truth-table verifier is
purely deterministic. (When `R`/`HMR` are enabled, the verifier reuses `sim.rs`'s seeded RNG and
checks the *phase=0 / ancilla=0* gates exactly as the real harness does, so measured-uncompute
skills also transfer.)

**Compact representation (the op-stream DSL):** exactly the real harness text format, one op
per line:

```
# task header is provided by the env, NOT emitted by the model:
#   REGISTER r0 ; APPEND_TO_REGISTER q0 r0 ; ... (declares IN/OUT/ancilla regs + width)
# model emits the body:
REGISTER r0
APPEND_TO_REGISTER q0 r0
APPEND_TO_REGISTER q1 r0
CCX q0 q1 q2          # write carry into ancilla q2
CX q2 q3
X q0
CCX q0 q2 q4 if b0    # conditional Toffoli
SWAP q3 q5
```

- Whitespace-tokenized, `#` comments ignored — byte-for-byte the same grammar as
  `Op::from_text`, so a model trained here emits streams that drop straight into a `.kmx` file.
- The env supplies a fixed **prompt schema**: the task spec (family, `n`, `m`, direction), the
  declared input/output/ancilla register layout (qubit-index ranges), the reference cost to
  beat, and the rule list. The model emits ONLY the op-stream body.
- **Canonical scoring form** the verifier reports back: `{valid, toffoli, peak_width, cost,
  vs_ref_pct}` plus a parse/validity error string when invalid (for dense feedback / SFT).

---

## (c) Verifier algorithm

Deterministic, panic-free, returns a structured report. Steps:

1. **Parse** the emitted text with the real-harness grammar (`Op::from_text` semantics).
   Reject on: unknown token, bad arity, operand aliasing (`Op::validate` rules), reference to a
   qubit/bit index outside the declared layout's allowed scratch budget, register-shape
   violation (output reg not the contracted width / wrong qubit-vs-bit type). → `valid=false,
   reason=parse/validate`. (No Toffoli credit for malformed streams — same as the real harness
   rejecting a forged `ops.bin`.)
2. **Compute peak width** = `max referenced qubit index + 1` (== `analyze_ops`). Reject if it
   exceeds the task's hard width cap (`ref_width + slack`), so the model can't trivially buy
   correctness with unbounded ancilla. (Width is a *scored* term, not just a cap — see §d.)
3. **Check reversibility (bijectivity) by exhaustive enumeration for small n.** Build the full
   `2^W` truth table by simulating the op stream (reusing `sim.rs`, 64 inputs per batch). The
   map on the *full* `W`-qubit state must be a **permutation** (each output state hit exactly
   once). Cheap exact test: simulate all `2^W` basis states (W ≤ ~16 enumerable; the core
   curriculum keeps total declared qubits `W ≤ 12` so `2^W ≤ 4096` = 64 batches). If any
   collision → not reversible → `valid=false`. For bands where `2^W` is too large, fall back to
   **sampled** bijectivity (hash-seeded random basis states; a collision is a witness of
   non-reversibility) — same Fiat-Shamir anti-overfit seeding as the real harness.
4. **Check it computes the target `f` on all/enough inputs.** For every input assignment to the
   declared input register(s) (ancillas initialized `|0>`, all `2^n_in` of them enumerated),
   read the output register(s) and compare to `f`. Mismatch → `valid=false,
   reason=classical_mismatch` (records count + first witness, like the harness).
5. **Phase + ancilla gates (mirror `run_tests`):** after the forward pass, `sim.phase` must be
   0 on all live shots (phase clean); after zeroing the *output* registers, every other qubit
   must be `|0>` on all live shots (ancilla uncomputed). Either failing → `valid=false`.
6. **Forward∘reverse identity:** apply the stream then its reversed-gate inverse from a random
   (hash-seeded) live state; require the original state restored. Catches subtle
   non-self-inverse / conditioned-op asymmetries.
7. **Cost:** if all gates pass, `toffoli = #CCX + #CCZ executed` (averaged over shots if any
   conditioning makes it input-dependent, exactly like `avg_tof`), `peak_width` from step 2,
   `cost = toffoli × peak_width`. Report `vs_ref_pct = cost / ref_cost`.

All steps run inside `catch_unwind` so a pathological stream yields `valid=false`, never a
crash (matches the harness's hardened loader).

---

## (d) Reward shaping

Layered so the gradient is informative even for mostly-broken outputs, but **validity is a hard
gate** (you can never out-score a correct circuit by being a faster *wrong* one — the README's
core principle). Let `ref_cost` be the reference circuit's `Toffoli × width`.

```
if not parseable/valid-shape:        r = -1.0 + 0.05 * parse_progress      # tiny dense credit
elif reversible but wrong f:         r =  0.0 + 0.15 * frac_inputs_correct  # partial correctness
elif correct f but dirty (phase/    r =  0.3 + 0.10 * frac_ancilla_clean   # "almost there"
     ancilla/identity fails):
else (fully VALID):                  r =  1.0 + cost_bonus
     cost_bonus = clip( (ref_cost - cost) / ref_cost , -0.3, +1.0 ) * 1.5
        # ties ref → +0;  matches challenge "strict < best" rule (a tie is not a win)
        # +25% cheaper → +0.375 ;  2× cheaper → +1.0 (clipped)
        # slightly worse-but-valid still beats any invalid (floor of validity is +0.7)
```

Properties:
- **Validity gate dominates:** any fully valid circuit (≥ +0.7 even if costlier) strictly beats
  any invalid one (≤ +0.3). The model can never learn to cheat correctness for cost — same
  invariant the real harness enforces.
- **Partial credit** on (i) parse progress, (ii) fraction of inputs correct, (iii) fraction of
  ancilla clean — gives GRPO a usable signal on near-misses, which is essential because random
  initial policies almost never emit a valid stream.
- **Cost improvement vs reference** is the only thing that grows reward past 1.0, mapping
  exactly to the challenge objective. Tie-with-reference = no bonus, mirroring the leaderboard's
  *strict <* rule (from memory: a tie submission was rejected).
- **Width and Toffoli both enter through `cost`** (their product), so the model feels the same
  6-phase-co-bind / compute-vs-space tradeoff that defines the real frontier — e.g. an
  out-of-place multiply can cut Toffoli by adding an ancilla, but only wins if the ancilla
  doesn't raise the high-water width. This is the exact tension memory flags as the live
  frontier ("self-hosted square saves 16,766 T but raises peak 1434→1542").
- Optional **per-token length penalty** (tiny, e.g. `-1e-3 * n_ops`) to discourage no-op
  padding; off by default to avoid distorting the cost signal.

---

## (e) Curriculum (2–3 bits up)

Bands advance when the policy's rolling valid-rate > 80% AND median `vs_ref_pct ≤ 1.0` on the
band. Width stays small so enumeration is exhaustive throughout the core.

| Band | n (input bits) | tasks | total qubits W (enum size) | what it teaches |
|---|---|---|---|---|
| B0 | 2 | const-add mod {3}, 2-bit boolean perms (swap, bit-flip), in-place | W ≤ 4 (16) | legal op shapes, the DSL, reach a target perm |
| B1 | 3 | const-add/sub mod {5,7}, controlled-add, bit-reversal | W ≤ 6 (64) | conditioning (`CCX`, `if bM`), uncompute a single ancilla |
| B2 | 4 | reg-add mod {11,13}, modular reduce step, GF(2) linear maps | W ≤ 8 (256) | multi-qubit carry chains, Toffoli minimization |
| B3 | 5 | small modular multiply by unit `k` (in-place perm) | W ≤ 9 (512) | permutation realization w/ scratch, width pressure |
| B4 | 6 | out-of-place `x·y mod m`, conditional add/sub | W ≤ 11 (2048) | Bennett compute-uncompute, phase/ancilla cleanliness |
| B5 | 6–7 | modular inverse `x^{-1} mod m` (cond-sub + swap chains) | W ≤ 12 (4096) | GCD-like inverse — the real challenge's top cost center |
| B6 (stretch) | 8–10 | measured-uncompute (`R`/`HMR`) variants of B4/B5; Fiat-Shamir sampled verify | sampled | measurement-based uncompute + anti-overfit verify |

Within a band, instances are sampled over `(m, constant, direction, io_layout)` so the model
can't memorize a single circuit. Reference circuits are regenerated per instance.

---

## (f) Held-out generalization task type (NOT used in training)

To test *transfer of the skill* rather than memorization, the eval-only family is:

**"Modular multiply–accumulate / fused step" — `f(x,y,c) = (x·y + c) mod m`, out-of-place,**
at a held-out width (e.g. `n=7`, `m` a prime not seen in training), **with a reference circuit
built from a different textbook decomposition** than any training reference. This composition
(multiply followed by a conditional modular add, with shared ancilla that must be uncomputed
across the two sub-blocks) never appears in training as a single task, yet it is built entirely
from the sub-skills the curriculum teaches: out-of-place multiply (B4), modular add (B2),
conditional reduce (B2/B5), and clean cross-block uncompute (B4/B5). A model that *learned the
method* (cheap Toffoli realization + width-aware uncompute) should produce a valid,
cost-competitive circuit; a model that memorized per-task circuits will fail. **Second held-out
probe:** an arbitrary fixed S-box permutation on `n=6` given only by its truth table (no
arithmetic structure) — tests pure "realize this permutation cheaply" generalization. Held-out
families are scored on the same `cost` metric and reported separately from the training bands.

---

## (g) Estimated rollout cost + why it is fast

- **Verification = simulation, not solving.** The `sim.rs` simulator runs **64 inputs in
  parallel per `u64`** with branch-free bit-twiddling. For core bands `W ≤ 12` → `2^W ≤ 4096`
  basis states = `≤ 64` batches of one `apply_iter` pass. A small circuit is `O(10²)` ops, so a
  full exhaustive verify is `≤ 64 × ~200` op-applications ≈ `~1.3e4` u64 ops, plus a `2^W`
  permutation-collision check (`≤ 4096` bucket inserts). On a modern core at ~`10⁸–10⁹` simple
  ops/s that is **tens of microseconds per rollout** for small bands, well under a millisecond
  even for B5.
- **No cryptographic reference math per shot.** Unlike the real harness (which does a
  256-bit scalar-mul `curve.mul` per shot to derive each test point), the proxy's reference `f`
  is a tiny table lookup or `u64` modular op — essentially free. This removes the harness's
  dominant per-shot cost.
- **No process spawn, no file I/O.** The real `ecdsafail run` path is `build_circuit` →
  `ops.bin` (552 MB!) → `eval_circuit` across two processes (~seconds–minutes). The proxy is a
  single in-process function call on an in-memory op vector.
- **Throughput estimate:** with per-rollout verify in the 10–300 µs range across bands, a single
  CPU core sustains **>1000 rollouts/sec** (B0–B3 are ~5–30k/s; B5 with full 4096-state enum is
  ~3–10k/s; only the sampled stretch band approaches the 1k floor). Batching rollouts across
  cores scales linearly. This is **3–5 orders of magnitude faster** than the real harness, which
  is exactly why it is usable as a GRPO inner loop.
- **Determinism:** full enumeration (or hash-seeded sampling) → identical reward for identical
  op stream → stable RL credit assignment and reproducible eval.

---

## (h) Explicit mapping back to ECDSA-fail (why winning here = the target skill)

| Proxy element | ECDSA-fail harness element | Same? |
|---|---|---|
| Cost `= Toffoli × peak_width`, lower better | `score = round(avg_Toffoli) × peak_qubits` (`write_score`) | **Identical metric** |
| Toffoli = `#CCX/#CCZ`; `X` free, `CX/SWAP` clifford | `sim.rs apply_iter` cost accounting | **Identical** |
| Peak width = max qubit index + 1 | `analyze_ops num_qubits` | **Identical** |
| Op set `X/CX/CCX/SWAP/CZ/CCZ/Z/BIT_*/cond` | `OperationType` enum | **Subset, same semantics** |
| Text op-stream DSL `KIND q.. bN rN if bM` | `Op::from_text` grammar | **Identical grammar** |
| Validity = correctness + reversibility + phase-0 + ancilla-0 + fwd∘rev identity | `run_tests` four gates + README contract | **Identical gate set** |
| Anti-overfit via hash-seeded inputs (stretch band) | Fiat-Shamir SHAKE256 over op stream | **Same mechanism** |
| Register I/O contract (`REGISTER`/`APPEND_TO_REGISTER`, fixed widths) | 4×256 reg layout check in `eval_circuit main` | **Same mechanism, small widths** |
| Reference circuit to beat; strict-improvement bonus | leaderboard "strict < best" (ties rejected) | **Same win condition** |
| Cost centers practiced: modular add/sub, multiply, **modular inverse / cond-sub + swap chains** | challenge cost centers: field mul, **GCD/safegcd inverse (~25% T), cswap, apply mod add/sub (29% T)** | **Same primitive families, scaled down** |
| Width-vs-Toffoli tradeoff (out-of-place ancilla raises high-water width) | self-hosted-square saves T but raises peak (memory frontier) | **Same tension** |

**Conclusion.** The proxy is the secp256k1 challenge with `n=256` replaced by `n≤12` and the
field/curve reference replaced by tiny modular tables — but the op model, the exact cost
function, the four hard validity gates, the emitted text op-stream, the register I/O contract,
the strict-improvement win condition, and the *families of arithmetic primitives that dominate
the real circuit's cost* are all preserved. A policy that learns to emit valid, Toffoli-and-width
minimal reversible op streams here has learned precisely the skill the ECDSA-fail leaderboard
rewards: **verifier-guided, cost-minimizing optimization of a reversible op-stream under hard
correctness + reversibility + phase constraints.** Strong proxy performance is therefore direct
evidence of transfer to the real challenge (validated end-to-end by the T-CFG headline eval in
`PLAN.md`, which runs the model's moves through the *real* harness).

---

## Open implementation notes (for the build phase, not part of this design)

- Reuse `circuit.rs` + `sim.rs` unmodified as a path dependency to guarantee semantic parity;
  add only a thin `proxy_verify(op_text, task_spec) -> Report` wrapper + a task generator +
  reference-circuit library.
- Reference circuits: textbook reversible constructions (ripple/QFT-free modular adder, Bennett
  multiply, conditional-subtract reduction) — kept deliberately un-optimized so there is real
  headroom for the policy to beat them.
- Keep `W ≤ 12` for the exhaustively-verified core; gate the sampled bands behind a flag.
- Emit dense structured feedback (`reason`, witness input, `frac_correct`, `frac_ancilla_clean`)
  for both RL reward shaping and SFT trace construction.
</content>
</invoke>
