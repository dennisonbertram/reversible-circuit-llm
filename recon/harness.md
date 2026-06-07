# ECDSA-fail challenge harness recon (sim + circuit IR + scoring)

Repo: `/Users/dennison/develop/quantum-project/ecdsafail-challenge`
Scope: reversible secp256k1 point-add circuit. Editable path: `src/point_add` (contestant code). Trusted scoring never imports contestant code.

## 1. Exact build + score commands

- **Build** (from `benchmark.sh:108`, reproduced from `setup.sh:212`):
  ```
  RUSTFLAGS="-C linker=$CC" cargo build --release --locked --offline --bin build_circuit --bin eval_circuit
  ```
  Toolchain pinned by `rust-toolchain` → channel `1.93.0`. Release profile: opt-level 3, thin LTO, codegen-units 1. `$CC` = first of `gcc`/`cc`/`clang`. Deps: `alloy-primitives 1.5`, `ruint 1.12`, `sha3 0.10`.
- **Score** is two stages (`benchmark.json` benchmarkCommand = `./benchmark.sh`; scorePath = `score.json`):
  1. `build_circuit` (UNTRUSTED): runs `point_add::build()`, serializes ops to `ops.bin`. Sandboxed (bubblewrap on Linux / sandbox-exec on macOS / unconfined dev fallback), run in scratch cwd, killed by process-group reap.
  2. `eval_circuit "$@"` (TRUSTED): re-reads `ops.bin`, re-simulates, validates, writes `score.json` + appends `results.tsv`. Args forwarded (`--note ...`).
- Do **not** run `./benchmark.sh` directly (a separate probe times it). To score an existing `ops.bin` cheaply, run only the trusted binary: `./target/release/eval_circuit` (cwd = repo root; it reads `./ops.bin`, writes score.json/results.tsv via `CARGO_MANIFEST_DIR`).
- Local fast pre-checks (early-exit, not authoritative): `target/release/quick_check` and `quick_check2` print `PASS <toffoli> <qubits> <score>` / `REJECT ...`. quick_check2 defers the ~18k scalar-muls per batch (lazy), so a dirty reroll bails fastest. Both mirror eval's Fiat-Shamir domain + input derivation exactly.

Current score.json: `score 1940606153 = toffoli 1484779 × qubits 1307`. (avg_tof rounded; see §3.)

## 2. Op / gate model (circuit IR + simulator)

### IR — `src/circuit.rs`
- `OperationType` (u8 enum, discriminants 0..=17): `Neg=0`(global phase flip), `Register=1`, `AppendToRegister=2`, `BitInvert=3`, `BitStore0=4`, `BitStore1=5`, `X=6`, `Z=7`, `CX=8`(CNOT), `CZ=9`, `Swap=10`, `R=11`(reset = HMR with ignored result), `Hmr=12`(X-basis measure + demolition to |0⟩), `CCX=13`(Toffoli), `CCZ=14`, `PushCondition=15`, `PopCondition=16`, `DebugPrint=17`.
- `Op { kind, q_control2, q_control1, q_target: QubitId, c_target, c_condition: BitId, r_target: RegisterId }`. Sentinels `NO_QUBIT/NO_BIT/NO_REG = u64::MAX`. Newtypes `QubitId(u64)`, `BitId(u64)`, `RegisterId(u64)`.
- `Op::validate()` (`circuit.rs:128`): panics on operand **aliasing** (target==control, control1==control2) and per-kind field-shape violations (which fields are BANNED/ALLOWED/REQUIRED per kind). Each kind's required/banned operand set is enumerated in the `match` at `circuit.rs:151-195`. eval_circuit wraps this in `catch_unwind` → forged op → clean rejection, not crash.
- `Op::from_text` parses a text DSL (`CCX q.. q.. q.. [if b..]`, `REGISTER r..`, `APPEND_TO_REGISTER q../b.. r..`, etc.) — handy for a tiny harness writing circuits by hand.
- `Circuit::from_text` / `from_kmx(path)` parse a whole `.kmx` text file into `Circuit { num_qubits, num_bits, num_registers, operations: Vec<Op>, registers: Vec<Vec<QubitOrBit>> }`.
- `analyze_ops(iter)` → `(num_qubits, num_bits, num_registers, registers)`. **peak qubits = max qubit index referenced + 1** (max over q_control2/q_control1/q_target). Registers are built from `AppendToRegister` ops in little-endian append order. This is the function eval uses to size the sim and to define the score's "qubits".

### Simulator — `src/sim.rs`
- `Simulator<'a, R: XofReader> { phase: u64, qubits: Vec<u64>, bits: Vec<u64>, num_qubits, num_bits, xof: &mut R, stats: SimStats }`. **64 shots are packed into the bit-lanes of each `u64`** — one qubit/bit per `Vec` slot, bit `s` of the u64 = shot `s`. So state is classical/stabilizer-free: only X/CX/CCX/Swap (bit permutations) + phase (Z/CZ/CCZ/Neg flip a per-shot phase bit) + reset/measure. This is a reversible-classical (Toffoli-network) simulator, not an amplitude simulator — it is exact and O(ops) per 64-shot batch.
- `new(num_qubits, num_bits, &mut xof)`, `clear_for_shot()` (zero qubits/bits/phase), `qubit(id)/qubit_mut(id)/bit(id)/bit_mut(id)`.
- `apply_iter(ops)` is the core (`sim.rs:71`). Per op it computes `cond` = base condition stack AND (optional) `c_condition` bit mask, then:
  - **CCX**: `q_target ^= cond & q_c1 & q_c2`. **CX**: `q_target ^= cond & q_c1`. **X**: `q_target ^= cond`. **Swap**: conditional 3-XOR swap.
  - **CCZ/CZ/Z/Neg**: XOR into `self.phase` (the reversibility/identity check; see §below).
  - **Hmr** (`sim.rs:140`): reads 8 RNG bytes from xof, writes random bit into `c_target` on live shots, `phase ^= q_target & rng & cond`, then zeroes `q_target` on live shots. **R** (`sim.rs:149`): same minus the bit write — `phase ^= q_target & rng & cond; q_target &= !cond`. This is how a **dirty free is punished**: if a qubit is not |0⟩ when reset, the random `rng_val` injects garbage into the global phase, which the phase check then catches with overwhelming probability.
  - **BitInvert/BitStore0/BitStore1**: classical bit writes. **PushCondition/PopCondition**: maintain `condition_stack` (nested classical control). **AppendToRegister/Register/DebugPrint**: no state effect.
- `set_register(reg, val: U256, shot_idx)` / `get_register(reg, shot_idx) -> U256`: little-endian load/read of a register's qubits+bits for one shot. Used to inject inputs and read outputs.
- **Reversibility / forward-reverse-identity / phase enforcement** is done by `eval_circuit::run_tests` (`eval_circuit.rs:214`), NOT inside sim:
  - **Correctness**: after `apply_iter`, regs 0/1 (output gx,gy) must equal the reference `curve.add` result per shot.
  - **Phase**: `sim.phase & cond_mask` must be 0 on all live shots (a clean reversible circuit leaves zero relative phase). Nonzero ⇒ "PHASE GARBAGE".
  - **Ancilla cleanup ("forward = identity on ancillas")**: after zeroing the 4 register qubit-lanes, **every other qubit must be |0⟩ on every live shot** (`eval_circuit.rs:304-331`). Nonzero ⇒ "ANCILLA GARBAGE". This is what forces the circuit to uncompute all scratch.
- `SimStats { clifford_gates, toffoli_gates }`: counted in `apply_iter` weighted by `cond.count_ones()` (number of live shots the op actually executed on). **CCX+CCZ → toffoli_gates**; CX/CZ/Swap/R/Hmr → clifford_gates; **X/Z are NOT counted** (trackable in classical control). Toffoli is the optimization axis.

## 3. How eval computes the score

`write_score(avg_tof, qubits)` at `eval_circuit.rs:410`:
```
toffoli = avg_tof.round() as u64;          // avg_tof = sim.stats.toffoli_gates / n_nondegenerate_shots
score   = toffoli.saturating_mul(qubits);  // qubits = total_qubits from analyze_ops (peak index+1)
```
`avg_tof = sim.stats.toffoli_gates as f64 / n` where `n` = number of NON-degenerate test shots (after skipping equal-x / point-at-infinity cases). Both quick_check tools compute the same `toffoli.saturating_mul(total_qubits)`. So **score = round(total executed Toffoli / live shots) × peak qubits**. Lower is better (`direction: "-"`).

Register-shape gate (eval, `eval_circuit.rs:449-486`): exactly **4 registers, each width 256**; reg0/reg1 = qubits (the quantum point being added, gx/gy outputs), reg2/reg3 = classical bits (the classical point offset). A circuit violating this fails before scoring.

## 4. Parameterizability (for a faster proxy)

- **No env vars / config** gate shot count or width in the trusted path. `grep` for `env::var` in sim/circuit/eval finds only `CARGO_MANIFEST_DIR` (output paths). The only `env::var` in the whole tree is `TRACE_PEAK` in the contestant builder `point_add/mod.rs` (peak-logging, irrelevant to sim/score).
- **Shot count**: `const NUM_TESTS: usize = 9024;` (`eval_circuit.rs:38`, and identically in both quick_check binaries). Hard-coded const — reducible only by code edit. `const BATCH: usize = 64;` (`eval_circuit.rs:258`) is fixed by the 64-shot u64 packing and should NOT change. To make a fast proxy that still matches eval's Fiat-Shamir RNG positioning, you must consume the xof identically (read 2×32 bytes per test for all 9024) — quick_check2 shows the minimal lazy pattern. For a *non-FS-matching* fast proxy (small arithmetic, fixed inputs) you ignore Fiat-Shamir entirely (see §5).
- **Register width 256**: enforced by eval's reg-shape check (`r.len() != 256`) and by `secp256k1()` being 256-bit. Reducible only for a *standalone* proxy that bypasses eval (use a smaller curve / smaller registers directly against `Simulator`). The `Simulator` itself is width-agnostic — `num_qubits`/`num_bits` are runtime args to `Simulator::new`; nothing in sim.rs hardcodes 256.
- Net: for the **official** score, nothing is tunable without editing consts. For a **proxy**, the sim+circuit crate is fully width/shot-agnostic and can be driven at any size (see §5).

## 5. CRITICAL — sim.rs + circuit.rs as a standalone fast verifier for SMALL reversible circuits

**Yes — clean and directly reusable.** `circuit.rs` and `sim.rs` have no secp256k1 / 256-bit / 9024-shot assumptions; all of that lives in `eval_circuit.rs` and `weierstrass_elliptic_curve.rs`. The reusable surface:

- **Op model**: build `Vec<Op>` either programmatically (`Op { kind: OperationType::CCX, q_control1: QubitId(a), q_control2: QubitId(b), q_target: QubitId(c), ..Op::empty() }`) or from a text line via `Op::from_text("CCX q0 q1 q2")` / a whole program via `Circuit::from_text(text)`. Call `op.validate()` to reject aliasing.
- **Sizing**: `analyze_ops(ops.iter()) -> (num_qubits, num_bits, num_registers, registers)` gives peak qubits + the register layout (from `AppendToRegister` ops) for free.
- **Simulate**: you need any `sha3::digest::XofReader` for the `&mut xof` (only consumed by R/Hmr). Minimal:
  ```rust
  use sha3::{Shake256, digest::{ExtendableOutput, Update}};
  let mut xof = { let mut h = Shake256::default(); h.update(b"seed"); h.finalize_xof() };
  let mut sim = Simulator::new(num_qubits as usize, num_bits as usize, &mut xof);
  sim.clear_for_shot();
  sim.set_register(&regs[i], input_val_u256, shot /*0..64*/);   // load inputs, one u256 per shot lane
  sim.apply_iter(ops.iter());                                    // run
  let out = sim.get_register(&regs[j], shot);                    // read output
  let dirty_phase = sim.phase;                                   // must be 0 for a clean reversible circuit
  let toffolis = sim.stats.toffoli_gates;                        // optimization metric
  ```
- **Verifier recipe for a small reversible arithmetic circuit** (e.g. an n-bit adder, n « 256):
  1. Define input/output registers as `Vec<QubitOrBit>` (or via `AppendToRegister` ops + `analyze_ops`).
  2. For each of up to 64 test cases, `set_register` inputs into shot-lane `s`.
  3. `sim.apply_iter(ops)`; per lane check `get_register(out) == expected`, check `sim.phase` bit `s` == 0 (reversible/no-phase), and check all non-output qubits == 0 (ancilla clean) — same three checks eval does, just at small width and with your own fixed inputs (no Fiat-Shamir needed).
  4. Score proxy = `sim.stats.toffoli_gates / n_shots × (peak qubits from analyze_ops)`.
- This is exactly what the proxy RL env should wrap: it is exact (bit-packed classical sim), O(ops·batches), no amplitude blowup, and reuses the *trusted* sim semantics (same Toffoli accounting, same reset/phase punishment) so a circuit that passes the proxy at small width uses the identical primitives that eval scores at 256-bit. The contestant builder `B` (alloc_qubit/free/emit_inverse, in `point_add/`) is a separate convenience layer and is NOT needed — drive `Op`/`Simulator` directly.

### Build the reusable crate as a dependency
Add a tiny bin under the existing crate (it already exposes `pub mod circuit; pub mod sim;` in `lib.rs`) — drop a `src/bin/proxy_verify.rs` and `cargo build --release --bin proxy_verify`. Or vendor `circuit.rs` + `sim.rs` (only crate dep they need: `ruint` for `U256` in set/get_register; `sha3` only for the `XofReader` trait bound — substitutable). No other repo files are required for small-width verification.

## Key files
- `src/circuit.rs` — Op/OperationType/Circuit IR, validate(), analyze_ops (peak qubits + register layout).
- `src/sim.rs` — `Simulator` (64-shot bit-packed reversible/classical sim), stats counters, set/get_register.
- `src/bin/eval_circuit.rs` — trusted scoring: load_ops, Fiat-Shamir (`fiat_shamir_seed`), run_tests (correctness/phase/ancilla checks), write_score (score = round(avg_tof)×qubits). NUM_TESTS=9024, BATCH=64.
- `src/bin/build_circuit.rs` — untrusted: `point_add::build()` → `ops.bin` (56-byte LE per op, magic `QECCOPS1`).
- `src/bin/quick_check.rs`, `quick_check2.rs` — local early-exit validators (quick_check2 = lazy scalar-mul).
- `src/weierstrass_elliptic_curve.rs` — reference secp256k1 add/mul (the correctness oracle).
- `src/point_add/` — contestant code (editable). `mod.rs::build()`, builder `B` (alloc_qubit/free/emit_inverse), arith/ submodules. Not needed for the proxy verifier.
- `benchmark.sh` / `setup.sh` / `Cargo.toml` / `rust-toolchain` (1.93.0) / `benchmark.json`.
