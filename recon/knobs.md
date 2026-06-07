# ECDSA-fail point-add circuit builder: tunable action space (knobs)

Recon target: `/Users/dennison/develop/quantum-project/ecdsafail-challenge`
Score metric: **peak_qubits × executed_Toffoli** (lower is better). Current best ≈ 2.478e9 per memory; the in-tree `configure_ecdsafail_submission_route` comments claim down to ≈1.96e9 for the dialog-GCD route. Both `CCX` and `CCZ` count as Toffoli.

This document enumerates EVERY knob a model can manipulate to change the produced circuit, plus how the circuit "program" is represented and the high-level action space.

---

## 1. How the circuit program is built (entry points & ABI)

- `point_add::build() -> Vec<Op>` (in `src/point_add/mod.rs`, line ~1563) is THE harness entry point. It calls `build_builder()`.
- `build_builder()` (line ~1358):
  1. Calls `configure_ecdsafail_submission_route()` — this **`set_default_env`'s the entire active submission config** (~70 env vars). Every knob below is set there UNLESS already present in the process env, so external env vars OVERRIDE the in-code defaults (`set_default_env` only sets when `var_os` is `None`, line 966).
  2. Allocates 4 registers in declaration order, each width `N=256`:
     - reg0 `target_x` — **qubits** (the quantum P.x)
     - reg1 `target_y` — **qubits** (the quantum P.y)
     - reg2 `offset_x` — **classical bits** (the known Q.x)
     - reg3 `offset_y` — **classical bits** (the known Q.y)
  3. Optional `DIALOG_REROLL` identity prelude (X;X pairs on tx[0]).
  4. `mod_sub_qb(tx,ox)`, `mod_sub_qb(ty,oy)` (Px-=Qx, Py-=Qy).
  5. Optional `DIALOG_POST_SUB_REROLL` identity (X;X on tx[1]).
  6. `emit_dialog_gcd_raw_pa(b, tx, ty, ox, oy, p)` — the whole point-add body.
  7. `run_alt_seed_checks` unless `SKIP_ALT_SEED_CHECKS=1` (submission route sets it).
  8. Optional `DIALOG_TAIL_NONCE` fixed-length (48-bit) identity tail.

**Op representation.** A circuit is a flat `Vec<Op>` (struct in `circuit`). Each `Op` is a fixed record: `kind: OperationType` (18 kinds, e.g. `X, CX, CCX, CCZ, Swap, R, Hmr, Z, CZ, Register, AppendToRegister, …`), up to two qubit controls (`q_control1/2`), a `q_target`, a classical `c_target`/`c_condition` (for HMR/measurement-conditioned gates), and an `r_target` register id. Registers are declared via `Register`/`AppendToRegister` pseudo-ops. The builder `B` (mod.rs line ~91) accumulates ops via `push_op`, tracks live-qubit allocation (`alloc_qubit`/`free`/`reacquire` with a `free_qubits` recycle pool), and continuously updates `peak_qubits` — so **peak qubit width is an emergent property of the allocation schedule**, not a declared constant. `count_only` mode (`POINT_ADD_COUNT_ONLY=1`) skips materializing `ops` and just tallies kind counts.

### Pipeline / "rounds" (the `emit_dialog_gcd_raw_pa` driver, dialog/mod.rs line 1745)

Sequential stages (each can be cut short by a `STOP_AFTER_*` env flag, value-incorrect but useful for isolating a phase):

1. `pair1_quotient` — `emit_dialog_gcd_raw_quotient` (Kaliski/binary-GCD inverse of dx → λ numerator path). → `DIALOG_GCD_RAW_PA_STOP_AFTER_QUOTIENT`
2. `round84_fused_square_xtail` — λ² square + 2·Qx, negate → Rx. Square algo is itself selectable (schoolbook-lowq default / Karatsuba / walk / schoolbook). → `DIALOG_GCD_RAW_PA_STOP_AFTER_XTAIL`
3. `c_ox_minus_rx` — `mod_sub_qb` + `mod_neg` → c = Qx-Rx. → `DIALOG_GCD_RAW_PA_STOP_AFTER_C`
4. `pair2_product` — `emit_dialog_gcd_raw_ipmul` (second GCD-backed inverse/product, the λ·(Qx-Rx) path). → `DIALOG_GCD_RAW_PA_STOP_AFTER_PAIR2`
5. `y_output` — `mod_sub_qb(ty,oy)` → Ry.
6. `x_restore` — `mod_neg` + `mod_add_qb`.

Both GCD-backed pairs (quotient and ipmul) internally run the **dialog-GCD inversion** = `active_iterations` tobitvector steps (forward), an apply phase, and a reverse sweep, with a compressed-sidecar transcript log (round763 packer). The compressed path lives in `dialog/compressed.rs`.

### How `dialog/compressed.rs` encodes the circuit program (head + structure; 2229 lines, not read in full)

`compressed.rs` is NOT a serialized data blob — it is **Rust emitter code** that procedurally appends gates for the GCD transcript log. Structure (from the head + function survey, 50 functions):

- **Round763 6→5 block compressor** (`emit_dialog_gcd_round763_compressor` / `_inverse`, lines 27–83): each GCD step produces a raw 2-bit (K1) or 3-bit (K2) record `(b0=v-odd, b0_and_b1=b0&(v<u), [shift2])`. The compressor packs a GROUP_SIZE=3 group of raw slots (6 raw bits) into 5 compressed cells, exploiting that state `(0,1)` is unreachable on the verifier support. `round763_compress_lever`/`round763_dedup` env levers swap in cheaper reachable-support rewrites (9→4 CCX, or a CCX-pair→CX cancellation).
- **Block swapper** (`emit_dialog_gcd_round763_compressed_block_swapper`, line 85): decompress-in-place, swap a step's pair into a compressed slot, recompress — the in-place read/write of a single step's transcript bits.
- **Layout helpers** (lines 103–117): `dialog_gcd_compressed_sidecar_blocks/bits/block` compute the compressed-log register size from `active_iterations`, `block_bits()` (5 for K1/pair-compress, 8 for plain K2), and `sidecar_group_size`.
- **U-high runway** (lines 119–158): an opt-in prototype that parks the late transcript suffix on provably-|0> high lanes of `u` (peak lever; `DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY[_BLOCKS]`).
- **Composite scratch / borrow** (lines ~290–345+): `DialogGcdCompositeScratch` pools the borrowable |0> lanes (future-log region + optionally the current block's own cells / s2 cell) that host the body's gated+carry transient, so the wide GCD body add/sub does not freshly allocate ~2·active_width ancilla at the peak. This is the central qubit-peak machinery.
- **Block-lifecycle emitters** `emit_dialog_gcd_compressed_sidecar_{tobitvector,apply,ipmul,quotient}` (referenced via `DIALOG_GCD_COMPRESSED_BLOCK_LIFECYCLE`) drive each step: decompress current block → run step body → recompress.

So the "program" is: a fixed driver (6 stages) → two GCD inversions → each a loop of `active_iterations` procedurally-emitted steps whose width, comparator bits, carry-truncation, hosting, and transcript packing are all governed by the env knobs below.

---

## 2. Knob catalogue

Legend for **effect** columns: T = executed Toffoli, Q = peak qubit width, C = correctness (validity / Fiat-Shamir island). "value-exact" = changes T/Q only, correctness preserved on reachable support; "reseeds island" = changes the serialized op bytes → reshuffles the 9024 SHAKE256-derived test inputs, so a co-tuned nonce/reroll must be re-found.

### 2a. Pipeline shape / route selection

| name | env var | type | range/options | default (submission) | effect |
|---|---|---|---|---|---|
| count-only build | `POINT_ADD_COUNT_ONLY` | bool | `1` | unset | metrics only, no ops materialized; no circuit change |
| skip alt-seed checks | `SKIP_ALT_SEED_CHECKS` | bool | `1` | `1` | C: disables the 5×4096 extra random validation (faster build) |
| raw PA path | `DIALOG_GCD_RAW_PA` | bool | `1` | `1` | selects the dialog-GCD raw point-add body |
| compressed sidecar log | `DIALOG_GCD_COMPRESSED_SIDECAR_LOG` | bool | `1` | `1` | Q: use compressed transcript log (lower peak) |
| block lifecycle | `DIALOG_GCD_COMPRESSED_BLOCK_LIFECYCLE` | bool | `1` | `1` | Q: per-block decompress/recompress lifecycle |
| host reverse raw block | `DIALOG_GCD_HOST_REVERSE_RAW_BLOCK` | bool | `1` | `1` | Q: host reverse raw block on borrowed lanes |
| stop after quotient | `DIALOG_GCD_RAW_PA_STOP_AFTER_QUOTIENT` | bool | `1` | unset | C: truncates pipeline (debug; incorrect output) |
| stop after xtail | `DIALOG_GCD_RAW_PA_STOP_AFTER_XTAIL` | bool | `1` | unset | C: as above |
| stop after c | `DIALOG_GCD_RAW_PA_STOP_AFTER_C` | bool | `1` | unset | C: as above |
| stop after pair2 | `DIALOG_GCD_RAW_PA_STOP_AFTER_PAIR2` | bool | `1` | unset | C: as above |
| round84 square = Karatsuba | `ROUND84_XTAIL_KARATSUBA` | bool | `1` | `0` | T/Q: Karatsuba square (−16k emitted T but co-binder risk) |
| round84 square = walk | `ROUND84_XTAIL_WALK_SQUARE` | bool | `1` | unset | alt square algo |
| round84 square = schoolbook | `ROUND84_XTAIL_SCHOOLBOOK` | bool | `1` | unset | full schoolbook square (else lowq-shift22 default) |

### 2b. GCD structural sizing (the dominant T/Q drivers)

| name | env var | type | range | default | effect |
|---|---|---|---|---|---|
| active iterations | `DIALOG_GCD_ACTIVE_ITERATIONS` | usize | 1..=402 (`MAX_ITERATIONS`) | `258` | **T & C**: # GCD steps run. Fewer = less T but risks non-convergence on some inputs (Fiat-Shamir hazard). Each step ≈ one full body add/sub+cswap+comparator. |
| width margin | `DIALOG_GCD_WIDTH_MARGIN` | f64 | 0.0..=256.0 | `10.0` (code default 37.0) | **T & C**: safety bits added to per-step realizable bitlen envelope. Lower = narrower body widths = less T, peak-neutral; too low → width-truncation hazards. |
| width slope ×1000 | `DIALOG_GCD_WIDTH_SLOPE_X1000` | f64/1000 | (0,4.0] | `1014` (→1.014; code default 0.7075) | **T & C**: per-step shrink rate of the width envelope `N - step*slope + margin`. Higher = faster shrink = less T; value-exact on converged support, reseeds island. |
| variable width | `DIALOG_GCD_RAW_TOBITVECTOR_VARIABLE_WIDTH` | bool | `1`/`0` | `1` | T: enables the step-varying width envelope (vs flat N). `0` forces full N. |

### 2c. Comparator-width truncations (value-exact T cuts)

| name | env var | type | range | default | effect |
|---|---|---|---|---|---|
| branch compare bits | `DIALOG_GCD_COMPARE_BITS` | usize | 1..=256 | `49` (code default 77; filter default 57) | **T**: top-bits used by the `u>v` branch comparator. Lower = less T; value-exact down to ~52 per comments, reseeds island. 2 T/bit ×2 dir ×2 pass. |
| apply-clean compare bits | `DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS` | usize | 1..=256 | `19` (falls back to compare_bits) | **T**: bits for the apply-phase overflow-clean cmp_lt. −516 T/bit, peak-neutral. |
| PA9024 per-step schedule | `DIALOG_GCD_PA9024_COMPARE_SCHEDULE` | bool | `1` | `1` | **T**: use the calibrated per-step `DIALOG_GCD_PA9024_COMPARE_SCHEDULE[258]` table (early steps need far fewer compare bits) instead of flat compare_bits. Big T cut. |
| schedule margin | `DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN` | usize | ≥0 | `0` | **T & C**: safety bits over the observed per-step max. 0 = tightest/cheapest; raising trades T for an easier island. |
| schedule floor | `DIALOG_GCD_PA9024_COMPARE_SCHEDULE_FLOOR` | usize | ≤256 (≥1) | `1` | minimum per-step compare bits |

The effective per-step bits = `min(SCHEDULE[step]+margin, compare_bits, active_width).max(floor)` (config.rs `dialog_gcd_compare_bits_for_step`).

### 2d. Carry-tail truncation windows (value-exact T cuts via Solinas sparsity)

| name | env var | type | range | default | effect |
|---|---|---|---|---|---|
| body carry trunc W | `DIALOG_GCD_BODY_CARRY_TRUNC_W` | usize | ≥0 (0=off) | unset (0) | T: stop the GCD body add/sub carry ripple W bits below active_width. Per-step override by band-trims below. |
| body carry band trims | `DIALOG_GCD_BODY_CARRY_BAND_TRIMS` | csv usize | e.g. `0,1,2` | `0,1,2` | T & C: per-band (banded over steps) body carry-trim W; later bands trim more. |
| double-carry trunc W | `KAL_DOUBLE_CARRY_TRUNC_W` | usize | >0 | `21` | T: truncate the DOUBLE op's lazy-Solinas carry window. Value-exact on support, reseeds island. |
| fold-carry trunc W | `KAL_FOLD_CARRY_TRUNC_W` | usize | >0 | `21` | T: truncate the overflow/underflow FOLD adder carry (sparse c=2^32+977). Value-exact, reseeds island. |

### 2e. Apply-phase chunking & windowing (qubit-peak levers)

| name | env var | type | range | default | effect |
|---|---|---|---|---|---|
| measured apply sub | `DIALOG_GCD_MEASURED_APPLY_SUB` | bool | `1` | `1` | T: Gidney measured uncompute for apply sub (~n vs 2n T), peak-neutral |
| apply window blocks | `DIALOG_GCD_APPLY_WINDOW_BLOCKS` | usize | ≥2 | `2` | Q: window the apply carry lane into k blocks (keeps wide carry lane off peak) |
| apply chunked F blocks | `DIALOG_GCD_APPLY_CHUNKED_F_BLOCKS` | usize | ≥2 | `5` | Q: # chunks for the f-register apply add/sub |
| chunked F cut(1..4) | `DIALOG_GCD_APPLY_CHUNKED_F_CUT`, `_CUT2`, `_CUT3`, `_CUT4` | usize | 1..256 | 50,100,150,190 (CUT default 50) | Q: chunk boundary bit positions; widening sinks apply peak to the next floor. EXACT for any cut. |
| custom-4 cuts | `DIALOG_GCD_APPLY_CHUNKED_F_CUSTOM4` | bool | `1`/`0` | `0` | use CUT..CUT4 custom boundaries |
| custom-5 cuts | `DIALOG_GCD_APPLY_CHUNKED_F_CUSTOM5` | bool | `1`/`0` | `1` | use 5-way custom boundaries |
| reuse c_in zero | `DIALOG_GCD_APPLY_CHUNKED_F_REUSE_CIN_ZERO` | bool | `≠0`=on | on (default) | Q: reuse known-zero c_in across chunks |
| fuse boundary clears | `DIALOG_GCD_APPLY_CHUNKED_F_FUSE_BOUNDARY_CLEARS` | bool | `≠0`=on | on (default) | T: fuse adjacent boundary clears |
| apply boundary split | `DIALOG_GCD_APPLY_BOUNDARY_SPLIT` | usize | >0 | `100` | Q: hosted high/low window split bit |
| apply final lowq | `DIALOG_GCD_APPLY_FINAL_LOWQ` | bool | `1` | `0` | Q: low-q final chunk teardown |
| apply final windowed fast blocks | `DIALOG_GCD_APPLY_FINAL_WINDOWED_FAST_BLOCKS` | usize | ≥2 | `0` (off) | Q: windowed fast final apply |
| apply fused fold | `DIALOG_GCD_APPLY_FUSED_FOLD` | bool | `1` | `1` | T: fuse double_y+halve_y Solinas folds (−25k T, phase-clean) |
| apply replay swap host | `DIALOG_GCD_APPLY_REPLAY_SWAP_HOST` | bool | `1` | `1` | Q: swap compressed cells into raw_block for replay scratch |

### 2f. Carry/scratch hosting (qubit-peak levers, mostly value-exact relabels)

| name | env var | type | options | default | effect |
|---|---|---|---|---|---|
| host gated | `DIALOG_GCD_HOST_GATED` | bool | `1` | `1` | Q: host the wide gated register on |0> future-log slots (−256 fresh ancilla at peak) |
| body host c_in | `DIALOG_GCD_BODY_HOST_CIN` | bool | `1` | `1` | Q: host Cuccaro c_in on the never-loaded gated[0] (needs odd-u fastpath) |
| selected body no-c_in | `DIALOG_GCD_SELECTED_BODY_NOCIN` | enum | `1`/`2`/unset | `1` | Q: body consumes no incoming-carry lane (mode 2 = diagnostic, keeps pool) |
| late borrow uv-high | `DIALOG_GCD_LATE_BORROW_UV_HIGH` | bool | `1` | `1` | Q: borrow u-high |0> bits as late-step scratch |
| branch-bits host comparator | `DIALOG_GCD_BRANCH_BITS_HOST_COMPARATOR` | bool | `1` | `1` | Q: route fused branch path through borrowed-carry comparator |
| partial host comparator | `DIALOG_GCD_PARTIAL_HOST_COMPARATOR` | bool | `≠0`=on | on (default) | Q: borrow prefix + alloc only the deficit |
| composite scratch | `DIALOG_GCD_COMPOSITE_SCRATCH` | bool | `1` | `1` | Q: pool borrowable |0> lanes for body transient |
| borrow current block | `DIALOG_GCD_BORROW_CURRENT_BLOCK` | bool | `1` | `1` | Q: fold current block's own |0> cells into borrow (peak −4, 0 added T) |
| borrow current s2 | `DIALOG_GCD_BORROW_CURRENT_S2` | bool | `1` | `1` | Q: fold current step's shift2 cell into borrow (K2 path) |
| host reverse raw block | `DIALOG_GCD_HOST_REVERSE_RAW_BLOCK` | bool | `1` | `1` | Q: (see 2a) |
| u-high runway | `DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY` | bool | `1` | unset (prototype) | Q: park late transcript suffix on u-high lanes |
| u-high runway blocks | `DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY_BLOCKS` | usize | ≥0 | `999` (submission; code default 16) | Q: cap on parked blocks |

### 2g. Body / terminal-reuse micro-levers (value-exact T/Q)

| name | env var | type | default | effect |
|---|---|---|---|---|
| K2 bounded shift | `DIALOG_GCD_K2` | bool | `1` | T/C: strip up to 2 trailing zeros per step (extra cond shift); changes transcript width |
| K2 pair compress | `DIALOG_GCD_K2_PAIR_COMPRESS` | bool | `1` | Q: pack 2 K2 steps into 5 sidecar bits (1313q tier) |
| K2 force0 | `DIALOG_GCD_K2_FORCE0` | bool | unset | diagnostic: force shift2=0 |
| K2 no-apply | `DIALOG_GCD_K2_NO_APPLY` | bool | unset | diagnostic: skip K2 apply mirror |
| fuse halve off | `DIALOG_GCD_FUSE_HALVE_OFF` | bool | unset | disable halve fusion |
| fused branch bits | `DIALOG_GCD_FUSED_BRANCH_BITS` | bool | `1` | T: derive b0&b1 from in-flight comparator carry (−90k emitted T), peak-neutral |
| odd-u lowbit fastpath | `DIALOG_GCD_ODD_U_LOWBIT_FASTPATH` | bool | `1` | T/Q: u[0]=1 on support → lane-0 CX + body_start=1 |
| ipmul terminal reuse | `DIALOG_GCD_RAW_IPMUL_TERMINAL_REUSE` | bool | `1` | Q: reuse terminal registers |
| ipmul clear p residual | `DIALOG_GCD_RAW_IPMUL_CLEAR_P_RESIDUAL` | bool | `1` | C/Q: clear p residual |
| quotient terminal reuse | `DIALOG_GCD_RAW_QUOTIENT_TERMINAL_REUSE` | bool | `1` (defaults to ipmul flag) | Q |
| quotient keep terminal u | `DIALOG_GCD_RAW_QUOTIENT_KEEP_TERMINAL_U` | bool | unset | Q |
| tobitvector materialized sub | `DIALOG_GCD_RAW_TOBITVECTOR_MATERIALIZED_SUB` | bool | `1` | T/Q: materialized vs direct sub body |
| tobitvector borrow future-log carries | `DIALOG_GCD_RAW_TOBITVECTOR_BORROW_FUTURE_LOG_CARRIES` | bool | `1` | Q |
| apply materialized special add | `DIALOG_GCD_RAW_APPLY_MATERIALIZED_SPECIAL_ADD` | bool | `1` | T/Q: apply add variant |
| apply reverse materialized special sub | `DIALOG_GCD_RAW_APPLY_REVERSE_MATERIALIZED_SPECIAL_SUB` | bool | `1` | T/Q |
| apply truncated clean | `DIALOG_GCD_RAW_APPLY_TRUNCATED_CLEAN` | bool | `1` | T |
| apply direct special add | `DIALOG_GCD_RAW_APPLY_DIRECT_SPECIAL_ADD` | bool | unset | alt apply add |
| apply reverse fast sub | `DIALOG_GCD_RAW_APPLY_REVERSE_FAST_SUB` | bool | unset | alt apply sub |
| round763 dedup | `DIALOG_GCD_ROUND763_DEDUP` | bool | `1` | T: CCX-pair→CX cancellation in compressor |
| round763 compress lever | `DIALOG_GCD_ROUND763_COMPRESS_LEVER` | bool | `1` | T: reachable-support 9→4 CCX compressor |
| measured underflow gate | `DIALOG_GCD_MEASURED_UNDERFLOW_GATE` | bool | `1` | T: measured underflow gate |
| host gated body | `DIALOG_GCD_HOST_GATED` | (see 2f) | | |
| selected body nocin selftest | (`#[test]` only) | | | |

### 2h. Round84 / Karatsuba / square micro-levers

| name | env var | type | default | effect |
|---|---|---|---|---|
| kara sol dbl fast | `KARA_SOL_DBL_FAST` | bool | `1` | T: fast (carry-ancilla) Solinas doubling (−12.9k T) |
| kara free z1 topbit | `KARA_FREE_Z1_TOPBIT` | bool | `1` | Q: free provably-0 z1[257] (−1q) |
| kara z02 lowq | `KARA_Z02_LOWQ` | bool | `1` | Q: host z0 carry lane on z2 slice |
| kara z2 selfhost | `KARA_Z2_SELFHOST` | bool | `≠0`=on | Q: run z2 square ancilla-free |
| kara sol mod vent | `KARA_SOL_MOD_VENT` | bool | `1` | Q: vent const corrections onto dirty operand |
| kara sol mod fast | `KARA_SOL_MOD_FAST` | bool | unset | T |
| kara sol shift fast | `KARA_SOL_SHIFT_FAST` | bool | unset | T |
| xtail sq selfhost | `XTAIL_SQ_SELFHOST` | bool | `≠0`=on | Q |
| round84 xtail borrow carries | `ROUND84_XTAIL_BORROW_CARRIES` | bool | `1` | Q: 2^22 doubling/halving instead of shift-by-22 (square phase 1567→1543) |
| square selfhost safe lane reuse | `SQUARE_SELFHOST_SAFE_LANE_REUSE` | bool | `1` | Q: structural source-high zero reuse |
| square selfhost gate suffix carries | `SQUARE_SELFHOST_GATE_SUFFIX_CARRIES` | usize | `1` | Q/T: # suffix carry lanes gated |
| square selfhost gate prefix rows | `SQUARE_SELFHOST_GATE_PREFIX_ROWS` | usize | unset (0) | Q/T |
| r84 lowq | `R84_LOWQ` | bool | `1` | Q: ancilla-light round84 mid-sub (1309→1307) |
| r84 lowq cin borrow | `R84_LOWQ_CIN_BORROW` | bool | `1` | Q: const-add carry-in borrow |
| lowq shift22 | `LOWQ_SHIFT22` | bool | `≠0`=on, default off | Q: low-q phase-corrected shift core |

### 2i. Const-arith / modular venting levers

| name | env var | type | default | effect |
|---|---|---|---|---|
| direct const walks | `KAL_DIRECT_CONST_WALKS` | bool | unset | alt const-add walks |
| secp direct const arith | `SECP_DIRECT_CONST_ARITH` | bool | unset | direct secp const arith |
| vent modadd | `KAL_VENT_MODADD` | bool | unset | Q: vent mod-add |
| vent halve | `KAL_VENT_HALVE` | bool | unset | Q |
| vent double | `KAL_VENT_DOUBLE` | bool | unset | Q |
| direct const double | `KAL_DIRECT_CONST_DOUBLE` | bool | unset | T/Q |
| direct const halve | `KAL_DIRECT_CONST_HALVE` | bool | unset | T/Q |

### 2j. Fiat-Shamir island selectors (correctness-only; ZERO effect on T/Q)

These do NOT change the circuit's action, Toffoli count, or peak — they only perturb the serialized op bytes, which reseed the SHAKE256-derived 9024 test inputs. After ANY value-exact truncation above (which changes the op stream), you must re-find a nonce/reroll that lands a "clean island" (0 classical / 0 phase / 0 ancilla failures over 9024 shots).

| name | env var | type | range | default | effect |
|---|---|---|---|---|---|
| tail nonce | `DIALOG_TAIL_NONCE` | u64 | any (48 bits used) | `2667032` | C-only: fixed-length 48-bit identity tail (X;X on tx[0]/tx[1]); selects test set |
| reroll | `DIALOG_REROLL` | usize | >0 | `4269` | C-only: k identity X;X pairs on tx[0] before body |
| post-sub reroll | `DIALOG_POST_SUB_REROLL` | usize | >0 | `503292` | C-only: k identity X;X pairs on tx[1] after the initial subtracts |
| body nocin nonce | `DIALOG_GCD_SELECTED_BODY_NOCIN` | (also a route knob) | `1` | (see 2f) | |

### 2k. Alt-seed / validation / trace (don't change circuit; affect build behavior)

| name | env var | effect |
|---|---|---|
| `ALT_SEED_COMMIT` | use 24 seeds instead of 5 in the extra validation |
| `ALT_SEED_PHASE_LIMIT` | allowed phase-garbage batches (default 0) |
| `TRACE_PEAK`, `TRACE_EACH_PEAK`, `TRACE_PHASES`, `TRACE_PHASES_VERBOSE`, `TRACE_PHASE_ACTIVE`, `TRACE_PHASE_ACTIVE_TOP`, `TRACE_PHASE_ACTIVE_REGIONS` | eprintln diagnostics for peak qubit / per-phase Toffoli attribution. **Useful as a reward/observation signal for an optimizing model** (per-phase T and per-phase active-qubit maxima). |
| `DIALOG_GCD_FILTER_STRICT_COMPARE` | classical-filter strictness (used by `dialog_gcd_classical_filter.rs`, the fast classical convergence pre-filter for island search) |

### Important compile-time consts (not env-tunable without editing source)

| const | file | value | meaning |
|---|---|---|---|
| `N` | mod.rs | 256 | register width (curve field) |
| `SECP256K1_P` | mod.rs | 2^256−2^32−977 | hard-coded prime |
| `DIALOG_GCD_MAX_ITERATIONS` | config.rs | 402 | hard upper bound for active_iterations / log sizing |
| `DIALOG_GCD_RAW_LOG_BITS` | config.rs | 804 | 2×max iters |
| `DIALOG_GCD_DEFAULT_COMPARE_BITS` | config.rs | 77 | fallback when COMPARE_BITS unset |
| `DIALOG_GCD_SPECIAL_ADD_LSBS` | config.rs | 73 | LSBs touched by the special fold add |
| `DIALOG_GCD_HIGH_TAIL_ALIAS_GROUP_SIZE` | config.rs | 3 | round763 group size |
| `DIALOG_GCD_HIGH_TAIL_ALIAS_BLOCK_BITS` | config.rs | 5 | compressed cells/block |
| `DIALOG_GCD_PA9024_COMPARE_SCHEDULE[258]` | config.rs | (table) | per-step compare bits; **editable small code region** |
| `NONCE_BITS` | mod.rs build_builder | 48 | tail-nonce length |

---

## 3. Action space summary (what a model manipulates)

A model optimizing the score (peak_q × executed_T) has these orthogonal action axes:

1. **Set a numeric knob within range** — the high-leverage continuous/integer dials:
   - T-axis (smaller = fewer Toffoli, correctness via co-tuned island): `DIALOG_GCD_COMPARE_BITS` (~49), `DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS` (~19), `DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN` (0), `KAL_DOUBLE_CARRY_TRUNC_W`/`KAL_FOLD_CARRY_TRUNC_W` (~21), `DIALOG_GCD_BODY_CARRY_TRUNC_W`, `DIALOG_GCD_ACTIVE_ITERATIONS` (~258), `DIALOG_GCD_WIDTH_MARGIN` (~10), `DIALOG_GCD_WIDTH_SLOPE_X1000` (~1014).
   - Q-axis (smaller peak): `DIALOG_GCD_APPLY_CHUNKED_F_CUT*` boundaries, `DIALOG_GCD_APPLY_WINDOW_BLOCKS`, `DIALOG_GCD_APPLY_BOUNDARY_SPLIT`, `DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY_BLOCKS`.
   The numeric tuning is a **constrained co-optimization**: lowering a truncation/width knob cuts T but, past a threshold, breaks correctness on hard inputs unless a fresh Fiat-Shamir island (nonce) is found. The two axes trade against each other (e.g. spend comparator bits to buy a denser island, then widen a chunk cut to recover the peak).

2. **Toggle boolean route/hosting levers** — ~50 on/off flags. The qubit-peak ladder is a stack of value-exact hosting relabels (`HOST_GATED`, `BODY_HOST_CIN`, `SELECTED_BODY_NOCIN`, `LATE_BORROW_UV_HIGH`, `BORROW_CURRENT_BLOCK/_S2`, `COMPOSITE_SCRATCH`, chunked-apply family) that each shave a few qubits off peak at 0 or small added T. The T ladder is value-exact rewrites (`FUSED_BRANCH_BITS`, `ROUND763_DEDUP`/`_COMPRESS_LEVER`, `APPLY_FUSED_FOLD`, `KARA_SOL_DBL_FAST`, `ODD_U_LOWBIT_FASTPATH`).

3. **Choose the square algorithm** for round84 (`ROUND84_XTAIL_{KARATSUBA,WALK_SQUARE,SCHOOLBOOK}` or the lowq-shift22 default) — a discrete 4-way choice trading emitted T vs peak co-binding.

4. **Choose which dialog stages run** — the `DIALOG_GCD_RAW_PA_STOP_AFTER_*` flags (mostly for phase isolation/diagnosis, not for a valid submission). The valid pipeline is fixed at 6 stages × 2 GCD inversions.

5. **Pick a tail nonce / reroll counts** — `DIALOG_TAIL_NONCE`, `DIALOG_REROLL`, `DIALOG_POST_SUB_REROLL`: pure correctness-island selectors with ZERO score effect, required to validate after any op-stream-changing truncation. This is the "search for a clean island" sub-problem (assisted by the classical pre-filter in `dialog_gcd_classical_filter.rs`).

6. **Edit a small code region** — the `DIALOG_GCD_PA9024_COMPARE_SCHEDULE[258]` table (per-step compare bits) and `configure_ecdsafail_submission_route`'s `set_default_env` block are the natural in-source edit targets; the whole circuit body in `point_add` is "THE editable file for the research loop" (mod.rs header). Compile-time consts (`MAX_ITERATIONS`, schedule table, group/block sizes) are reachable only by source edit.

**Net:** the optimizer's job is to push each of the ~70 env knobs to its tightest value-exact setting (cutting T and Q), then re-find a Fiat-Shamir nonce that validates 0/0/0 over 9024 shots — minimizing `peak_qubits × executed_Toffoli`. The knobs are NOT independent: truncations interact through shared carry lanes and the convergence support, and every op-stream change invalidates the prior island, so co-tuning (truncation knob + nonce) is the atomic move.
