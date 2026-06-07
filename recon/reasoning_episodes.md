# Reasoning Episodes — ECDSA.fail point-add solver (SFT seed corpus)

Mined from `ecdsafail-challenge/src/point_add/`: the live audit-loop note,
the dated `memory/*.md` frontier notes, and ~20 deleted research/plan docs
recovered from git (all deleted at commit `69bd3d7`, read via `git show
<sha>^:<path>`).

This is supervised-fine-tuning seed material. Every episode below quotes REAL
knob names and REAL measured numbers so a model can learn (a) the Tony audit
shape `inspect -> diagnose -> cite evidence -> explain impact -> smallest fix
-> validate -> classify`, and (b) the Anton submission-hygiene shape `claim
stack -> role safety -> claim hygiene -> positioning -> actionable note`.

## Problem / Harness context (constant across episodes)

- Task: reversible in-place secp256k1 affine point-add `|Px,Py,0_anc> ->
  |Rx,Ry,0_anc>`, classical offset `Q=(ox,oy)`, `n=256`.
- Primary metric (SCORE, lower=better): `avg_executed_Toffoli x peak_qubits`.
- Acceptance gate: over `9024` shots (24 seeds x ~376), require `0` classical
  mismatches, `0` phase-garbage batches, `0` ancilla-garbage batches.
- Two failure-mode classes the audit must distinguish:
  - **Structural / value error**: a truncation drops a bit that is sometimes
    nonzero on the reachable support -> mismatches that NO nonce dodges.
  - **Fiat-Shamir / tail-nonce island**: the lever is value-exact on the
    reachable support; residual failures are stream-dependent and dodged by
    re-rolling `DIALOG_TAIL_NONCE` / `DIALOG_REROLL` / `DIALOG_POST_SUB_REROLL`.
- The whole loop is "find a structurally-attractive Toffoli/qubit cut, prove it
  value-exact, then hunt a clean Fiat-Shamir island for it." Brute force stops
  when failures repeat without a source-backed reason.

Two eras are visible in the corpus:
- **Era 1 (Apr 22 - Apr 28): architecture research.** Old 2-Kaliski scaffold
  (~4.18M Toffoli / 2716q). Episodes are about whether a cheaper architecture
  even exists (1-inversion, coset, venting, Luo, jump-GCD).
- **Era 2 (Jun 02 - Jun 06): dialog-GCD grinding.** A 1-inversion dialog-GCD
  core is live; episodes are precise binder-elimination + truncation-island
  notch loops carrying the score from ~2.74B down to ~1.96B.

---

# ERA 1 — Architecture research episodes (2-Kaliski scaffold, ~4.18M/2716q)

## E1. The "1 vs 2 inversion" binary question (the central architectural lever)
- **Frontier/situation**: 2-Kaliski scaffold, `4,136,878` Toffoli / `2716` q
  (commit `c1aeeb4`). Kaliski is 78% of cost: `kal_*` fwd phases `1,603,088`
  CCX + `bk_*` bwd `1,610,283` CCX = ~`1.60M` per invocation, x2 invocations.
- **Hypothesis**: "Gap to Google 2.7M is ~1.4M ~= exactly one Kaliski
  invocation. Can point-add be done with ONE Kaliski meeting the in-place
  contract?" (from `research_log_2026_04_27.md`).
- **Evidence cited**: benchmark phase trace (`TRACE_PHASES`); `single_inv_numeric.rs`
  3 replayed strategies; Bennett reversible-computation argument that
  forward+backward = 2x forward.
- **Smallest fix attempted**: none yet — settle the question from primary
  sources before coding (`No code changes to build()` discipline).
- **Validation**: primary-source read of Google 2026, HRSL 2020, Kim 2026,
  Litinski 2023, GE2021. Finding: **Google's 2.7M is for the HARDER windowed
  task `Q += P[k]` (w=16, 3 table lookups, ~200k lookup overhead), so implied
  bare-add cost ~2.5M.** Our bare-add 4.14M is ~1.5x over; that is a "technique
  gap," not a "magic trick." HRSL confirms 2 divisions per Weierstrass add.
- **Outcome**: dead-end as stated (no public 1-inversion affine scheme), but
  reframed the gap as believable.
- **Lesson**: Always check you are comparing the same functional circuit.
  Our harness measures **bare** point-add (classical Q, no lookup) — strictly
  simpler than Google's windowed task; the published SOTA number is not an
  apples-to-apples target.

## E2. Single-inversion Strategy A — invert `dx*dy` bundle (DEAD via algebra)
- **Hypothesis**: invert product `a = dx*dy` once, extract `1/dx = dy*a^-1`,
  finish affine without a 2nd Kaliski (`single_inv_plan.rs`).
- **Evidence/smallest test**: `replay_strategy_a()` classical replay, 200 random
  curve-point pairs, tracking each register as U256.
- **Validation**: `Rx` matched 200/200; **`Ry` off by exactly `+dy`**. Step-7
  algebra: starting `ty=dy`, `ty += lambda*tx` lands `ty = Py - 2Qy - Ry`; the
  `Py` term has no classical handle (Py is quantum, overwritten).
- **Outcome**: DEAD — "Py trapped in ty," unrescuable without an extra n-qubit
  output register.
- **Lesson**: write the falsification prediction BEFORE running ("Rx will match,
  Ry will NOT, off by the obstruction term") — a passing replay that matches the
  predicted-failure is still informative.

## E3. Single-inversion Strategy C — invert `w = dx^3` (CLASSICALLY ALIVE)
- **Hypothesis**: one Kaliski on `w = dx*(Rx-Qx)*dx^2 = dx^3` yields BOTH inverses
  (`1/dx` and `1/(Rx-Qx)`) since `w^-1 * dx^2 = 1/(Rx-Qx)`, etc.
- **Evidence**: algebra closes: `dx^2*(Rx-Qx) = dy^2 - dx^2*(Px+Qx)`.
- **Validation**: `replay_strategy_c()` 200/200 on both Rx and Ry, **ancilla
  leak NONE** (vs Strategy B2 which leaks a `-lambda` copy).
- **Outcome**: classically alive; but the quantum schedule was never written —
  honest op-count `~3.7-4.3M` Toffoli (saving 13-24% vs 4.91M@iters=511), NOT
  the "3M napkin number from earlier monologues." Peak budgeting blocked it:
  naive ~7n persistent extra registers overshoot the 2800q cap; needs Bennett
  interleaving (free dx^2 after dx^3, etc.).
- **Lesson**: a classically-correct strategy is necessary but not sufficient —
  it is "worth implementing" only after the reversible peak AND Toffoli both
  pencil under the caps. Replace optimistic napkin numbers with an honest
  per-op tally before committing.

## E4. The `s = p` register-recycling insight (KAL_FREE_S)
- **Hypothesis**: after forward Kaliski (407 iters) the state is deterministic:
  `u=1, v_w=0, r=-inv_raw, s=p` (exactly the secp256k1 prime). So free `s` with
  X-flips (0 Toffoli) and reuse the 256q.
- **Evidence**: `/tmp/our_kaliski.py` 5 random trials all show `s=p` at iters=407
  and 511; `s != p` at 256/350 (Kaliski not yet terminated). Matches HRSL "three
  registers contain 0, 1, and the modulus p."
- **Smallest fix**: `KAL_FREE_S` (default on); X-flip p to zero s after forward,
  re-alloc + X-flip p to reload before backward.
- **Validation**: body peak dropped ~2460 -> ~2204 at `pair1_mul1/mul2`. **Global
  peak NOT reduced** — backward `bk_step6_7_8` still hits 2716 because s is
  re-allocated and live during backward.
- **Outcome**: partial win (body), architectural groundwork.
- **Lesson**: reducing ONE peak phase does not move the global peak when other
  co-bound phases sit at the same height. Peak is a max over ALL phases.

## E5. Classical m_hist via measurement-uncompute (DEAD — repeated independently)
- **Hypothesis**: replace the 407q `m_hist` quantum register with a classical
  bit via HMR + `cz_if` phase correction (`KAL_M_CLASSICAL=1`).
- **Validation**: classical correctness 0 mismatches, but **320 phase-garbage
  batches** across 5 seeds x 4096 shots.
- **Root cause**: HMR returns a RANDOM classical bit, not the stored quantum
  value; `m_hist` is REUSED in the backward pass, so the random bit can't
  represent the entangled value -> residual phase.
- **Outcome**: DEAD.
- **Lesson** (stated twice in the corpus): "Measurement-based uncompute works
  only for qubits NOT used after measurement." A reuse-after-measure is the
  disqualifier — check the lifetime before reaching for MBU.

## E6. Step-9 cswap removal with persistent `a_f` (DEAD — different trajectory)
- **Hypothesis**: the two cswaps per iter (step3 + step9) cost ~660k Toffoli;
  keep `a_f` live across rounds and apply one final parity-swap to drop step9.
- **Evidence/test**: Python classical sim FIRST (cheap falsification).
- **Validation**: variants diverge after ~10 iters; only **13/30 trials** match
  final `(u,v,r,s)` even with a final parity-swap correction.
- **Root cause**: `a_k` is computed from CURRENT register contents; without
  step9 the contents evolve differently, so the no-step9 variant computes a
  genuinely DIFFERENT trajectory, not a swap-equivalent one.
- **Outcome**: DEAD. (The cswap merge identity `cswap(c1)·cswap(c2) =
  cswap(c1^c2)` only holds when the two swaps are ADJACENT — see E13.)
- **Lesson**: validate "obviously equivalent" reorderings with a classical sim
  before any quantum work; cross-iteration coupling breaks naive swap algebra.

## E7. u64 shift UB bug in the venting adder (FIXED — subtle, context-specific)
- **Frontier**: venting adder (Gidney 2025, arXiv 2507.23079) ported as 9
  primitives in `venting.rs`, all passing ~1000 standalone trials.
- **Symptom**: wiring venting-halve into backward Kaliski gave **320
  phase-garbage batches**.
- **Diagnosis**: Rust `x >> k` for `k >= 64` is UB; release `-O` x86_64 does
  masked shift `k % 64`, so `bit(k)` returned phantom set-bits at positions
  64/128/192.
- **Smallest fix**: `if k >= 64 { false } else { (x >> k) & 1 != 0 }` in all 6
  occurrences.
- **Validation**: 320 -> **1 phase batch / 20480 shots** (deterministic at
  seed=3).
- **Outcome**: mostly fixed; residual 1-batch is a separate cross-call issue
  (E8).
- **Lesson**: standalone primitive tests (phase-clean in isolation) do NOT catch
  context-specific interactions; the bug only surfaced when wired into a real
  shared-dirty-qubit sequence.

## E8. The seed-3 cross-call phase leak (DIAGNOSED via bisection, unresolved)
- **Symptom**: 1 phase batch / 20480 shots at seed=3 after the UB fix.
- **Diagnosis (call-count bisection)**: `0..577` calls -> 0 batches; `0..578` ->
  0; **`0..579` -> 2**; `0..1000` -> 1; full -> 1. Different call subsets give
  different counts => a cross-call phase interaction, NOT a primitive bug.
  Gidney's `cz_if(dirty, vent_keys)` corrections aren't composing across
  sequential calls that share dirty qubits.
- **Outcome**: identified, parked (compare emitted gate stream vs Python ref at
  2 sequential `cisub_dirty_2clean_classical` sharing dirty qubits).
- **Lesson**: when a count changes non-monotonically with a prefix-length sweep,
  the bug is in cross-call composition (shared ancilla state), not the unit.

## E9. Rowwise mul to shrink `tmp_ext` 2n->n (DEAD on peak — multi-site peak)
- **Hypothesis**: `KAL_ROWWISE_MUL=1` streams schoolbook so tmp is n-wide not
  2n, saving 256q at `pair1_mul1`.
- **Validation**: at `pair1_mul1` only: correctness ok, **peak unchanged 2716**
  (other phases hit it), Toffoli +300k. Applied to all 3 mul sites: 1
  phase-garbage batch (halve-back uncompute edge case).
- **Lesson**: same as E4 — to move peak you must cut ALL simultaneous-peak
  phases; a single-site cut buys nothing on the product.

## E10. Coset representation (Zalka / GE2021) — recurring blocked candidate
- **Hypothesis**: coset encoding `k mod N -> sum|jN+k>` makes modular add cost
  `4n` vs `10n` (~60%), padding `cpad ~= 26` qubits (deviation ~1e-8).
- **Blocker (deterministic)**: harness checks `get_register == expected mod p`;
  coset registers hold `jN+k` (matches mod p but exceeds n bits). Also coset
  **breaks comparators** (`u > v_w`, `v_w == 0`) which Kaliski relies on.
- **Conditional unblock**: under the later-confirmed approximate-correctness
  option (<= 0.1% / <= 9 failures, padding residue OK), coset on NON-Kaliski
  mod-adds (Path A) is viable: ~500k CCX of targets, 60% -> ~300k savings (~7%).
- **Outcome**: parked; never crossed the apples-to-apples + comparator hurdle.
- **Lesson**: a technique's headline speedup is gated by harness semantics and
  by which sub-ops it breaks (comparators). Apply it only where no comparator
  lives, or exit/re-enter coset around comparisons at small `cpad` cost.

## E11. Qubit-floor reality check — 1200q is Google's withheld circuit
- **Situation**: user wants ~1100-1200q AND few-million Toffoli.
- **Evidence (literature tally at n=256)**: Luo 2025 = 1333q but ~200M
  Toffoli/pt-add; Chevignard = ~1100q but >3000M (and outputs a 1-bit Legendre
  hash, NOT exact `(Rx,Ry)` — disqualified); HRSL Low-W = 2124q / ~12-66M;
  Google low-qubit = 1175q / 2.7M (withheld, coset+windowed).
- **Outcome**: honest conclusion — "no public method achieves BOTH <=1200q AND
  <=5M Toffoli; that point is Google's withheld circuit." Offered operating-point
  options A (Luo, qubit-first, 100-500M Toffoli) / B (Toffoli-first, ~2700q) /
  C (hybrid ~1500-1800q) / D (approximate single-inversion).
- **Lesson**: when the target is provably outside the public frontier, say so
  explicitly and convert the goal into a choice of operating point, rather than
  brute-forcing toward an unreachable pair.

## E12. Bernstein-Yang & jump-GCD survey — matrix-entry growth kills the naive win
- **Hypothesis**: batch T binary-GCD steps into a 2x2 transition matrix
  (safegcd-style) selected from the low ~2T bits; apply once full-width => kill
  per-step cswaps.
- **Evidence (classical surveys, real secp256k1 inputs)**:
  - B-Y divstep2 (w=1): observed iters min 502 / max 567 / mean 531 vs the
    pessimistic bound 742; modinv 10,000/10,000 — but per-iter reversible cost
    `10-12n` CCX still worse than Kaliski's ~`2180`.
  - Corrected jump survey: scaled matrix entries DO hit full `2^w` growth
    (w=16 -> max |entry| 65536), restoring the pessimistic cost model.
  - BUT class compression is real: distinct transition matrices = `2^(w-1)`x
    fewer (w=8: 1,343,488 states -> 10,496 matrices, 128x); the `(r,s)` side
    compresses IDENTICALLY to `(u,v)`; **joint pairs stay equal (125/125/1133)**.
  - Key `(u_low,v_low,cmp0,cmp1,cmp2)` determines the **3-step bulk prefix
    EXACTLY** on 99.17% (full 4-step) windows; residual ambiguity is just the
    final odd/odd branch bit; only 36 distinct bulk 3-step transforms.
- **Outcome (Era 1 verdict)**: full B-Y replacement too expensive; best prototype
  is "exact 3-step bulk core + cheap residual + tail fallback." (Era 2 later
  measures the realized jump and finds K2 optimal — see E20.)
- **Lesson**: a scary raw state-space can hide a tiny EXACT local-transition
  family; measure distinct-class counts and the minimal disambiguating key
  before declaring batching dead OR alive.

## E13. cswap-merge identity is real but blocked by asymmetric neighbors
- **Hypothesis**: merge adjacent apply cswaps via `cswap(c1)·cswap(c2) =
  cswap(c1^c2)`.
- **Diagnosis**: in the apply step `double_y; cadd(b0); cswap(c)`, consecutive
  cswaps are separated by `double_y` and `cadd`, both ASYMMETRIC (single out y).
  Commuting a cswap through them conjugates into c-CONTROLLED double/add (~2x
  cost on both registers) -> net LOSS. dialog-GCD already uses only 1 cswap/step
  (the old 2-cswap/iter merge was captured by the algorithm choice).
- **Outcome**: BLOCKED; "nothing left to merge."
- **Lesson**: an algebraic identity needs its operands ADJACENT; intervening
  asymmetric ops turn a free merge into controlled ops that cost more than the
  thing removed.

---

# ERA 2 — dialog-GCD grinding episodes (1-inversion core, score 2.74B -> 1.96B)

The dialog-GCD core runs ONE half-GCD forward (records branch bits to a
compressed transcript "log"), applies a Bezout reconstruction, then reverses
the GCD (same family as Schrottenloher 2026's "dialog" decomposition). Two phase
families dominate: **tobitvector** (forward GCD body) and **apply** (Bezout
reconstruction). The grind is binder-elimination + truncation-island hunting.

## E14. Odd-u low-bit body skip (WIN -2.5M score)
- **Frontier**: dialog-GCD compressed sidecar, measured apply sub.
- **Hypothesis**: in the controlled sub/add body, the reachable branch-swap
  state has `subtrahend[0]=1`, `acc[0]=ctrl`, so bit-0 computes `ctrl-ctrl=0`
  with no borrow into bit 1; lane-0 reduces to `CX(ctrl, acc[0])` and the
  Cuccaro body can start at bit 1.
- **Smallest fix**: extend the odd-u low-bit fastpath into the measured
  tobitvector add/sub body; co-tune `DIALOG_REROLL=1`, `DIALOG_POST_SUB_REROLL=12`.
- **Validation**: 9024/9024 OK, 0/0/0. avg Toffoli `1,745,201`, peak `1571`,
  score `2,741,710,771` (delta vs `005e17a`: **-2,507,316**).
- **Lesson**: when a low bit is provably fixed on the reachable support, the
  whole bit-0 lane collapses to a single CX — start the adder one bit higher.

## E15. Fused branch bits (WIN) — and a paired over-reach (PARTIAL)
- **Hypothesis**: `DIALOG_GCD_FUSED_BRANCH_BITS=1` fuses the branch comparator
  with the `b0` controlled update of `b0_and_b1`.
- **Validation**: 0/0/0 over 9024; `1,861,990` Toffoli, `1698` q.
- **Paired over-reach (do not reuse)**: `DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS=1`
  was classically correct WITH fusion but **phase-dirty: 141 phase-garbage
  batches**; dropping the pseudo-Mersenne special-fold width to 8 crossed 3B
  statically but **failed all 9024**.
- **Lesson**: bundle one validated win with explicit "do-not-reuse" notes on the
  adjacent knobs that looked attractive but were classical-correct/phase-dirty
  or value-broken — saves the next solver the same dead-ends.

## E16. cswap FLOOR analysis (mostly NEGATIVE — the honest "stop here" episode)
- **Frontier**: clean baseline `1,704,086` / `1698`. Targets: tobitvector cswap
  fwd+rev `227,720`; apply cswap `204,288` (full-width N=256 over x/y).
- **Three approaches, all BLOCKED, each with a cited mechanism**:
  1. Merge adjacent apply cswaps: NO — separated by asymmetric `double_y/cadd`
     (see E13).
  2. Tighten/fuse the tobitvector cswap: TAPPED — width LOCKED to active_width
     (the following controlled-sub reads the same bits). Envelope floors proven:
     slope floor 0.7075 (0.74->303 mism, 0.78->6208, 0.82->9005); margin floor
     27 (26 -> 1+ mism, 24 -> 9-18). `ACTIVE_ITERATIONS=399` is the convergence
     floor (397 -> 1 input fails).
  3. Relabel/measure the cswap away: NO — swaps act on LIVE data not ancilla, so
     MBU doesn't apply; Hmr gives a random faithful outcome, can't read a control
     into `c_condition`; ~3/4 of apply swaps that don't fire still cost a full
     256-CCX sweep (information-theoretically locked).
- **Empirical proof apply cswap is full-width**: `DIALOG_GCD_APPLY_CSWAP_TRUNC_W=255`
  (drop only top bit) -> **9024/9024 classical mismatches + 141 phase-garbage**.
  x,y are uniform 256-bit field elements; no envelope exists.
- **Only clean win**: `DIALOG_GCD_WIDTH_MARGIN=27 DIALOG_REROLL=5` ->
  `1,704,086 -> 1,699,450` (-4,636), peak 1698, 0/0/0 (cswap share only ~1,432;
  rest is co-located sub/add).
- **Outcome**: the big cswap cut is only reachable via an algorithmic
  jump/safegcd rewrite — not a single-session flag.
- **Lesson**: prove a width is irreducible with a one-bit truncation probe
  (`*_TRUNC_W=255`) — a 9024/9024 wipeout is decisive evidence the data fills
  the full width, and saves a long doomed island hunt.

## E17. Chunked apply-F: peak-safe replacement (WIN, value-neutral F_CUT)
- **Hypothesis**: instead of materializing all 256 bits of `f = ctrl & a` across
  the apply ripple, load one slice at a time via `DIALOG_GCD_APPLY_CHUNKED_F_BLOCKS=2`,
  clear by HMR, clear the carry boundary with a controlled truncated comparator.
- **Key property**: the chunked apply is EXACT regardless of `F_CUT` (full
  Cuccaro per block + exact boundary-borrow clear); `F_CUT` is **value-neutral**,
  it only rebalances block widths (block 1 = `[F_CUT,257)`, whose f-load+carry
  lane is the peak transient) and reseeds Fiat-Shamir.
- **Validation**: `F_CUT=70` clean at reroll 4/15: `1567` q, `1,689,505` T,
  score `2,647,954,335`. Cut scan: 68 -> first seed failed (needs reroll);
  70 -> clean; 72 -> `1,691,101` T (rising).
- **Lesson**: a value-neutral knob is the ideal binder lever — it can only change
  peak and the FS stream, never correctness; sweep it to the lowest peak that
  still clears the next floor, then island-hunt only the stream.

## E18. The binder-walk pattern: 1567 -> 1543 -> ... (REPEATABLE WIN recipe)
- **Pattern (named in the corpus)**: "each round is identify co-binders ->
  eliminate one -> widen `F_CUT` to the next floor." Break-even ~`1,700
  Toffoli/qubit` for a product-neutral trade.
- **Concrete steps**:
  - `ROUND84_XTAIL_BORROW_CARRIES=1` + `F_CUT 70->78` sinks apply 1558 -> 1543
    (round84 doubles floor 1542). cut=77 -> 1544, cut=78 -> 1543 (lowest cut
    reaching the floor). Cost +6,384 T for -15 peak (inside break-even).
    Island: COMPARE_BITS=59, F_BLOCKS=2/F_CUT=78, APPLY_CLEAN_COMPARE_BITS=19,
    REROLL=28/POST_SUB_REROLL=5 -> score `2,615,519,241`, 0/0/0.
- **Lesson**: peak reduction is a sequence of co-binder eliminations, not one
  knob; after each elimination the previously-second-highest phase becomes the
  binder, so re-trace and widen the value-neutral cut to the new floor.

## E19. Body c_in host + uv-high carry borrow: 1466 -> 1446 (WIN, value-exact)
- **Frontier**: four co-bound phases pinned 1466 (two apply add/sub, two GCD-body
  add/sub).
- **Two value-EXACT carry-lane reclaims (no truncation lottery)**:
  - `DIALOG_GCD_BODY_HOST_CIN=1`: the body adder allocated a FRESH `c_in` even
    though the carry lane was borrowed. With odd-u fastpath `body_start=1`,
    `gated[0]` stays |0> and is disjoint from operands+carries, so it serves as
    the Cuccaro carry-in. Exact: c_in=0 either way, restored to |0>.
  - `DIALOG_GCD_LATE_BORROW_UV_HIGH=1`: at late steps the compressed log shrank
    below `2*active_width-1`, so the body fell back to allocating its own lane.
    The GCD has converged there, so `u[active_width..]` is |0> by the SAME
    premise width-truncation already relies on -> use it as scratch.
  - `F_CUT 116 -> 126` sinks the apply pair to 1446 (their min; >126 rebalances
    upward).
- **Validation**: island REROLL=7/POST_SUB_REROLL=13, 0/0/0 over 9024;
  `1466 x 1,732,283 = 2,539,526,878 -> 1446 x 1,740,263 = 2,516,420,298`
  (-23,106,580, -0.91%).
- **Lesson**: prefer value-EXACT reclaims (reuse provably-zero idle qubits as
  carry/scratch) over width-truncation gambles — they carry NO new failure mode,
  only a Fiat-Shamir re-roll. "Borrow from what the envelope already guarantees
  is zero."

## E20. Deeper jump-GCD K3/Kj measured qubit-NEGATIVE; K2 is the optimum
- **Frontier**: deployed route runs K2 (`DIALOG_GCD_K2=1`, removes up to one
  extra trailing zero of v/step; flag in `k2_shift2_log`).
- **Hypothesis**: generalize to K3/Kj for more step reduction.
- **Smallest test**: classical convergence model FIRST (`ISLAND_MEASURE_JUMP=1`),
  over the 18048 real GCD factors of the baked nonce, before any quantum rewrite.
- **Evidence (measured table)**: depth1 401 steps / log 670; **K2: 259 steps
  (-35%) / log 696 (+4%)** — huge Toffoli win, ~free on qubits; K3: 227 (-12%
  more) / log 836 (+20%, +140q at peak); deeper worse. Each jump level adds one
  UNCOMPRESSED shift flag/step (`5 + 3*(j-1)` bits/block); shift2 is ~Bernoulli(1/2),
  incompressible; the log is the largest single peak block (~691q).
- **Outcome**: K3/Kj CONTRAINDICATED (raises the product); K2 is the sweet spot.
- **Lesson**: measure the classical convergence/log-size tradeoff BEFORE the
  quantum rewrite; a step-count win that inflates the peak-binding log can be
  net-negative on the `Toffoli x qubits` product.

## E21. Ghost-log / log-pack already deployed (DON'T re-derive)
- **Investigation**: try "spooky-pebble register dropping" and "host transcript
  log on freed u/v-high bits" to break the ~1319q floor.
- **Finding (`ISLAND_DUMP_PEAK`)**: peak `1319` at `apply_chunk_sub_final_ripple`;
  the bare ~691q log is NOT all co-live with tx+ty. Both ideas are ALREADY
  enabled: `DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY=1 (_BLOCKS=999)` (log-pack),
  `DIALOG_GCD_COMPRESSED_BLOCK_LIFECYCLE=1` + `HOST_REVERSE_RAW_BLOCK=1`
  (spooky-pebble), plus `APPLY_REPLAY_SWAP_HOST`, `BODY_HOST_CIN`,
  `LATE_BORROW_UV_HIGH`, `BRANCH_BITS_HOST_COMPARATOR`.
- **Outcome**: the binder is the apply ripple, not the bare log; remaining EV is
  the truncation-island notch loop + a dedicated apply-ripple/tobitvector
  co-binder teardown.
- **Lesson**: before "inventing" a structural qubit lever, dump the peak phase and
  check the current config — the idea may already be deployed and the true binder
  is elsewhere.

## E22. 1320q apply teardown (STRUCTURAL win + island problem)
- **Hypothesis**: replace the final apply chunk's fast ripple with no-carry
  Cuccaro (`DIALOG_GCD_APPLY_FINAL_LOWQ=1`, drops final ripple from 1382q,
  +44,376 T), split the boundary comparator (`DIALOG_GCD_APPLY_BOUNDARY_SPLIT`),
  rebalance multi-cuts (`F_CUT/F_CUT2/F_CUT3`, custom 5-block).
- **Validation arc** (active=258/260, peak 1320): base seed `8 classical / 4
  phase`; +`KAL_DOUBLE_CARRY_TRUNC_W=22` -> `5 classical / 2 phase` (under
  budget). Hosted boundary split (split on a retained boundary carry) ->
  `1320q, 1,558,597 T`. Best structural seed (BLOCKS=5, custom cuts 50/100/150/200,
  ACTIVE=260) found clean at **nonce 108**: 0/0/0, score `2,066,350,440`.
  Active 261 fits 1320q but regresses (2c/2p); active 262 crosses to 1327q.
- **Outcome**: "1320q target structurally solved with T headroom; the old phase
  problem is gone at active 260 nonce 108." Reframed from brute-force to a
  structural teardown + small island.
- **Lesson**: convert "wide blind nonce search" into "structural teardown that
  reaches the target peak, then a SMALL island hunt for the residual phase" —
  and note a fast classical/phase pre-filter is needed because full eval is too
  slow for wide scans.

## E23. Dropping a whole GCD iteration also drops PEAK (the width-envelope lesson)
- **Frontier**: COMPARE_BITS=52 lineage, 4 promotions in one session.
- **Lesson 1 (the big one)**: `DIALOG_GCD_ACTIVE_ITERATIONS 259 -> 258` cut
  ~3,446 T AND moved peak **1390 -> 1382** (the dropped iteration's scratch row
  is no longer live) — first peak reduction in the lineage, worth far more than a
  carry-window bit because it re-opens the width x peak product.
- **Lesson 2 (break-set cancellation)**: on the 1382 base, `slope1005` alone =
  4+3 breaks, `active258` alone = 4+3, but **`slope1005 + active258` = 3+3**
  (gentler than either) -> two savings under ONE island. (Did NOT recur on the
  next base: `slope1006 + active257 = 26+18`, purely additive.) "Always measure
  the COMBO break count, not the sum."
- **Density datum**: phase breaks hurt island density more than classical count
  alone: 6+6 took ~13.6k nonces (~30 min) vs 3+3 at ~3.6k and 0+2 at ~55.
- **Lesson**: iteration-count is a coupled lever (Toffoli AND peak); and break-sets
  can CANCEL across levers, so co-search candidate combos rather than stacking
  serially.

## E24. KAL_FOLD 24->23 + tail-nonce island (PROMOTED, the canonical notch win)
- **Base**: promoted `1664274` (robertkodra, cb72): 1390q x 1,531,871 T.
- **Smallest fix**: `KAL_FOLD_CARRY_TRUNC_W 24 -> 23` (FOLD window only; DOUBLE
  stays 24). -518 T, peak-neutral 1390q. Value-exact on the reachable support
  (dropped fold-carry bit is 0 there); failures are pure Fiat-Shamir.
- **Density reality (measured at inherited nonce)**: this tight base breaks on
  every gentle lever — slope1005=13, margin8=16, applyc19=11, active258=10,
  kal_d23=9, **kal_f23=6 (gentlest)**; uniform body-carry trims break thousands
  (no exact slack on a tight base).
- **Island**: `DIALOG_TAIL_NONCE=3155`, found after ~3200 nonces of full
  validation (~7.5 min, 11 threads); density ~1/3000 for the 6-break lever.
- **Validation**: `cade2d0` PROMOTED, 1390q x **1,531,353 T = 2,128,580,670**
  (-720,020). Tool: `src/bin/island_search_jac.rs` (bit-exact self-check vs
  affine k*G, ~7 nonce/s).
- **Lesson**: on a tight base, pick the GENTLEST lever (fewest broken inputs);
  island density ~ `e^-(breaks)`, so a 6-break lever (~1/3000) is searchable
  locally while a 9-16-break lever (1e-4..1e-7) needs big remote compute.

## E25. The repeated dead-end of the late grind: structurally-attractive but island-limited
- **Frontier**: post-a66 (`a66b042`, 1,503,355 T x 1309 q = 1,967,891,695),
  multiple one-bit successors probed.
- **Pattern (from the audit-loop note)**: e.g. `APPLY_CLEAN_COMPARE_BITS=19`
  structural target 1,502,839 T (would beat by 675,444 if clean); `COMPARE_BITS=48`
  target 1,503,211 (beat by 188,496); `ACTIVE_ITERATIONS=257` target 1,500,368
  (beat by ~3.89M). EVERY one fails the inherited nonce (11-18 classical, 5-14
  phase) and the staged GCD pre-filter finds ~280-323 candidates passing 2,048
  shots but **0 passing the full 9,024-shot filter**. Rejects are
  width/nonconvergence, NOT comparator mismatches.
- **Tony classification**: "structurally attractive but full-shot GCD
  island-limited; the sampled rejects are width/nonconvergence." `ACTIVE=257`
  appears to create a nonconvergence FLOOR (full rejects: 102 nonconvergence +
  30 width).
- **Outcome**: dead-end without either (a) a genuinely faster full-shot nonce
  filter, or (b) a structural width/convergence relief.
- **Lesson**: STOP brute force when failures repeat without a source-backed
  reason. When 2,048-shot survivors all die at 9,024 shots, the bottleneck is
  the full-shot GCD width/convergence envelope, not the lever — widen the filter
  or relieve the envelope, don't sweep more nonces.

## E26. The validated successor that beat the inherited frontier (audit-loop, WIN)
- **Tony pre-change audit**: `DIALOG_GCD_APPLY_FINAL_LOWQ=0` +
  `APPLY_FINAL_WINDOWED_FAST_BLOCKS=0` removes the final apply chunk's
  low-q/windowed carry overhead while the global peak stays bound by
  `round84_fused_square_xtail_dx_sub_lam_square_lowq` at 1309q. Raw fast-final at
  active 258: structural target `1,486,327` T x 1309 = 1,945,602,043, but
  inherited nonce failed (19 classical / 8 phase) and the first 500 2k-shot
  survivors gave no full 9024 hit.
- **Smallest fix**: spend part of the recovered budget on convergence:
  `ACTIVE_ITERATIONS=262`, `WIDTH_MARGIN=10`, `WIDTH_SLOPE_X1000=1014` — stays
  1309q, structural target `1,497,795` T. GCD prefilter densified: 500 pass 2048,
  93 pass 4096, 6 pass 8192 (`614,1328,1718,2148,2432,2499`), 4 pass all 9024.
- **Quantum confirmation**: 1328 -> 1 phase-garbage; 2148 -> 1 classical + 2
  phase; **2432 -> CLEAN 0/0/0**; 2499 -> 1 classical + 2 phase.
- **Validation**: `1,497,795 x 1309 = 1,960,613,655`, beat `a66b042` by
  7,278,040. `./benchmark.sh --note 'validate lowq0 active262 nonce2432'`
  reproduced and wrote `score.json`.
- **Lesson**: a structural cut that fails on convergence can be RESCUED by
  spending part of the recovered Toffoli budget back on iterations
  (ACTIVE/MARGIN/SLOPE) — trade pure-Toffoli for convergence density so an
  island actually exists. Note even GCD-clean nonces (1328/2148/2499) can still
  fail quantum phase/classical validation — the GCD prefilter is necessary, not
  sufficient.

## E27. The Anton correction-note episode (submission hygiene under error)
- **Situation**: submission `436b516` promoted (`1,503,871` T x 1309 =
  1,968,567,139, beating `83e3b66` `1,968,793,475` by 226,336). The public prose
  had ARITHMETIC TYPOS in the displayed score and frontier delta, though the CLI
  claimed score/metrics/validation were correct.
- **Action**: public correction note `5ec74c1` recording the typo and affirming
  the CLI-claimed numbers.
- **Lesson (Anton claim hygiene)**: when the machine-verified number is right but
  the human-written prose is wrong, publish a correction that (a) names the
  artifact (`5ec74c1`), (b) isolates exactly what was wrong (displayed score +
  delta arithmetic), (c) re-affirms what remains trustworthy (the CLI score,
  metrics, validation, leaderboard result). Never quietly edit.

---

# Reasoning Patterns

This section distills the two reasoning STYLES the corpus is built to teach.

## Pattern A — Tony pre-change audit (inspect -> diagnose -> cite -> impact -> smallest fix)

A disciplined pre-change audit, used before touching anything:

1. **Problem**: name the exact waste/risk/contradiction. ("The body adder
   allocates a FRESH `c_in` even though the carry lane was borrowed — that single
   ancilla pins the peak at 1466.")
2. **Evidence**: cite a FILE / FUNCTION / ENV KNOB + a current METRIC + a prior
   note. ("`add/sub_nbit_qq_fast_borrowed_carries`, peak phase
   `materialized_sub_body`, 1466q; see [[2026-06-02-cswap-floor-analysis]].")
   Never "a prior session said X" as authority — re-verify from source.
3. **Why it matters**: expected Toffoli / qubit / correctness / phase / cleanup
   impact, quantified. ("Body phases 1466 -> 1446; exact, c_in=0 either way.")
4. **Source check**: compare against harness invariants (9024 shots, 0/0/0) and
   the CURRENT promoted best; confirm the change is value-exact vs an
   island-gamble.
5. **Smallest useful fix**: ONE bounded change. (`BODY_HOST_CIN=1`, not a body
   rewrite.) One knob, one bit, one lane.

## Pattern B — Tony post-run audit & failure classification

After validating, classify EVERY residual failure as one of:
- **Structural / value error**: fails on the reachable support, no nonce dodges
  it (e.g. `APPLY_CSWAP_TRUNC_W=255` -> 9024/9024 mismatches). Revert; the width
  is irreducible.
- **Fiat-Shamir / tail-nonce island**: value-exact lever, stream-dependent
  residual; dodge via `DIALOG_TAIL_NONCE` / `REROLL` / `POST_SUB_REROLL`. Island
  density ~ `e^-(breaks at inherited nonce)`.
- **Convergence floor**: full-shot GCD rejects dominated by nonconvergence/width
  (e.g. ACTIVE=257: 102 nonconvergence + 30 width). Spend budget back on
  iterations or relieve the envelope.
- **Measurement noise**: rare, transient.
STOP brute force when failures repeat without a source-backed reason (e.g.
2,048-shot survivors all die at 9,024 — bottleneck is the envelope, not the
lever).

## Pattern C — falsify cheaply and FIRST

- Run a CLASSICAL replay / Python sim before any quantum work (Strategies A/B/C;
  step-9 removal; jump convergence model). Write the predicted failure mode
  BEFORE running ("Rx matches, Ry off by +dy").
- Use a one-bit truncation probe (`*_TRUNC_W=255`) to prove a width is
  irreducible in one shot.
- Use a staged shot filter (512 -> 2048 -> 4096 -> 8192 -> 9024) and a classical
  GCD pre-filter to triage nonces cheaply before the full ~10.24M-op eval.
- Bisect by prefix/call-count to localize cross-call phase leaks.

## Pattern D — the binder/notch grind loop (Era 2 core skill)

1. Dump the peak phase (`TRACE_PEAK`, `ISLAND_DUMP_PEAK`); identify ALL co-bound
   phases at that height (peak is a max — cutting one of several does nothing).
2. Eliminate ONE binder, preferably by reusing provably-zero idle qubits as
   carry/scratch (value-EXACT, no new failure mode) rather than width truncation.
3. Widen the value-neutral cut (`APPLY_CHUNKED_F_CUT`) to sink the next phase to
   the new floor.
4. Re-roll the Fiat-Shamir island for the new op-stream (2-D `REROLL x
   POST_SUB_REROLL`, ~0.4-0.6% clean density).
5. Respect break-even (~1,700 Toffoli/qubit for a product-neutral trade) and
   measure COMBO break-counts (levers can cancel).

## Pattern E — Anton submission hygiene (claim stack -> role safety -> claim hygiene -> positioning -> note)

1. **Claim stack**: exact change, exact score, exact validation status, exact
   caveat. ("`KAL_FOLD_CARRY_TRUNC_W 24->23` + `TAIL_NONCE=3155`; 1390q x
   1,531,353 T = 2,128,580,670; 0/0/0 over 9024; -720,020 vs base.")
2. **Role safety**: keep ECDSA.fail / Eigen-Google / StarkWare / Starknet / SNF
   roles distinct; don't conflate the benchmark with the entities behind it.
3. **Claim hygiene**: do NOT claim ECDSA is practically broken today, or that any
   system is fully post-quantum safe. Don't inflate napkin numbers (call out the
   honest 3.7-4.3M vs the wishful 3M). Publish corrections when prose disagrees
   with the verified metric (E27).
4. **Positioning fit**: this is a quantum-circuit OPTIMIZATION benchmark and a
   durability-measurement signal — frame wins as score deltas on a shared
   leaderboard, not as cryptographic breaks.
5. **Actionable note**: every public note must help the next solver either
   REPRODUCE the win (exact knobs + nonce + benchmark command) or AVOID the
   dead-end (the negative-evidence notes: 0/9024 full-shot survivors, K3
   qubit-negative, cswap floor, do-not-reuse adjacent knobs).

## Quick knob glossary (canonical names, from recon/knobs.md)

- Width envelope: `DIALOG_GCD_WIDTH_MARGIN`, `DIALOG_GCD_WIDTH_SLOPE_X1000`,
  `DIALOG_GCD_ACTIVE_ITERATIONS`, `DIALOG_GCD_COMPARE_BITS`,
  `DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS`.
- Apply teardown: `DIALOG_GCD_APPLY_CHUNKED_F_BLOCKS`,
  `..._F_CUT[/2/3/4]`, `DIALOG_GCD_APPLY_FINAL_LOWQ`,
  `DIALOG_GCD_APPLY_BOUNDARY_SPLIT`, `DIALOG_GCD_APPLY_FINAL_WINDOWED_FAST_BLOCKS`.
- Carry-lane hosting (value-exact): `DIALOG_GCD_BODY_HOST_CIN`,
  `DIALOG_GCD_LATE_BORROW_UV_HIGH`, `DIALOG_GCD_BRANCH_BITS_HOST_COMPARATOR`,
  `DIALOG_GCD_APPLY_REPLAY_SWAP_HOST`, `DIALOG_GCD_COMPRESSED_LOG_U_HIGH_RUNWAY`,
  `DIALOG_GCD_COMPRESSED_BLOCK_LIFECYCLE`, `DIALOG_GCD_HOST_REVERSE_RAW_BLOCK`.
- Fastpath / fusion: `DIALOG_GCD_ODD_U_LOWBIT_FASTPATH`,
  `DIALOG_GCD_FUSED_BRANCH_BITS`, `DIALOG_GCD_K2`, `DIALOG_GCD_ROUND763_COMPRESS_LEVER`.
- Truncation (island-gated): `KAL_FOLD_CARRY_TRUNC_W`, `KAL_DOUBLE_CARRY_TRUNC_W`,
  `DIALOG_GCD_BODY_CARRY_TRUNC_W`, `DIALOG_GCD_APPLY_CSWAP_TRUNC_W` (probe only).
- Karatsuba/Solinas qubit cuts: `KARA_Z02_LOWQ`, `KARA_SOL_MOD_VENT`,
  `KARA_Z2_SELFHOST`, `ROUND84_XTAIL_BORROW_CARRIES`.
- Fiat-Shamir island: `DIALOG_TAIL_NONCE`, `DIALOG_REROLL`, `DIALOG_POST_SUB_REROLL`.
- Era-1 (legacy 2-Kaliski): `KAL_FREE_S`, `KAL_M_CLASSICAL`, `KAL_ROWWISE_MUL`,
  `KAL_VENT_HALVE/DOUBLE/MODADD`.

## Tooling referenced (for the proxy env / verifier)

- `./benchmark.sh --note '<...>'` — official path; writes `score.json`.
- `build_circuit` / `eval_circuit` — fast probe (TRACE_PEAK, TRACE_PHASES).
- `src/bin/island_search_jac.rs` — full-validation tail-nonce searcher,
  bit-exact self-check vs affine k*G (~7 nonce/s, 11 threads).
- `island_search_prefilter ISLAND_MEASURE_JUMP=1` — classical jump-convergence
  model (no quantum sim).
- `src/point_add/kaliski_classical_replay.rs` / `dialog_gcd_classical_filter`
  — classical GCD replay; the missing piece is a width-envelope-aware,
  both-factor (`dx` AND `c = Qx-Rx`) pre-filter to make island search tractable
  locally.
