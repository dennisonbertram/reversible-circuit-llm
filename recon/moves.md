# Accepted Optimization MOVES — recon summary

Built from the git history of `ecdsafail-challenge` by
`recon/build_moves.py` (deterministic, `/usr/bin/python3`, stdlib + `git`).
Each commit whose subject starts with `Accept submission`, reachable from
`HEAD`, is one accepted optimization move (its diff vs its first parent).

- Dataset: `ecdsa-model/data/moves_raw.jsonl` (JSONL, **oldest commit first**, one line per commit)
- Builder: `ecdsa-model/recon/build_moves.py`
- Range: oldest `c55911f37d` … newest `2c3b795465` (= current HEAD)

## Per-record schema

```json
{
  "sha": "...",                 // commit sha
  "parent": "...",              // first-parent sha
  "subject": "Accept submission <uuid>",
  "body": "...",                // git log -1 --format=%b (verbatim, trailing newline trimmed)
  "numstat": [[adds, dels, file], ...],   // git show --numstat --format=; binary -> null
  "n_files": N,                 // len(numstat)
  "is_small_move": bool,        // see definition below
  "small_diff": "...|null",     // unified diff of the single file when is_small_move
  "reconstructed_score": {...}|null  // score/toffoli/qubits ints parsed from BODY via regex
}
```

`is_small_move` is true iff, after dropping every changed file that is a
`.md` file or lives under a `memory/` path, exactly **one** file remains, that
file is `src/point_add/mod.rs` or a `dialog/config.rs`, and its changed lines
(`adds + dels`) are `< 120`.

## Stats

| metric | value |
|---|---|
| total accept commits | **264** |
| small high-signal moves | **146** |
| moves touching big arith files (adder / modular / multiply / const_arith) | **15** |
| moves with reconstructable score (from **body**) | **0** |
| distinct authors (Co-authored-by trailers) | **47** |

Note on the brief's "~108": the actual count of `Accept submission` commits
reachable from HEAD is **264**. All 264 are included.

### Score reconstruction caveat

`reconstructed_score` is parsed **from the commit body** as specified. Every
accepted commit's body is just a `Co-authored-by:` trailer (no score / toffoli
/ qubit text), so this field is `null` for all 264 records. The real
score/qubit/Toffoli numbers live in the **diff comments** inside
`src/point_add/mod.rs` (e.g. `1309q x 1,503,355 T = 1,967,891,695`,
`1,512,823 -> 1,506,043 @ 1313`). Those are captured verbatim inside
`small_diff` and can be mined downstream if body-level reconstruction is not
required.

### Big-arith churn caveat

The big arith files (`adder.rs`, `const_arith.rs`, `modular.rs`,
`multiply.rs`) and `rounds/dialog/compressed.rs` frequently show up with
identical `adds == dels` and a `@@ -1,N +1,N @@` whole-file hunk — i.e. the
whole file is rewritten with no net line change (formatting / regeneration
churn that rides along with a real `mod.rs` tweak). Only **15** accept commits
touch those arith files at all; the high-signal optimization almost always
lands in `mod.rs`.

## Authors (Co-authored-by, top of 47)

40 Gajesh2007 · 23 mpjunior92 · 17 newjordan · 16 10d9e · 13 robertkodra ·
12 anupsv · 8 zuiris · 7 PhantasticUniverse · 7 nikhiljha · 7 saucegodbased ·
6 Epistetechnician · 6 aburan28 · 6 josusanmartin · 6 pldallairedemers ·
5 BitWonka · 4 antojoseph · 4 jackylee0424 · 4 johnbxx · 4 runeape-sats ·
4 solimander · 4 welttowelt · … (27 more with 1–3 each)

## Move taxonomy (observed)

Derived empirically from the `set_default_env(...)` knobs that change on the
`+` side of the 146 small-move diffs. Counts are how many small moves flip
that lever (a move may flip several).

1. **tail-nonce reroll** — change `DIALOG_TAIL_NONCE` (re-roll the
   Fiat-Shamir tail nonce to land a clean validating island). 63 moves.
2. **reroll-count tweak** — change `DIALOG_REROLL` / `DIALOG_POST_SUB_REROLL`
   / `DIALOG_GCD_REROLL` (how many reroll attempts before commit). 38 + 1 moves.
3. **tighten compare bits** — lower `DIALOG_GCD_COMPARE_BITS` /
   `DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS` (shrink comparator bit-width to cut
   Toffoli). 37 + 17 moves.
4. **width-envelope tune** — adjust `DIALOG_GCD_WIDTH_SLOPE_X1000` /
   `DIALOG_GCD_WIDTH_MARGIN` (the converged-operand width envelope/slope). 34 + 17 moves.
5. **carry-truncation widen/trim** — `KAL_FOLD_CARRY_TRUNC_W` /
   `KAL_DOUBLE_CARRY_TRUNC_W` / `DIALOG_GCD_BODY_CARRY_BAND_TRIMS` (truncate or
   restore carry bits in fold/double/body steps). 17 + 13 + 1 moves.
6. **active-iteration count** — `DIALOG_GCD_ACTIVE_ITERATIONS` (number of
   Kaliski/GCD active iterations vs nonconvergence pressure). 16 moves.
7. **compare-schedule margin** — `DIALOG_GCD_PA9024_COMPARE_SCHEDULE_MARGIN`
   (per-step comparator schedule slack across the 9024 shots). 12 moves.
8. **apply-path chunk/cut restructure** — `DIALOG_GCD_APPLY_CHUNKED_F_CUT*`,
   `..._F_BLOCKS`, `..._F_CUSTOM*`, `..._BOUNDARY_SPLIT`,
   `DIALOG_GCD_APPLY_FINAL_*` (where the apply-phase F register is cut /
   chunked / split for the teardown). ~30 moves combined.
9. **fast-path / fusion toggle** — `DIALOG_GCD_ODD_U_LOWBIT_FASTPATH`,
   `DIALOG_GCD_MEASURED_APPLY_SUB`, round-specific compress/levers
   (`DIALOG_GCD_ROUND763_COMPRESS_LEVER`). small handful.
10. **arith-primitive restructure** — the 15 moves that touch
    `adder/modular/multiply/const_arith` (Cuccaro/MAJ-UMA adder, Solinas
    modular-reduction, Karatsuba/schoolbook multiply, const-arith) plus
    Karatsuba/Solinas levers (`KARA_SOL_*`, `ROUND84_XTAIL_*`,
    `KARA_FREE_Z1_TOPBIT`). Includes "restructure adder", "modular-reduction
    tweak", "change uncompute path".

Cross-cutting buckets requested in the brief map as:
`tighten compare bits` → (3); `tail-nonce reroll` → (1)+(2);
`restructure adder` / `modular-reduction tweak` → (10);
`change uncompute path` → carry-truncation / apply-teardown (5)+(8).

## Verification

- `wc -l data/moves_raw.jsonl` → 264
- Every one of the 264 lines parses as JSON with exactly the 9 schema keys
  (checked with `/usr/bin/python3 -c json`).
- Line 1 = oldest accept commit `c55911f3`; line 264 = newest = HEAD `2c3b7954`.
