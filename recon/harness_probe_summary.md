# Harness probe (local, macOS) — 2026-06-06

Ran `setup.sh` + timed build + full `benchmark.sh` on the freshly-pulled `src/point_add`.

## Headline numbers
- **Full benchmark wall time: ~13 seconds** (incremental rebuild + build_circuit + eval over 9024 shots).
- Clean release build from scratch: ~9.4s.
- Current working-tree circuit is VALID and scores **1,940,606,153**
  (avg Toffoli **1,484,779** × peak qubits **1307**), 0 classical / 0 phase / 0 ancilla failures.
  (This is better than the memory log's 1.967e9 frontier — the pull brought a stronger circuit.)
- build_circuit emits **9,860,456 ops** → `ops.bin` ~552 MB. eval uses 1307 qubits, 889,297 classical bits.

## Why this matters for the model project
1. **Real harness = cheap eval oracle (~13s).** We can validate model-proposed moves directly,
   and run a parallel knob-search / light RL on Modal CPU containers (hundreds–thousands of evals/hr).
   → "Test against ECDSA fail" is a first-class, runnable eval, not a thought experiment.
2. **sim.rs is fast** (9.86M ops in seconds). On *small* reversible circuits it will be ~microseconds
   → perfect to wrap as the dense proxy-env verifier for GRPO (thousands of rollouts/sec on CPU).
3. **Two RL/eval surfaces:**
   - Proxy env (tiny reversible blocks, CPU, dense reward) → teach the general transferable skill.
   - Real DIALOG-config harness (~13s, parallel) → headline eval + optional light RL/search.

## Build/run commands confirmed working locally
- `cd ecdsafail-challenge && ./setup.sh` (idempotent; cargo 1.93 already present)
- `./benchmark.sh --note "..."` → writes `score.json` (offline, sandbox-exec on macOS)
- Knobs are env vars (e.g. `DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS`, `DIALOG_TAIL_NONCE`) — pass before benchmark.sh.
