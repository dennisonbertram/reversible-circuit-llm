# ECDSA-fail Specialist — SFT Dataset

Built deterministically by `build_sft.py` (no LLM generation). Chat format: each line is `{"messages":[{system},{user},{assistant}]}`.

- Total unique examples: **580** (train 523 / val 57)
- Sources:
  - `proxy_synth`: 245
  - `moves`: 290
  - `reasoning`: 45

## Sources
1. **proxy_synth** — (task prompt -> valid reference op-stream) from `proxy/tasks.py` curriculum (bands B0-B6, 7 families). Teaches the harness op-stream DSL + reversible synthesis. Op-streams capped at 140 lines.
2. **moves** — (current DIALOG knob -> next bounded move + validation plan) from the 146 small high-signal accepted-submission diffs in `data/moves_raw.jsonl` (dummy/trace knobs filtered). Teaches the (tighten knob)+(re-find Fiat-Shamir nonce) atomic move.
3. **reasoning** — (situation -> Tony/Anton smallest-bounded-change audit) from the 27 episodes + 5 patterns in `recon/reasoning_episodes.md`. Teaches the transferable method.

Shared system prompt establishes the reversible-circuit-optimization specialist persona and the validity-first cost metric (Toffoli x peak width).
