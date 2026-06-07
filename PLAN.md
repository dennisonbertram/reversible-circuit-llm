# ECDSA-Fail Specialist Model — Plan

**Goal (user, 2026-06-06):** Train an open-source *small* model (served in Ollama) on the
successful runs of the ECDSA-fail challenge, specialized for "this kind of solution work"
and intended to generalize the skill to other problems. Need not be SOTA — prove it works,
then test against ECDSA fail. Budget: $500 on Modal (creds present). Work autonomously,
pursue multiple hypotheses in parallel.

## What the challenge actually is
Build the cheapest **reversible quantum circuit** for one secp256k1 point-addition.
Score = `avg_Toffoli × peak_qubits` (lower better), validated by 9024-shot simulation with
hard **correctness + reversibility + phase-cleanliness + forward∘reverse-identity** checks.
- Current local best (`score.json`): **2.478e9** (Toffoli 1,728,069 × qubits 1434).
- Promoted frontier (memory log): ~**1.967e9** — already below Google's published Pareto (3.0e9).
- Editable surface: `src/point_add/` only. Circuit built by a parametric **DIALOG** system
  with env-var knobs (`DIALOG_GCD_APPLY_CLEAN_COMPARE_BITS`, `DIALOG_TAIL_NONCE`, …).
- Harness: `build_circuit` (untrusted → `ops.bin`) → `eval_circuit` (trusted, `sim.rs` → `score.json`).
  Runs on macOS; final eval can run locally.

## The transferable skill we are training
**Verifier-guided, cost-minimizing optimization of a structured artifact (reversible circuit /
op-stream) under hard correctness constraints**, done via evidence-cited *smallest-bounded-change*
audits (the "Tony/Anton" loop). This is exactly SFT-able (reasoning traces exist) and RL-able
(the environment has a clean pass/fail × cost reward).

## Parallel hypotheses (tracks)
- **T-SFT (primary):** distill the audit-loop reasoning + git "optimization moves" + algorithm
  docs into (instruction → reasoning → answer) traces; LoRA-SFT a small code model. Generalizable
  because it teaches the *method*, not a memorized circuit.
- **T-RL:** GRPO on top of SFT against a **fast proxy verifier** — small reversible-arithmetic /
  boolean straight-line program optimization sharing the Toffoli×width cost metric (the full
  harness is too slow per-rollout). Proves the model *learned the skill*.
- **T-CFG (headline eval):** the model proposes DIALOG knob/code moves; validate with the **real**
  harness — does it produce valid, score-improving submissions vs. baseline?
- **Base models as parallel hypotheses (ALL Apache-2.0, ALL Ollama-native, ALL Unsloth LoRA+GRPO):**
  - **Arm A — Qwen2.5-Coder-1.5B-Instruct**: smallest/fastest, strongest small *code* model, vLLM-colocate GRPO.
  - **Arm B — Gemma 4 E2B-it** (2.3B eff): fast Google arm, lowest VRAM (RL ~9GB).
  - **Arm C — Gemma 4 E4B-it** (4.5B eff): stronger Google arm; replaces Llama-3.2-3B (better license + newer).
  - Gemma 4 license = Apache-2.0 (vs Gemma 3's custom terms / Llama's community license). Verify against
    ai.google.dev/gemma/terms before shipping the release notice.
  - CAVEAT: Gemma 4 E-series is NOT vLLM-supported → GRPO must use `fast_inference=False` (slower rollouts,
    no colocate). SFT unaffected. E-series SFT loss ~13–15 is a normal multimodal quirk, not a bug.

## Phases (see task list)
1. Deep recon (6 facets) → `recon/`.
2. SFT dataset → `data/`.  3. Proxy RL env + verifier.  4. Modal infra + base model(s).
5. LoRA SFT.  6. GRPO RL.  7. GGUF + Ollama + cards.  8. Eval vs base + real-harness test + release.

## Budget discipline
Start cheapest viable (1.5B LoRA on A10G/L40S, short runs); scale only if PoC needs it.
Log all Modal spend. Target total well under $500.

## Layout
`recon/` findings · `data/` datasets · `train/` Modal training apps · `eval/` harness+eval ·
`infra/` env/probe logs · `artifacts/` weights, GGUF, Modelfile, cards.
