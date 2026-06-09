# ecdsa-coder-1.5b — a reversible-circuit-optimization specialist

> A small, open-source model fine-tuned for the *kind of work* the
> [ECDSA.fail](https://ecdsa.fail) secp256k1 point-addition challenge demands:
> **verifier-guided, cost-minimizing optimization of reversible circuits under hard
> correctness constraints.**

- **Base model:** `Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0)
- **License:** Apache-2.0 (base + this fine-tune)
- **Method:** LoRA SFT (+ GRPO RL on a verifier reward) via Unsloth/TRL on Modal
- **Serving:** Ollama (`ollama run ecdsa-coder-1.5b`)
- **Parallel arms:** `gemma-4-E2B/E4B-it` (Apache-2.0) — same recipe, different family

> ## ⚠️ 2026-06 update — read this first
> Work continued past this 1.5B PoC: a **state-externalizing tool** (`ToolEnv`) and a larger **8B**
> base (`Qwen3-8B`), plus a **self-improvement flywheel** (expert iteration on verifier-confirmed
> solutions). Honest headline findings:
> - **The tool removes the real wall.** Without it, 1.5B and 7B one-shot-synthesize *identically*
>   (~4.8%) — the bottleneck is symbolic execution, not capacity. With the tool, trained models
>   solve held-out tasks, and **scale then matters** (1.5B caps at n=4, 8B reaches n=5).
> - **The self-harvest flywheel did NOT improve held-out capability** — a clean negative result.
>   base ≈ iter-1 ≈ iter-2 at ~58% best-of-5. An apparent "n=6 cracked 0→7.5%" was a *best-of-2
>   sampling artifact* (the base already solves n=6 at ~5%). SFT on a model's own correct outputs
>   re-teaches what it already does; it can't push the frontier.
> - **Measurement discipline was the real lesson** — two phantom "wins" died to an
>   adequately-sampled, fixed held-out eval.
>
> Full story (every number + dead-end): **`docs/PROCESS_LOG.md`** and **`docs/WRITEUP.md`** in the
> [GitHub repo](https://github.com/dennisonbertram/reversible-circuit-llm). The evals below are the
> original 1.5B PoC numbers and stand as-is.

## What it does
1. **Emits reversible-circuit op-streams** in the challenge's DSL (`X / CX / CCX(Toffoli) / SWAP /
   CCZ`, `if bM` conditioning; last qubit token = target; cost = executed Toffoli × peak qubit width)
   that satisfy the four validity gates (correct on all inputs, reversible, phase-0, ancilla-clean).
2. **Proposes bounded optimization moves** on the real secp256k1 circuit via its ~105 `DIALOG_*`
   knobs, and *always* remembers the key domain fact: any op-stream-altering change voids the
   Fiat-Shamir nonce island, so a clean `DIALOG_TAIL_NONCE` must be re-found and re-validated
   (0 classical / 0 phase / 0 ancilla over 9024 shots).
3. **Audits candidates** in the "smallest bounded change" style: inspect → cite the exact lever/metric
   → quantify the Toffoli/qubit/phase impact → one bounded fix; classifies validation failures
   (structural value-error vs Fiat-Shamir island vs width floor) and decides reroll-vs-abandon.

## Evaluation (held-out, base vs this model)
**T-CFG — move quality on the real challenge** (16 held-out historical accepted moves):

| metric | base Qwen2.5-Coder-1.5B | **this model (SFT)** |
|---|---|---|
| names a real DIALOG knob | 0.44 | **1.00** |
| matches historical move direction | 0.00 | **0.625** |
| cites nonce-island re-validation | 0.13 | **1.00** |

**Proxy reversible-circuit synthesis** (30 held-out tasks): mean reward −1.00 (base) → **−0.59**
(this model); the base emits Python/prose, this model emits clean op-streams (format 0.98).

**GRPO RL:** a GRPO run on the verifier reward was attempted but collapsed (format-reward hacking on a
small curriculum) — so **the shipped model is the SFT checkpoint**, not the GRPO one. See
`eval/EVAL_REPORT.md` for the analysis + fix path. The same SFT recipe also trained a Gemma-4-E4B
arm (Apache-2.0, cross-family transfer).

## Intended use & limitations
- **Use:** an assistant/proposer for reversible-circuit optimization and ECDSA.fail-style knob tuning;
  a generator of small reversible arithmetic/boolean circuits; a reasoning partner for the audit loop.
- **Not:** an end-to-end solver that one-shots a full 1434-qubit secp256k1 circuit (no 1.5B model can;
  the action space is astronomical). On *unseen* exact-synthesis tasks it produces plausible but not
  always correct circuits — this is a proof-of-concept, not SOTA.
- Specialized: general chat quality is unchanged-to-slightly-degraded vs the base; use the base for
  general tasks.

## Training data
580 examples (`data/sft_*.jsonl`): proxy op-stream synthesis from a faithful verifier curriculum
(245), real-challenge knob moves mined from 146 accepted-submission diffs (290), and Tony/Anton
reasoning-audit episodes (45). See `data/DATASET.md`. The reward verifier (`proxy/proxy_env.py`) is
**bit-identical (800/800)** to the real Rust simulator (`proxy_rs/`).

## Reproduce
See `train/README.md` and `eval/EVAL_REPORT.md`. Pipeline: `data/build_sft.py` → Modal `sft` →
`grpo` → merged weights → `ollama create` → `eval/eval_proxy.py` + `eval/eval_cfg.py`.
Total Modal cost for the full PoC: well under the $500 budget.
