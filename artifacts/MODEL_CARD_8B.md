---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
- reversible-circuits
- tool-use
- quantum
- ecdsa-fail
- expert-iteration
language:
- en
library_name: transformers
---

# reversible-circuit-8b-tool — tool-driven reversible-circuit synthesis (Qwen3-8B)

A small open model fine-tuned to **drive a verifier-backed tool, gate by gate, to synthesize
reversible circuits** for GF(2) linear maps — a faithful proxy for the kind of work the
[ECDSA.fail](https://ecdsa.fail) secp256k1 point-addition challenge demands.

- **Base:** `Qwen/Qwen3-8B` (Apache-2.0) · **License:** Apache-2.0 · **Method:** LoRA SFT (Unsloth/TRL on Modal)
- **Full writeup (read this):** [`docs/PROCESS_LOG.md`](https://github.com/dennisonbertram/reversible-circuit-llm/blob/main/docs/PROCESS_LOG.md) · [`docs/WRITEUP.md`](https://github.com/dennisonbertram/reversible-circuit-llm/blob/main/docs/WRITEUP.md)

## What it does
Given a GF(2) linear-map target on *n* bits, it drives a state-externalizing tool (`ToolEnv`) one
op per turn (`CX`, `CCX`/Toffoli, `SWAP`), reacting to the residual shown after each gate, until a
simulator (bit-for-bit identical to the reference) confirms the circuit is correct.

## Honest evaluation (held-out, 40 tasks/band, best-of-5)
| Band | n | solve rate |
|------|---|-----------|
| B1 | 3 | 95% |
| B2 | 4 | 92.5% |
| B3 | 5 | 40% |
| B4 | 6 | 5% |
| **Overall** | | **~58%** |

Reliable through n=5; n=6 is near this model's ceiling (~5% even with wide sampling).

## What we learned (and what did NOT work — stated plainly)
- **The tool removes the real bottleneck.** Without it, a 1.5B and a 7B model one-shot-synthesize
  *identically* (~4.8%) — the limiter is symbolic execution, not capacity. With the tool, **scale
  then matters** (a trained 1.5B caps at n=4; this 8B reaches n=5).
- **A self-harvest "flywheel" (expert iteration on the model's own verified solutions) did NOT
  improve held-out capability** — a clean negative result. base ≈ iter-1 ≈ iter-2 (~58% best-of-5).
  An earlier apparent "n=6 cracked 0→7.5%" was a **best-of-2 sampling artifact** (this base already
  solves n=6 at ~5% with enough attempts). SFT on a model's own correct outputs re-teaches what it
  already does; it cannot push the frontier.
- **Measurement discipline was the real lesson:** under-sampled evals manufactured two phantom
  "wins" that an adequately-sampled, fixed held-out set erased.

This checkpoint is the **SFT base** (the strongest model in the study). The flywheel iterations did
not beat it, so the base is what's shipped.

## Intended use & limitations
A research artifact / proposer for reversible-circuit synthesis on the proxy task — not an
end-to-end solver for the full 256-bit secp256k1 circuit, and not a general chat model. Use the
base Qwen3-8B for general tasks.

## Reproduce
Code, data factories, eval harness, and the complete process log:
<https://github.com/dennisonbertram/reversible-circuit-llm>
