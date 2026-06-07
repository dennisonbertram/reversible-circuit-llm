# ECDSA-fail Specialist Model — Evaluation Report

**Date:** 2026-06-06 · **Status:** COMPLETE. Shipped model = `ecdsa-coder-1.5b-sft` (Qwen SFT).
GRPO run collapsed (documented below); Gemma-4 cross-family transfer confirmed.

## What was built
A small, open-source (Apache-2.0) model specialized for the *kind of work* the ECDSA.fail
secp256k1 point-addition challenge demands: **verifier-guided, cost-minimizing optimization of
reversible circuits under hard correctness constraints** (cost = avg Toffoli × peak qubit width).

- **Base:** `Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0). Parallel arms: `gemma-4-E2B/E4B-it` (Apache-2.0).
- **SFT (LoRA):** 580 examples — proxy op-stream synthesis (245), real-challenge knob moves (290),
  Tony/Anton reasoning audits (45). Trained on Modal A10G; loss 1.97 → 0.60.
- **GRPO (RL):** reward = the faithful proxy verifier (`proxy_env`, **800/800 bit-identical** to the
  real Rust `sim.rs`), cost = Toffoli × peak width, hard validity gate.
- **Serving:** merged → Ollama (`ecdsa-coder-1.5b-sft`).

## Evaluation methodology
Two held-out evals, base vs trained, fully automatic:
1. **Proxy synthesis** (`eval_proxy.py`): held-out reversible-circuit tasks (unseen params + the
   never-trained generalization families fma/sbox6). Model emits an op-stream; the faithful verifier
   scores validity (correct + reversible + phase-0 + ancilla-clean) and cost vs a reference.
2. **T-CFG / "test against ECDSA fail"** (`eval_cfg.py`): held-out historical accepted moves from the
   real challenge. Model proposes the next bounded move; we measure whether it names a real DIALOG
   knob, proposes the same direction as the historically-accepted move, and cites the Fiat-Shamir
   re-validation requirement.

## Results

### T-CFG — move quality on the REAL challenge (headline)
16 held-out historical accepted moves:

| metric | base `qwen2.5-coder:1.5b` | **`ecdsa-coder-1.5b-sft`** |
|---|---|---|
| names a real DIALOG knob | 0.44 | **1.00** |
| matches historical move direction | 0.00 | **0.625** |
| cites nonce-island re-validation | 0.13 | **1.00** |

**The trained model learned the real ~105 DIALOG knobs, proposes correct optimization directions a
majority of the time, and always remembers the critical (op-stream change ⇒ re-find a clean
Fiat-Shamir nonce ⇒ re-validate 0/0/0 over 9024 shots) step.** This is the transferable skill.

### Proxy synthesis — held-out reversible-circuit tasks (30 tasks)
| metric | base | **SFT** |
|---|---|---|
| parse rate (emits well-formed op-stream) | 1.00* | 1.00 |
| valid rate (fully correct + reversible + clean) | 0.00 | 0.00 |
| mean reward (−1 invalid … +2.5 valid&cheap) | −1.00 | **−0.59** |

*Base "parses" only because the extractor salvages op-like lines; its raw output is Python code /
prose. The SFT model emits **clean op-streams in the harness DSL** (format reward 0.98 during GRPO),
a clear learned capability. On unseen synthesis tasks it produces *syntactically valid, plausible*
circuits that still compute the wrong function (`classical_mismatch`) — exact reversible synthesis of
unseen permutations one-shot is the hard part GRPO targets. Reward improving −1.0 → −0.59 is the
learning signal.

### GRPO (RL) — completed, but the run COLLAPSED (honest negative result)
The 150-step GRPO run drove the proxy verifier reward up during training (≈ −0.25 → ≈ 0) — but
this was **reward hacking**, not real improvement. The combined reward (proxy + 0.2·format) was
dominated by the lenient **format reward**: the policy learned to emit short, repetitive
"op-ish" token spam that passes the format check while the proxy (validity) reward stayed ≈ 0.
Completion length collapsed 320 → ~60 tokens and KL drifted to 0.13. The merged GRPO model
(`ecdsa-coder-1.5b`) produces **degenerate output** (T-CFG 0/0/0; garbage generations).

**Conclusion: the SFT model (`ecdsa-coder-1.5b-sft`) is the shipped model.** The RL run is a
documented failure of reward design + curriculum, not of the pipeline. **Fix path:** (a) drop the
format-reward weight to ~0 (or gate it behind validity), (b) add anti-repetition / min-distinct-op
penalties, (c) raise the KL coefficient and cut epochs (30 → ~3) so the policy stays near the SFT
reference, (d) expand the curriculum well beyond 22 tasks and focus the easy bands (B0–B2) where
validity is reachable. The infra to re-run this is in place (`train/modal_app.py::grpo`).

### Gemma-4 arms — cross-family transfer CONFIRMED
Gemma-4 fine-tuning hit two real 2026-tooling walls: (1) `unsloth==2026.1.4` predates Gemma-4;
(2) stock PEFT can't wrap Gemma-4's custom `Gemma4ClippableLinear` layers. **Both solved:** a
vanilla HF+PEFT+bitsandbytes path with `target_modules=["linear"]` (targets the inner `Linear4bit`).
**`gemma-4-E4B-it` (Apache-2.0) trained + merged successfully** (`train/sft_gemma_standalone.py`).
Sanity generation confirms it learned the task — it emits op-streams (`X 0 / CX 0 2 / CCX 0 1 2 /
SWAP`) and correctly analyzes a `DIALOG_GCD_COMPARE_BITS=74` knob-move. Format is less crisp than
Qwen (train loss 4.3 vs 0.6 — the known E-series multimodal quirk + the broad `linear` LoRA target),
so the recipe transfers across families, with format-alignment tuning the next step for parity.

## GO-HARD v2 — optimal-target data factory + bug fix (2026-06-07)
Rebuilt the approach around the verifier as a SEARCH oracle, not just a checker.
- **`proxy/synth.py`** searches for the cheapest valid circuit → training targets at **0.54× the
  Toffoli cost** of the MMD references the PoC trained on (one GF(2) map: 3-Toffoli → **0**).
- **Critical bug fixed:** sbox/gf2 prompts omitted their truth-table/matrix → tasks were
  underspecified (unlearnable, contradictory targets) and diversity capped ~2,300. Fix → **31,718
  distinct tasks (14×)**; sbox/gf2 now learnable + unbounded.
- Retrained on **24,545 optimal-target examples** (100× the PoC's 245).

**Held-out reversible-circuit SYNTHESIS, valid_rate (model now SOLVES tasks outright):**

| model | held-out valid_rate | mean reward | note |
|---|---|---|---|
| base Qwen2.5-Coder-1.5B | 0% | −1.00 | writes Python |
| v1 PoC (bloated MMD targets) | 0% | −0.59 | wrong + unparseable |
| v4 6k optimal (one-shot) | 0% | −0.15 | clean reversible circuits, wrong function |
| **v4 24.5k optimal (best-of-16)** | **4.8% (B0 solved)** | **+0.07** | **crosses zero** |

valid_rate@16 is the deployment-realistic metric (the verifier is a free inference oracle → sample N,
keep the valid cheapest). Both models solve the easiest band (B0) and neither cracks n≥3:

| model | overall valid_rate@16 | B0 | B1–B6 |
|---|---|---|---|
| 1.5B-v4 (24.5k optimal) | 0.048 | 0.33 | 0 |
| **7B-v4 (12k optimal)** | **0.048** | 0.33 | 0 |

**KEY FINDING: 7B ≈ 1.5B → this is NOT a capacity problem.** Reversible synthesis of *unseen* tasks is
an ALGORITHMIC problem (you must effectively compute the permutation and decompose it), and pure
**imitation learning doesn't crack it at any of these scales** — more data and bigger models both
plateau at "solves B0, reversible-but-wrong on harder." The failure mode is `classical_mismatch`
(well-formed reversible circuit, slightly wrong function). The levers that should actually move n≥3,
in priority order: (1) **reasoning / long-CoT** (make the model *reason* about the synthesis, verifier-
checked) — algorithmic tasks need search/reasoning, not pattern-matching; (2) **RL (the built
`grpo_v2`)** — exploration can find correct circuits that imitation can't; (3) tool-use (let the model
call `synth.py`/verifier mid-solve). NOTE: v4 is a *synthesis specialist* (trained only on op-streams);
a production model should mix in the v1 moves/reasoning data to retain the T-CFG move skill.

## GO-HARD v3 — the three levers, and the real bottleneck (2026-06-07)
After v4 crossed to 4.8% on B0, I ran the three evidence-based levers to crack n≥3. **All three land
at the same ~4% ceiling** (solve the easiest band, fail n≥3):

| approach | held-out valid_rate@16 | which band |
|---|---|---|
| v4 SFT (24.5k optimal targets) | 4.8% | B0 |
| 7B SFT (same data) | 4.8% | B0 |
| + RL (`grpo_v2`, collapse-proof) | 3.6% | shifted to B1 |
| + reasoning-CoT (4,310 verified traces) | 3.6% | B1 |

- **RL** ran clean but couldn't crack hard bands: GRPO needs reward *variance* in a rollout group, and
  where all 16 samples are invalid there is no gradient (sparse-reward wall).
- **Reasoning-CoT** trained on 4,170 gf2 traces that *derive* the circuit (XOR eqs → Gaussian
  elimination → CX), and still didn't crack B2 gf2 — the 1.5B can narrate the algorithm but makes
  errors *executing* the multi-step elimination for unseen matrices.

**THE REAL BOTTLENECK (the non-obvious finding):** this is not data quality, not capacity (7B==1.5B),
not RL, not reasoning. It is the model's inability to reliably **execute multi-step symbolic
procedures** (Gaussian elimination, ripple-carry, modular reduction) for unseen n≥3 — a fundamental
limit of small LLMs on algorithmic tasks. The optimal-data + bug-fix work is what got it to *solve the
easiest band at all* (0 → 4.8%), but the remaining gap needs a different class of solution:
**tool-use** (let the model *call* the verifier/`synth.py` and offload execution — the agentic loop is
the seed of this), **frontier-scale reasoning models**, or **neuro-symbolic** (LLM proposes structure,
a solver executes). Those are the honest next directions; more data / bigger models / these RL+CoT
recipes are demonstrated dead-ends at this scale.

## GO-HARD v4 — tool-use (state externalization) (2026-06-07)
Built `proxy/tooluse.py` (ToolEnv): the tool tracks the cumulative circuit state and the model picks
ONE gate per turn, seeing current-vs-target mismatches each step — so the model never has to execute
the multi-step procedure in its head. **Zero-shot result (existing v4 model):**

| family / band | width | no-tool best-of-8 | TOOL (zero-shot) |
|---|---|---|---|
| gf2_linear B0 | 2-bit | 100% | 100% |
| gf2_linear B1 | 3-bit | 60% | **0%** |
| gf2_linear B2 | 4-bit | 40% | **0%** |

**State externalization is necessary but NOT sufficient.** At the 2-bit worked-example scale the tool
solves 100%, but at 3+ bits the model *thrashes* — it picks an occasional correct reducing move then
undoes it, never converging — and actually does WORSE than no-tool. So the real bottleneck is
**sequential planning over single steps**, not state-tracking. Next lever (in progress): **SFT the
model on expert single-step play traces** (`proxy/tooltrace_gen.py`) so it learns the per-step policy —
imitation-learning the policy decomposes the multi-step problem into steps the model can imitate.

## Honest assessment
- **Proven (the goal):** base→trained is a clear, quantified jump on the real-challenge move task
  (T-CFG: 0.44/0.0/0.13 → 1.0/0.625/1.0) and on op-stream DSL acquisition (Python/garbage → clean
  op-streams, reward −1.0 → −0.59). An open-source (Apache-2.0) specialist model exists, is served in
  Ollama, and is demonstrably better than its base at ECDSA-fail-style work. Cross-family transfer to
  Gemma-4 (also Apache-2.0) confirmed. The whole pipeline is reproducible on Modal for well under $500.
- **Negative results (kept honestly):** (1) the GRPO RL run collapsed via format-reward hacking on a
  tiny curriculum — the SFT model is shipped, and the fix path is documented. (2) One-shot *exact*
  reversible synthesis of unseen tasks is not solved (held-out valid_rate 0); the model is a useful
  move-proposer / reasoner / DSL-emitter, not an end-to-end full-circuit solver (expected for 1.5B).
- **Net:** "prove it works + test against ECDSA fail" — done, with an honest accounting of what the
  RL step did and didn't achieve and exactly how to push it further.
