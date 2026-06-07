# ecdsa-coder — open-source reversible-circuit-optimization specialist

A small, **Apache-2.0** model fine-tuned (SFT + GRPO RL) for the *kind of work* the
[ECDSA.fail](https://ecdsa.fail) secp256k1 point-addition challenge demands: **verifier-guided,
cost-minimizing optimization of reversible circuits under hard correctness constraints**
(cost = average Toffoli count × peak qubit width; lower is better).

Built end-to-end and reproducibly: data mined from the challenge → LoRA SFT → GRPO against a
bit-faithful verifier → merged → served in **Ollama** → evaluated against the real challenge.

## Models
| name | base | license | status |
|---|---|---|---|
| **`ecdsa-coder-1.5b-sft`** ← SHIPPED | Qwen2.5-Coder-1.5B-Instruct | Apache-2.0 | ✅ trained, served, evaluated — the recommended model |
| `gemma4-e4b-sft` | gemma-4-E4B-it | Apache-2.0 | ✅ trained (cross-family transfer); format less crisp than Qwen |
| `ecdsa-coder-1.5b` (SFT+GRPO) | ↑ + GRPO RL | Apache-2.0 | ⚠️ RL run collapsed (format-reward hacking) — NOT recommended; see EVAL_REPORT |

**Use `ecdsa-coder-1.5b-sft`.** The GRPO arm is kept only as a documented negative result + fix path.

## Quickstart
```bash
ollama create ecdsa-coder-1.5b-sft -f artifacts/Modelfile.sft   # import the merged model
ollama run ecdsa-coder-1.5b-sft
```
Then give it a task spec (emit a reversible op-stream) or a current `DIALOG_*` config (propose the
next bounded optimization move). See `proxy/sample_task.txt` for the op-stream task format.

## What it does (and the proof)
1. **Emits valid reversible-circuit op-streams** in the harness DSL. Base Qwen-Coder writes Python;
   the trained model emits clean op-streams (format reward 0.98).
2. **Proposes bounded `DIALOG_*` optimization moves** on the real secp256k1 circuit and always
   remembers the Fiat-Shamir nonce-island re-validation step.
3. **Audits candidates** in the smallest-bounded-change style and classifies validation failures.

**Headline eval — "test against ECDSA fail" (T-CFG, 16 held-out historical accepted moves):**

| metric | base | trained (SFT) |
|---|---|---|
| names a real DIALOG knob | 0.44 | **1.00** |
| matches historical move direction | 0.00 | **0.625** |
| cites nonce-island re-validation | 0.13 | **1.00** |

**GRPO RL note:** the RL run raised the *training* reward but via format-reward hacking, and the
resulting model collapsed — so the **SFT model is shipped**. This is documented honestly (with the
reward-design fix path) in `eval/EVAL_REPORT.md`. **Gemma-4 E4B** (Apache-2.0) also trained with the
same recipe (cross-family transfer), after solving two 2026-tooling blockers (see EVAL_REPORT).

## Reproduce (Modal, well under the $500 budget)
```bash
python3 data/build_sft.py                                              # build the SFT dataset
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage sft   --run-name qwen-sft-v1 \
    --dataset-path /root/data/sft_train.jsonl
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage grpo  --base-model /artifacts/qwen-sft-v1-merged \
    --run-name qwen-grpo-v1 --max-steps 150
ECDSA_BUILD_HEAVY=1 modal run train/merge_lora.py --base-model /artifacts/qwen-sft-v1-merged \
    --adapter /ckpts/grpo/qwen-grpo-v1 --out-name qwen-grpo-v1-merged
modal volume get ecdsa-artifacts <merged> ./artifacts/...  &&  ollama create ... -f Modelfile
python3 eval/eval_proxy.py --model <m>     # held-out reversible-synthesis tasks
python3 eval/eval_cfg.py   --model <m>     # T-CFG move quality on the real challenge
```

## Repository layout
- `proxy/` — `proxy_env.py` (verifier+reward, **800/800 bit-identical** to the real Rust sim),
  `tasks.py` (curriculum + held-out generalization), `proxy_rs/` (Rust ground truth).
- `data/` — `build_sft.py`, `sft_*.jsonl`, `moves_raw.jsonl`, `DATASET.md`.
- `train/` — `modal_app.py` (sft/grpo/export), `merge_lora.py`, `sft_gemma_standalone.py`, `README.md`.
- `eval/` — `eval_proxy.py`, `eval_cfg.py`, `run_harness.py` (real harness wrapper), `EVAL_REPORT.md`, `results/`.
- `artifacts/` — merged weights, `Modelfile.sft`, `MODEL_CARD.md`.
- `recon/` — full reconnaissance (harness semantics, knobs, reasoning episodes, proxy design, training stack).

## Publishing to Hugging Face (requires your HF account/token)
The merged weights + `MODEL_CARD.md` + `data/DATASET.md` are ready. To publish:
```bash
huggingface-cli login    # your token
huggingface-cli upload <your-org>/ecdsa-coder-1.5b artifacts/sft_merged .
# (copy MODEL_CARD.md to README.md in the repo)
```
(Left as a user action since it pushes to your namespace.)

## License
Apache-2.0 (base Qwen2.5-Coder-1.5B is Apache-2.0; the LoRA fine-tune adds no restriction).
