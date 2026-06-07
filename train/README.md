# `train/` — Modal training app for the ECDSA-fail coder model

`modal_app.py` is a single `modal.App("ecdsa-coder")` implementing the
**SFT → GRPO → GGUF-export** pipeline from
[`recon/training_stack.md`](../recon/training_stack.md). It trains a small open-source
code model to emit cost-minimal reversible op-streams, scored by the fast Python proxy
verifier in [`../proxy/`](../proxy).

- **Primary base:** `unsloth/Qwen2.5-Coder-1.5B-Instruct` — Apache-2.0, **ungated** (no HF
  token needed), best code-per-param at 1.5B, fastest GRPO rollouts.
- **Secondary base:** `unsloth/Llama-3.2-3B-Instruct` — gated (needs an `HF_TOKEN`
  secret), different family/tokenizer as a hedge. Optional.

---

## Prerequisites

- Modal installed and authenticated (profile `dennisonbertram`): `modal --version`.
- The `modal` CLI on PATH (installed via `uv tool`).
- Volumes are auto-created on first use (`create_if_missing=True`):
  - `hf-cache`        → `/root/.cache/huggingface` (base weights / HF cache)
  - `ecdsa-ckpts`     → `/ckpts` (SFT + GRPO LoRA adapters)
  - `ecdsa-artifacts` → `/artifacts` (merged fp16 + GGUF)
- **No HF token required for Qwen.** For the gated **Llama** secondary base only, create a
  Modal secret named `huggingface` with `HF_TOKEN`:
  ```bash
  modal secret create huggingface HF_TOKEN=hf_xxx
  ```
  The app auto-detects this secret at load time and attaches it only if it exists; Qwen
  runs fine without it.

---

## The `ECDSA_BUILD_HEAVY` switch (important)

`modal run` resolves the **whole app** and eagerly builds **every** image attached to any
function — including the multi-GB CUDA + Unsloth + vLLM training image. To keep the
**smoke** check cheap, the heavy image is only constructed when `ECDSA_BUILD_HEAVY=1` is
set in the environment. Without it, the heavy functions fall back to the light image and
**refuse to run** (a guard raises a clear "relaunch with ECDSA_BUILD_HEAVY=1" error before
any GPU spins up).

- **Smoke / light path:** do **not** set the var → no heavy build, no GPU spend.
- **Any real training stage (sft/grpo/export):** **must** prefix `ECDSA_BUILD_HEAVY=1`.

---

## Stages

### 0. `smoke` — validate Modal plumbing (CPU, ~free)

Builds **only** the `python:3.12-slim` light image and runs a CPU container that prints the
platform and confirms the Modal wiring works. Imports no torch; triggers no heavy build.

```bash
modal run train/modal_app.py::smoke
```

Expected output (verified):

```
[smoke] Modal plumbing OK — light image container ran ecdsa-coder.smoke()
[smoke] python  : 3.12.13
[smoke] platform: Linux-...-x86_64-with-glibc2.41
[smoke] machine : x86_64
```

### 1. `sft` — LoRA SFT (Unsloth + TRL `SFTTrainer`)

Loads the base 4-bit (QLoRA), attaches an all-linear LoRA (`r=16`, `alpha=32`,
`dropout=0.05`), trains on a ChatML JSONL with the **prompt masked** (train on the
assistant response only), `max_seq_len=4096`, `lr=2e-4` cosine, batch `4 × ga 4`, 2 epochs.
Saves the LoRA + a merged 16-bit model.

```bash
# dataset_path can be a file inside a volume (e.g. /artifacts/sft.jsonl) or the
# image-mounted /root/data/sft.jsonl (the local ../data dir is baked into the heavy image).
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py::sft \
    --run-name qwen-sft-v1 \
    --dataset-path /root/data/sft.jsonl \
    --epochs 2
```

JSONL row format (either is accepted):
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "completion": "..."}
```

Outputs:
- `/ckpts/sft/<run_name>/` — LoRA adapter (+ tokenizer)
- `/artifacts/<run_name>-merged/` — merged 16-bit model
- `/artifacts/<run_name>-lora/` — adapter copy

### 2. `grpo` — GRPO RL on the proxy verifier (Unsloth + TRL `GRPOTrainer`)

Continues the SFT LoRA, runs single-turn GRPO with **colocated vLLM** rollouts
(`fast_inference=True`), `num_generations=8`, `lr=1e-5`, `max_steps` configurable. The
GRPO dataset is the rendered proxy curriculum (`tasks.curriculum()`); each rollout is
scored by `proxy_env.reward` (Toffoli × peak-width verifier, validity-gated reward
shaping) plus a light `format_reward`. The proxy package is baked into the image at
`/root/proxy`, so `import proxy_env` / `import tasks` work inside the container.

```bash
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py::grpo \
    --run-name qwen-grpo-v1 \
    --sft-path /ckpts/sft/qwen-sft-v1 \
    --max-steps 300
```

Outputs:
- `/ckpts/grpo/<run_name>/` — GRPO LoRA adapter
- `/artifacts/<run_name>-grpo-lora/` — adapter copy

Single-GPU GRPO env hardening (set automatically): `UNSLOTH_VLLM_STANDBY=1`,
`NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_ENABLE=1`, `gpu_memory_utilization≈0.55`.

### 3. `export_gguf` — merge LoRA → fp16 → GGUF (q4_k_m)

Merges the GRPO (or SFT) LoRA into the base and runs Unsloth's one-call
`save_pretrained_gguf(..., quantization_method="q4_k_m")` (llama.cpp under the hood).

```bash
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py::export-gguf \
    --run-name ecdsa-coder-1.5b \
    --ckpt-path /ckpts/grpo/qwen-grpo-v1 \
    --quant q4_k_m
```

Output: `/artifacts/<run_name>/*.gguf`. Download it locally:
```bash
modal volume get ecdsa-artifacts ecdsa-coder-1.5b ./ecdsa-coder-1.5b
```

Then package for Ollama **locally** (final eval is local — Ollama need not run on Modal),
using the Qwen ChatML `Modelfile` from `training_stack.md` §5:
```bash
ollama create ecdsa-coder-1.5b -f Modelfile
ollama run  ecdsa-coder-1.5b
```

> Quant note: validate the q4_k_m GGUF against the HF/vLLM checkpoint; small fine-tunes can
> be quant-sensitive — bump to `q5_k_m`/`q6_k` if outputs degrade (`--quant q5_k_m`).

---

## Dispatch via the local entrypoint

All stages are also reachable through `main(stage=...)`:

```bash
modal run train/modal_app.py                                            # smoke (default)
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage sft    --run-name qwen-sft-v1
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage grpo   --run-name qwen-grpo-v1
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage export --run-name ecdsa-coder-1.5b
```

---

## Secondary base (Llama-3.2-3B)

Pass `--base-model unsloth/Llama-3.2-3B-Instruct` to any stage. This base is **gated** —
create the `huggingface` secret first (see Prerequisites). For comfortable colocated GRPO
on 3B, switch the GPU to `L40S` (48 GB) by editing the `gpu=` kwarg on `grpo` (A10G is fine
for 3B SFT). Note the Llama 3.2 Community License redistribution obligations
("Built with Llama") — keep the Apache-2.0 Qwen as the shipped artifact.

---

## Expected cost (per `training_stack.md` §4, A10G ≈ $1.10/hr)

| Stage | GPU | Rough wall-clock | Rough $ |
|---|---|---|---|
| `smoke` | CPU | seconds | ~$0 (light image only) |
| Heavy image build (first time) | CPU builder | ~10–20 min | a few $ of build time, cached after |
| `sft` (1.5B, 1–2 epochs, few-k ex) | A10G | ~15–40 min/epoch | **$2–6** incl. a re-run |
| `grpo` (few hundred steps, num_gen=8, colocated vLLM) | A10G | a few hours | **$15–40** |
| `export_gguf` | A10G | ~10–30 min | **<$1** |
| **Full PoC** (both models + a couple sweeps + eval + GGUF) | A10G/L40S | — | **~$80–150** |

Comfortably under the **$500** budget. Discipline: start A10G + 1.5B + short runs; log each
run's wall-clock × rate; scale GPU or steps only when the PoC needs it. 3B (L40S) ≈ 2× the
1.5B numbers.

---

## Version pinning (the version-sensitive surface)

The GRPO + vLLM + LoRA path is the most fragile part of the stack. The heavy image pins
(CUDA 12.9 base, `torch==2.8.0+cu129`, `unsloth==2026.1.4`, `unsloth_zoo==2026.1.4`,
`trl==0.23.0`, `peft==0.18.0`, `transformers==4.57.1`, `vllm==0.11.0`, `triton==3.4.0`,
`bitsandbytes==0.48.1`, `datasets==3.6.0`, `accelerate==1.10.1`) all sit inside the
dependency windows declared by `unsloth==2026.1.4`. If a future Unsloth/vLLM release breaks
the build, refresh these from the Unsloth release notes or use their
`--force-reinstall --no-deps unsloth unsloth_zoo` resolver, and re-pin.

## Files

- `modal_app.py` — the Modal app (this directory).
- `../proxy/proxy_env.py` — exact-semantics reversible-circuit verifier + GRPO reward.
- `../proxy/tasks.py` — proxy curriculum + ChatML prompt rendering + `build_dataset()`.
- `../data/` — SFT data (mounted into the heavy image at `/root/data`).
