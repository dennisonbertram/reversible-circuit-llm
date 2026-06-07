# Training Stack Recommendation — SFT + GRPO + Ollama on Modal (≤$500)

Recon facet: **training stack** (Task #1). Decides the concrete, current (June 2026) toolchain to
SFT then GRPO-RL a small open-source code model, export to GGUF, and serve in Ollama, runnable on
Modal inside the $500 budget. Sources are cited inline; docs pulled via WebSearch + ctx7.

---

## TL;DR (the stack we will use)

| Layer | Choice | Why |
|---|---|---|
| **Base model (primary)** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Best code ability per param at 1.5B; **Apache-2.0** (clean to redistribute the GGUF); first-class GGUF/Ollama support; fastest RL rollouts. |
| **Base model (secondary)** | `meta-llama/Llama-3.2-3B-Instruct` | Different family/tokenizer as a hedge; strong general reasoning; widely Ollama-tested. (License has obligations — see §1.) |
| **SFT** | **Unsloth** `FastLanguageModel` + TRL `SFTTrainer` | 2x faster / ~70% less VRAM LoRA; one library does SFT → GRPO → GGUF export, removing format-mismatch risk. |
| **GRPO RL** | **Unsloth GRPO** (`FastLanguageModel(fast_inference=True)` + TRL `GRPOTrainer`) with a **custom Python `reward_funcs`** | Colocated vLLM rollouts on ONE GPU (fits 1.5B on 24 GB); reward = plain Python list, trivial to plug our proxy verifier. `verifiers` is the fallback if we go multi-turn/agentic. |
| **GPU** | **A10G (24 GB)** for 1.5B; **L40S (48 GB)** for 3B / headroom | Cheapest GPU that fits colocated train+vLLM for 1.5B; L40S only if 1.5B is tight or we run 3B. |
| **Export** | Unsloth `save_pretrained_gguf(..., quantization_method="q4_k_m")` (llama.cpp under the hood) → `ollama create` | Merges LoRA → fp16 → quantizes in one call. Q4_K_M is the sweet spot for 1.5–3B. |
| **Serve** | **Ollama** from a `Modelfile` (`FROM *.gguf` + matching chat TEMPLATE) | Final eval runs locally on macOS; Ollama is the deployment target. |

**Rough budget:** 1.5B LoRA SFT (1–2 epochs, few-k ex) ≈ **$2–6**; short GRPO (few hundred steps)
≈ **$15–40**. Full PoC incl. re-runs, 3B comparison, eval ≈ **$80–150** — comfortably under $500.

---

## 1. Base model: primary + secondary

Candidates: Qwen2.5-Coder-1.5B-Instruct, Qwen2.5-Coder-3B-Instruct, Llama-3.2-3B-Instruct.

**License (decisive for "release the GGUF"):**
- Qwen2.5-Coder **0.5B / 1.5B / 7B / 14B / 32B = Apache-2.0** (true OSS, redistribute freely).
- Qwen2.5-Coder **3B = Qwen-Research license (non-commercial / source-available)** — a real wart for a public release.
- Llama-3.2-3B = **Llama 3.2 Community License** (source-available): must display "Built with Llama",
  prefix derivative names with "Llama", ship the license, AUP applies, 700M-MAU commercial gate.
  Fine for a research PoC, but adds redistribution friction vs Apache-2.0.

**Code ability:** Qwen2.5-Coder is purpose-built for code and is the strongest small local coding
model in its class in 2026 (beats general Llama-3.2 on code at equal/larger size). Our task (editing a
structured op-stream / DIALOG knobs under a cost metric) is code-shaped, so a code-specialized base
gives the SFT a head start.

**RL rollout speed:** 1.5B generates ~2x faster than 3B → roughly 2x more GRPO steps per dollar,
which matters most for the RL phase where we pay for `num_generations` completions per prompt.

**GGUF/Ollama:** Qwen2.5 and Llama-3.2 are both fully supported by `convert_hf_to_gguf.py` and have
canonical Ollama Modelfiles/templates. No architecture-support risk for either.

**Decision:**
- **Primary = Qwen2.5-Coder-1.5B-Instruct** — Apache-2.0, best code/param, fastest rollouts, cheapest. This is the PoC workhorse.
- **Secondary = Llama-3.2-3B-Instruct** — run as a parallel sub-hypothesis (different family/tokenizer, stronger general reasoning) to show the method transfers across model families. Accept its license obligations for the PoC.
- **Explicitly NOT the headline:** Qwen2.5-Coder-3B — the **Qwen-Research (non-commercial) license** undercuts the "open-source release" goal. Keep it only as an internal ablation if we want a same-family size comparison, not as a shipped artifact.

Use the Unsloth pre-quantized repos for training convenience: `unsloth/Qwen2.5-Coder-1.5B-Instruct`,
`unsloth/Llama-3.2-3B-Instruct`.

---

## 2. SFT framework + LoRA hyperparams (~2–6k examples)

**Framework: Unsloth + TRL `SFTTrainer`.** Unsloth patches the model (custom Triton kernels,
padding-free packing, `use_gradient_checkpointing="unsloth"`) for ~2x speed / ~70% less VRAM, then you
hand it to TRL's `SFTTrainer`. Critically, the SAME Unsloth model object later does GRPO and the GGUF
export, so the chat template / EOS token stays consistent end-to-end (the #1 cause of broken exports
per Unsloth docs).

**Why not plain TRL alone?** TRL `SFTTrainer` + PEFT works (and is our reference for API), but Unsloth
is strictly faster/cheaper on a single small GPU and unifies the export path. We use TRL's trainer
*classes* under Unsloth's patched model.

**LoRA hyperparameters for ~2–6k SFT examples** (start here, sweep `r` / lr if under-fitting):

```python
# Model load (Unsloth)
max_seq_length = 4096          # reasoning traces (Tony/Anton audit loop) can be long
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,       # QLoRA; 1.5B fits 16-bit too, 4-bit just buys headroom
    dtype = None,              # auto (bf16 on Ampere+)
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                    # 16 for 2–6k ex; bump to 32 if under-fitting, drop to 8 if overfitting
    lora_alpha = 32,           # alpha = 2*r is a safe default
    lora_dropout = 0.05,
    bias = "none",
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],   # all-linear = best quality
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

```python
# SFTConfig (TRL)
SFTConfig(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,     # effective batch 16
    warmup_ratio = 0.05,
    num_train_epochs = 2,                # 1–2 for few-k ex; watch eval loss for overfit
    learning_rate = 2e-4,                # LoRA wants ~10x full-FT lr (TRL guidance)
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    weight_decay = 0.01,
    bf16 = True,
    max_grad_norm = 1.0,
    logging_steps = 5,
    packing = False,                     # keep off if traces must stay atomic; on to speed up
)
```

Notes: mask the prompt (train on completion/response only) so the model learns the *reasoning+move*,
not the instruction. Use the model's native chat template (Qwen ChatML / Llama-3 headers) — do not
hand-roll. Hold out ~5–10% for an eval split to catch overfitting at 2 epochs.

---

## 3. RL framework for GRPO with a custom programmatic reward

We have a **Python proxy verifier** (Toffoli×width cost on a small reversible/boolean straight-line
program) that returns a scalar. We want: easy plug-in of that reward, fast vLLM rollouts, single-GPU
viability, and clean continuation from the SFT LoRA.

**Option comparison:**

| | Plug a Python reward fn | vLLM rollouts | Single-GPU (1.5B) | Continues SFT LoRA | Verdict |
|---|---|---|---|---|---|
| **Unsloth GRPO** (TRL `GRPOTrainer` on patched model) | Trivial — `reward_funcs=[my_fn]`, fn gets `prompts, completions, **cols`, returns `list[float]` | **Colocated** via `fast_inference=True` (vLLM in-process) | **Yes** — built for it; `UNSLOTH_VLLM_STANDBY=1` for tight VRAM | **Yes** — same object from SFT | **Primary.** Lowest-friction, cheapest, one library. |
| **TRL `GRPOTrainer`** (stock) | Same reward API (`reward_funcs`, list of fns, weighted via `reward_weights`) | `use_vllm=True`, `vllm_mode="colocate"` or `"server"` | Colocate fits; server mode wants a 2nd GPU (2x cost) | Yes (`peft_config`) | Solid fallback; Unsloth wraps this anyway. |
| **`verifiers`** (PrimeIntellect) | Reward = `vf.Rubric` of Python fns inside a `load_environment()`; great for **multi-turn / tool / agentic** | Separate `vf-vllm` server + OpenAI-client trainer | Wants ≥2 GPU (server + trainer); heavier | Via prime-rl / its `RLTrainer` | **Only if** we make the task multi-turn (model calls our verifier as a tool over several turns). Overkill for single-turn propose-a-patch. |

**Decision: Unsloth GRPO (TRL `GRPOTrainer` on the Unsloth-patched model), single-turn, colocated vLLM.**
Our reward is a single-turn "propose op-stream/knobs → score with proxy verifier" — `verifiers`' agentic
machinery and extra GPU are unnecessary. Keep `verifiers` as a documented fallback if a later track
makes the verifier an interactive tool the model queries across turns.

**Custom reward shape (programmatic):**
```python
def proxy_cost_reward(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        prog = parse_program(text)               # extract candidate op-stream / knob set
        ok, toffoli, width = proxy_verifier(prog) # our fast Python verifier
        if not ok:
            rewards.append(-1.0)                  # invalid / non-reversible
        else:
            rewards.append(score_to_reward(toffoli * width))  # lower cost -> higher reward
    return rewards
# Pair with a cheap format reward (valid structure) at low weight, like Unsloth's XML examples.
```

**GRPO config (LoRA, colocated vLLM):**
```python
GRPOConfig(
    use_vllm = True, vllm_mode = "colocate",   # Unsloth: set via fast_inference=True
    learning_rate = 1e-5,                      # ~10x base for GRPO+LoRA, but << SFT lr
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 8,                       # group size 8 (try 4–16); cost scales with this
    max_prompt_length = 1024,
    max_completion_length = 1024,
    num_train_epochs = 1,                      # or cap by max_steps (a few hundred)
    bf16 = True,
)
```
Same LoRA `r=16` adapter continued from SFT. Set `os.environ["UNSLOTH_VLLM_STANDBY"]="1"` and tune
`gpu_memory_utilization` (~0.5–0.6) to balance vLLM KV cache vs training on 24 GB. If NCCL hangs on
weight sync, try `NCCL_P2P_DISABLE=1` / `NCCL_CUMEM_ENABLE=1`.

---

## 4. Modal specifics

**GPU choice.**
- **1.5B SFT + GRPO → A10G (24 GB)** is the cheapest GPU that fits colocated train+vLLM for 1.5B
  (well within range per Unsloth; even 70B QLoRA-GRPO targets 48–80 GB, so 1.5B has large headroom).
- **3B (Llama-3.2) → L40S (48 GB)** for comfortable colocated GRPO, or A10G for SFT-only.
- **A100-40/80 GB:** not needed for 1.5–3B LoRA. Only reach for it if we batch many long rollouts and
  want wall-clock speed (it costs ~2x A10G); usually not worth it for the PoC.

Modal per-hour (per pricing page, per-second billed): **A10G ≈ $1.10**, **L40S ≈ $1.95**,
**A100-40GB ≈ $2.10**. Apply a 3x multiplier only if we mark functions non-preemptible (we don't need
to for training).

**Image (include vLLM — yes, for colocated rollouts).** Modal's Python-native image chain (no
Dockerfile). Pin versions:
```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "unsloth", "unsloth_zoo",        # SFT + GRPO + GGUF export
        "trl", "peft", "transformers", "datasets", "accelerate",
        "vllm",                          # colocated rollouts during GRPO
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "UNSLOTH_VLLM_STANDBY": "1"})
)
# GGUF stage: either rely on Unsloth's bundled llama.cpp, or .apt_install("build-essential","cmake")
# + clone/build ggml-org/llama.cpp for convert_hf_to_gguf.py + llama-quantize.
```
For final Ollama packaging, GGUF can be built on Modal and downloaded, then `ollama create` runs
**locally on macOS** (final eval is local) — no need to run Ollama itself on Modal.

**Volume layout** (persist weights + checkpoints + artifacts; disks are ephemeral):
```python
hf_cache   = modal.Volume.from_name("hf-cache",        create_if_missing=True)  # base weights
ckpt_vol   = modal.Volume.from_name("ecdsa-ckpts",     create_if_missing=True)  # SFT + GRPO LoRA
artifacts  = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)  # merged fp16 + GGUF
# mounts: {"/root/.cache/huggingface": hf_cache, "/ckpts": ckpt_vol, "/artifacts": artifacts}
# layout: /ckpts/sft/<run>/  /ckpts/grpo/<run>/  /artifacts/<model>-merged/  /artifacts/<model>.gguf
```
Use a `modal.Secret` for `HF_TOKEN` (Llama gated) and optionally `wandb`.

**Rough cost (the numbers that matter for $500).**
- **1.5B LoRA SFT**, 1–2 epochs, ~2–6k ex, A10G: a few-k-example LoRA epoch on a 1.5B is on the order
  of **15–40 min/epoch** → **~$0.5–1.5/epoch** → **$2–6** for the SFT phase incl. a re-run.
- **Short GRPO**, few hundred steps, `num_generations=8`, colocated vLLM, A10G: rollouts dominate;
  budget **~$2–4/hr effective**, ~few hours → **$15–40** for the RL phase.
- 3B (L40S) secondary run roughly **2x** the 1.5B numbers.
- **Full PoC** (both models, SFT + GRPO + a couple sweeps + eval + GGUF builds): **~$80–150**.
  Large margin under $500; spend the rest on sweeps/longer GRPO only if the PoC proves out.

**Discipline:** start A10G + 1.5B + short runs; log every Modal run's wall-clock × rate; scale GPU or
steps only when the PoC needs it.

---

## 5. GGUF + Ollama export path + gotchas

**Path (Unsloth one-call, recommended):**
```python
# After GRPO, the LoRA is on the patched model. Merge + convert + quantize in one call:
model.save_pretrained_gguf("/artifacts/ecdsa-coder-1.5b", tokenizer,
                           quantization_method="q4_k_m")
# (Unsloth merges LoRA->fp16 then runs llama.cpp convert + llama-quantize internally.)
```

**Path (manual llama.cpp, if we want full control / imatrix):**
```bash
# 1) merge LoRA into base (Unsloth: save_pretrained_merged(..., save_method="merged_16bit"))
# 2) convert fp16 HF -> GGUF
python convert_hf_to_gguf.py /artifacts/merged --outfile model-f16.gguf --outtype f16
# 3) quantize (Q4_K_M sweet spot for 1.5-3B; --outtype can't emit K-quants, needs this step)
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
#    optional quality bump: build an imatrix and pass --imatrix
```

**Ollama Modelfile** (Qwen2.5 = ChatML; Llama-3.2 = Llama-3 headers). Example for Qwen:
```
FROM ./ecdsa-coder-1.5b-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.6
PARAMETER num_ctx 4096
SYSTEM "You optimize reversible-circuit op-streams under a Toffoli×width cost budget."
```
```bash
ollama create ecdsa-coder-1.5b -f Modelfile
ollama run  ecdsa-coder-1.5b
```

**Gotchas (each has bitten people in the cited docs):**
1. **Merge LoRA BEFORE convert.** `convert_hf_to_gguf.py` wants a full HF model; convert the merged
   fp16, not the adapter. (llama.cpp can convert a LoRA separately via `convert_lora_to_gguf.py`, but
   for Ollama we ship a merged GGUF.)
2. **Chat template / EOS must match training.** Unsloth's #1 "garbage output in another runtime" cause.
   Use the exact template+stop token the model trained with (ChatML `<|im_end|>` for Qwen; Llama-3
   `<|eot_id|>` for Llama). Mismatch → infinite generation or nonsense.
3. **Quant level for 1.5–3B: Q4_K_M is the default sweet spot.** Small fine-tuned models can be
   quant-sensitive — if outputs degrade at Q4, bump to **Q5_K_M / Q6_K** (one cited case: fine-tune was
   nonsense at Q4, fine at Q6). Validate the GGUF before declaring done.
4. **Architecture support:** Qwen2.5 / Llama-3.2 are both supported by current `convert_hf_to_gguf.py`;
   no risk. (Would matter for exotic/MoE bases — not ours.)
5. **`--outtype` can't emit K-quants** ({f32,f16,bf16,q8_0,...} only) → the separate `llama-quantize`
   (or Unsloth's `quantization_method`) step is mandatory for Q4_K_M.
6. **Validate post-export:** run a few prompts through Ollama and compare to the HF/vLLM checkpoint
   before shipping; catches template/quant regressions early.

---

## Modal app skeleton — OUTLINE

> Skeleton/outline only (structure + decorators + control flow), not full runnable code. Three Modal
> functions on the shared image/volumes above: SFT → GRPO → export. Ollama packaging is local.

```python
# train/modal_app.py  (OUTLINE)
import os, modal

app = modal.App("ecdsa-model")

# ---- image: CUDA + python + unsloth/trl/peft/vllm + (optional) llama.cpp build ----
image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
         .entrypoint([])
         .uv_pip_install("unsloth","unsloth_zoo","trl","peft","transformers",
                         "datasets","accelerate","vllm","huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER":"1","UNSLOTH_VLLM_STANDBY":"1"}))

# ---- volumes ----
hf_cache  = modal.Volume.from_name("hf-cache",        create_if_missing=True)
ckpts     = modal.Volume.from_name("ecdsa-ckpts",     create_if_missing=True)
artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)
VOLS = {"/root/.cache/huggingface": hf_cache, "/ckpts": ckpts, "/artifacts": artifacts}
SECRETS = [modal.Secret.from_name("huggingface")]   # HF_TOKEN for gated Llama

# ---- 1) SFT (LoRA) ----
@app.function(image=image, gpu="A10G", timeout=60*60*3, volumes=VOLS, secrets=SECRETS)
def sft(base="unsloth/Qwen2.5-Coder-1.5B-Instruct", data_path="/artifacts/sft.jsonl"):
    # FastLanguageModel.from_pretrained(base, max_seq_length=4096, load_in_4bit=True)
    # get_peft_model(r=16, lora_alpha=32, target_modules=all-linear, grad_ckpt="unsloth")
    # SFTTrainer(model, SFTConfig(bs=4, ga=4, epochs=2, lr=2e-4, cosine, bf16)).train()
    # mask prompt; native chat template; save -> /ckpts/sft/<run>/
    ...

# ---- 2) GRPO (custom Python reward, colocated vLLM) ----
@app.function(image=image, gpu="A10G", timeout=60*60*8, volumes=VOLS, secrets=SECRETS)
def grpo(sft_ckpt="/ckpts/sft/<run>", base="unsloth/Qwen2.5-Coder-1.5B-Instruct"):
    # FastLanguageModel.from_pretrained(base, fast_inference=True, max_lora_rank=16,
    #                                   gpu_memory_utilization=0.55); load SFT LoRA
    # def proxy_cost_reward(prompts, completions, **kw): -> list[float]  (our verifier)
    # GRPOTrainer(model, reward_funcs=[proxy_cost_reward, format_reward],
    #   args=GRPOConfig(use_vllm=True, vllm_mode="colocate", num_generations=8,
    #                   lr=1e-5, max_steps=300, bf16)).train()
    # save -> /ckpts/grpo/<run>/
    ...

# ---- 3) Export merged fp16 + GGUF (Q4_K_M) ----
@app.function(image=image, gpu="A10G", timeout=60*60*2, volumes=VOLS, secrets=SECRETS)
def export_gguf(grpo_ckpt="/ckpts/grpo/<run>"):
    # load base + GRPO LoRA -> save_pretrained_merged(merged_16bit) -> /artifacts/<m>-merged
    # model.save_pretrained_gguf("/artifacts/ecdsa-coder-1.5b", tok, quantization_method="q4_k_m")
    # artifacts.commit()  (then download GGUF locally for `ollama create`)
    ...

# ---- (secondary) same three fns parametrized for Llama-3.2-3B on gpu="L40S" ----
# ---- local: download GGUF -> write Modelfile (ChatML/Llama template) -> `ollama create` ----
@app.local_entrypoint()
def main(stage: str = "sft"):
    {"sft": sft, "grpo": grpo, "export": export_gguf}[stage].remote()
```

---

## Open risks / watch-items
- **Proxy↔real-harness gap:** GRPO optimizes the *fast Python proxy* (Toffoli×width). The headline
  test is the real ECDSA-fail harness — confirm proxy improvements transfer (track T-CFG).
- **Reward hacking:** model may emit trivially "valid" low-cost programs that don't generalize; keep a
  format/validity reward and inspect top-reward completions early.
- **Quant sensitivity at 1.5B:** validate Q4_K_M output vs the HF checkpoint; fall back to Q5_K_M/Q6_K
  if degraded.
- **Llama gating + license:** needs `HF_TOKEN`; redistribution carries "Built with Llama" obligations —
  keep Qwen (Apache-2.0) as the shipped artifact.
- **Version pinning:** pin `unsloth`/`vllm`/`trl` in the Modal image; the GRPO+vLLM+LoRA path is the
  most version-sensitive surface (NCCL/weight-sync quirks → `NCCL_P2P_DISABLE=1`).

## Sources
- TRL GRPOTrainer / SFTTrainer / vLLM integration / PEFT: https://huggingface.co/docs/trl/grpo_trainer ·
  https://huggingface.co/docs/trl/vllm_integration · https://github.com/huggingface/trl (ctx7 /huggingface/trl)
- Modal GRPO+TRL example & vLLM inference: https://modal.com/docs/examples/grpo_trl ·
  https://modal.com/docs/examples/vllm_inference
- Unsloth GRPO + GGUF export + Ollama saving: https://unsloth.ai/docs (ctx7 /websites/unsloth_ai) ·
  https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide
- verifiers (PrimeIntellect): https://github.com/willccbb/verifiers · https://deepwiki.com/willccbb/verifiers
  (ctx7 /primeintellect-ai/verifiers)
- llama.cpp convert/quantize + Ollama Modelfile: https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
- Modal GPU pricing: https://modal.com/pricing
- Licenses: Qwen2.5-Coder (Apache-2.0 for 1.5B; Research for 3B) https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE ·
  Llama 3.2 Community License (Meta)
