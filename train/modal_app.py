"""
modal_app.py — Modal training application for the ECDSA-fail coder model.

Implements the SFT -> GRPO -> GGUF-export pipeline from
`recon/training_stack.md` as a single `modal.App("ecdsa-coder")`.

Pipeline (all stages share volumes; only GRPO/SFT/export touch a GPU):

    smoke        LIGHT cpu image — proves Modal plumbing works (no torch).
    sft          A10G — Unsloth FastLanguageModel 4-bit + TRL SFTTrainer LoRA on JSONL.
    grpo         A10G — continues the SFT LoRA with Unsloth/TRL GRPO + colocated vLLM,
                 reward = proxy_env.reward (Toffoli x width verifier) over tasks.build_curriculum().
    export_gguf  A10G — merge LoRA -> fp16 -> save_pretrained_gguf (q4_k_m) into /artifacts.

Run the cheap light path (builds ONLY the slim image, CPU container — no heavy build):

    modal run train/modal_app.py::smoke

Heavy stages (build the multi-GB CUDA image + spend GPU $$ — run later). They REQUIRE
the ECDSA_BUILD_HEAVY=1 env var so the CUDA/unsloth/vllm image is actually constructed
(without it the heavy functions reuse the light image and refuse to run):

    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py::sft         --run-name qwen-sft-v1
    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py::grpo        --run-name qwen-grpo-v1
    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py::export-gguf --run-name qwen-grpo-v1

Or dispatch via the local entrypoint:

    modal run train/modal_app.py --stage smoke
    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage sft    --run-name qwen-sft-v1
    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage grpo   --run-name qwen-grpo-v1
    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage export --run-name qwen-grpo-v1

Cost (per training_stack.md): SFT ~$2-6, short GRPO ~$15-40, full PoC ~$80-150 (<<$500).
"""

import os
from pathlib import Path

import modal

# ----------------------------------------------------------------------------------
# App + shared resources
# ----------------------------------------------------------------------------------

app = modal.App("ecdsa-coder")

MINUTES = 60  # seconds, for readable timeouts
HOURS = 60 * 60

# Repo-relative paths to mount into the heavy image (proxy reward + SFT data).
HERE = Path(__file__).parent.resolve()                 # .../ecdsa-model/train
PROXY_DIR = (HERE.parent / "proxy").resolve()          # .../ecdsa-model/proxy
DATA_DIR = (HERE.parent / "data").resolve()            # .../ecdsa-model/data

# Default base models (training_stack.md §1). Qwen is Apache-2.0 + UNGATED (no token).
DEFAULT_BASE = "unsloth/Qwen2.5-Coder-1.5B-Instruct"   # primary, Apache-2.0, no HF_TOKEN needed
GEMMA_E2B = "unsloth/gemma-4-E2B-it"                    # parallel arm, Apache-2.0, Ollama-native
GEMMA_E4B = "unsloth/gemma-4-E4B-it"                    # parallel arm, Apache-2.0, Ollama-native
LLAMA_BASE = "unsloth/Llama-3.2-3B-Instruct"           # optional, gated -> needs HF secret

# Shared system prompt — MUST match data/build_sft.py SYSTEM so SFT and GRPO see the same
# persona/format (train/inference consistency; mismatch hurts transfer).
SYSTEM = (
    "You are a specialist in reversible (quantum) circuit optimization for the secp256k1 "
    "point-addition challenge and the broader class of cost-under-constraints reversible-circuit "
    "problems. You minimize cost = (executed Toffoli count) x (peak qubit width) under HARD validity "
    "constraints: the circuit must be classically correct on every input, be a full-state bijection "
    "(reversible), leave global phase zero, and return every ancilla qubit to |0>. Method: smallest "
    "bounded change — inspect, diagnose, cite the exact lever/metric, quantify the Toffoli/qubit/phase "
    "impact, make ONE bounded change; never trade correctness for a cheaper count. You emit circuits in "
    "the harness op-stream DSL (one op per line: X qT; CX qC qT; CCX qC1 qC2 qT (Toffoli, the cost lever); "
    "SWAP qA qB; Z/CZ/CCZ; optional `if bM`; the LAST qubit token is the target; only CCX/CCZ cost; "
    "peak width = max qubit index + 1), and you tune the secp256k1 circuit through its DIALOG_* env knobs."
)

# ----------------------------------------------------------------------------------
# Volumes (persist base weights + checkpoints + artifacts; container disks are ephemeral)
# ----------------------------------------------------------------------------------

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)        # base weights / HF cache
ckpts = modal.Volume.from_name("ecdsa-ckpts", create_if_missing=True)        # SFT + GRPO LoRA
artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)  # merged fp16 + GGUF

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/ckpts": ckpts,
    "/artifacts": artifacts,
}

# Optional HF token secret — ONLY needed for the gated Llama secondary base.
# Modal's `Secret.from_name` is lazy: it does NOT raise if the secret is missing until the
# app resolves the function (i.e. mid-run, even for an unrelated function like smoke()).
# So we eagerly *hydrate* it here at module load to detect existence, and attach it only if
# it really exists — the Qwen primary base is UNGATED and needs no token at all.
def _hf_secrets():
    """Return [Secret] iff a 'huggingface' secret exists in this workspace, else []."""
    try:
        s = modal.Secret.from_name("huggingface")
        s.hydrate()  # forces the lookup now; raises NotFoundError if absent
        return [s]
    except Exception:
        # No HF token configured -> run without it (fine for the Apache-2.0 Qwen base).
        return []


# Resolve once at import so every heavy function shares the same decision.
HF_SECRETS = _hf_secrets()


# ----------------------------------------------------------------------------------
# LIGHT image — slim Python, nothing heavy. Used by smoke() to validate plumbing cheaply.
# ----------------------------------------------------------------------------------

light_image = modal.Image.from_registry("python:3.12-slim")

# ----------------------------------------------------------------------------------
# HEAVY training image — CUDA + python 3.12 + Unsloth/TRL/vLLM stack.
#
# IMPORTANT — why this is gated behind BUILD_HEAVY:
#   `modal run app.py::smoke` resolves the WHOLE app and eagerly builds EVERY image
#   attached to any function — including this multi-GB CUDA image. That defeats the
#   purpose of a cheap light-path smoke test. So we only construct the heavy image when
#   the env var ECDSA_BUILD_HEAVY=1 is set (the heavy entrypoints set it for you). For a
#   plain `smoke` run the heavy functions fall back to the light image — they are never
#   invoked during smoke, so this is safe and keeps the smoke build tiny.
#
# Version pinning (2026): the GRPO + vLLM + LoRA path is the most version-sensitive
# surface in the whole stack (NCCL weight-sync, xformers/triton/torch ABI). These pins
# sit inside the dependency windows declared by unsloth==2026.1.4
# (peft>=0.18.0 ; trl in [0.18.2,0.24.0] !=0.19.0 ; transformers in [4.51.3,4.57.6]).
# If Unsloth/vLLM break on a future release, refresh this block from the Unsloth release
# notes (or use their `--force-reinstall --no-deps unsloth unsloth_zoo` resolver).
# ----------------------------------------------------------------------------------

BUILD_HEAVY = os.environ.get("ECDSA_BUILD_HEAVY", "0") == "1"


def _build_train_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
        )
        .entrypoint([])  # clear nvidia base entrypoint so Modal's own runner takes over
        .apt_install("git", "build-essential", "cmake", "curl")
        # Torch first, pinned to the cu129 wheel matching the CUDA 12.9 base image, so the
        # downstream unsloth/vllm resolution doesn't drag in a mismatched torch.
        .pip_install(
            "torch==2.8.0",
            index_url="https://download.pytorch.org/whl/cu129",
        )
        .pip_install(
            # --- Unsloth training core (SFT + GRPO + GGUF export) ---
            "unsloth==2026.1.4",
            "unsloth_zoo==2026.1.4",
            # --- HuggingFace RL/PEFT stack (inside unsloth's declared version windows) ---
            "trl==0.23.0",
            "peft==0.18.0",
            "transformers==4.57.1",
            "datasets==3.6.0",
            "accelerate==1.10.1",
            # --- colocated vLLM rollouts for GRPO (single-GPU) ---
            "vllm==0.11.0",
            # --- fast HF downloads ---
            "huggingface_hub[hf_transfer]==0.35.3",
            # --- misc pins the GRPO+vLLM path is sensitive to ---
            "triton==3.4.0",
            "bitsandbytes==0.48.1",
        )
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",   # fast model downloads
                "UNSLOTH_VLLM_STANDBY": "1",        # release vLLM VRAM during the train step
                "UNSLOTH_RETURN_LOGITS": "0",
                "TOKENIZERS_PARALLELISM": "false",
                # The heavy image IS the heavy build — carry the flag so _require_heavy()
                # passes inside the container (the local ECDSA_BUILD_HEAVY env is NOT
                # propagated into Modal containers; it only governs local image selection).
                "ECDSA_BUILD_HEAVY": "1",
            }
        )
        # Mount the proxy reward package so `import proxy_env` / `import tasks` work in GRPO.
        # add_local_dir is the final, runtime-only layer at /root/proxy.
        .add_local_dir(str(PROXY_DIR), remote_path="/root/proxy")
        # Mount the local SFT data dir as a fallback source for sft() when no volume path.
        .add_local_dir(str(DATA_DIR), remote_path="/root/data")
    )


# Heavy image only when explicitly requested; otherwise reuse the light image so a smoke
# run does NOT build the CUDA/unsloth/vllm stack. The heavy functions are never invoked
# during smoke, so the fallback image choice has no runtime effect there.
train_image = _build_train_image() if BUILD_HEAVY else light_image

# Make /root/proxy importable inside heavy functions.
PROXY_REMOTE = "/root/proxy"


def _require_heavy():
    """Fail fast (before any GPU spin-up) if a heavy stage was launched without the
    CUDA/unsloth image. Tells the caller exactly how to relaunch."""
    if not BUILD_HEAVY:
        raise RuntimeError(
            "This stage needs the heavy CUDA/unsloth image, which is only built when "
            "ECDSA_BUILD_HEAVY=1 is set. Relaunch e.g.:\n"
            "    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py::sft --run-name <name>"
        )


# ==================================================================================
# 0) SMOKE — LIGHT image, CPU only. Proves Modal plumbing without torch / GPU spend.
# ==================================================================================

@app.function(image=light_image, timeout=5 * MINUTES)
def smoke():
    """CPU-only sanity check. Imports nothing heavy; just confirms a Modal container
    spins up, runs our code, and returns. This is the ONLY thing safe to run for free."""
    import platform
    import sys

    info = {
        "message": "Modal plumbing OK — light image container ran ecdsa-coder.smoke()",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd_listing": sorted(os.listdir("/"))[:12],
    }
    print("=" * 64)
    print("[smoke] " + info["message"])
    print(f"[smoke] python  : {info['python']}")
    print(f"[smoke] platform: {info['platform']}")
    print(f"[smoke] machine : {info['machine']}")
    print("[smoke] (no torch imported here — heavy image is built lazily on GPU calls)")
    print("=" * 64)
    return info


# ==================================================================================
# Shared helpers for the heavy (GPU) stages
# ==================================================================================

CHATML_MAX_SEQ = 4096


def _family(base_model: str) -> str:
    b = base_model.lower()
    if "qwen" in b:
        return "qwen"
    if "gemma" in b:
        return "gemma"
    if "llama" in b:
        return "llama"
    return "other"


def _chat_cfg(base_model: str):
    """(chat_template_name_or_None, instruction_part, response_part) per model family.
    Wrong template/EOS markers are the #1 GGUF/garbage-output footgun, so be explicit.
    Gemma's assistant role marker is 'model' (not 'assistant'); use the native template."""
    fam = _family(base_model)
    if fam == "qwen":
        return ("qwen-2.5", "<|im_start|>user\n", "<|im_start|>assistant\n")
    if fam == "gemma":
        return (None, "<start_of_turn>user\n", "<start_of_turn>model\n")
    if fam == "llama":
        return ("llama-3.1",
                "<|start_header_id|>user<|end_header_id|>\n\n",
                "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return (None, None, None)


def _vllm_ok(base_model: str) -> bool:
    """Gemma-4 E-series is NOT supported by vLLM (2026) -> GRPO must use HF generate
    (Unsloth fast_inference=False). Everything else uses colocated vLLM rollouts."""
    b = base_model.lower()
    if "gemma" in b and ("e2b" in b or "e4b" in b):
        return False
    return True


def _prep_messages(messages, base_model):
    """Make a messages list safe for the model's chat template. Gemma's template does
    NOT accept a separate `system` role, so fold any leading system content into the
    first user turn. Other families pass through unchanged."""
    if _family(base_model) != "gemma":
        return messages
    out, sys_text = [], None
    for m in messages:
        if m["role"] == "system":
            sys_text = m["content"]
            continue
        if sys_text and m["role"] == "user":
            m = {"role": "user", "content": sys_text + "\n\n" + m["content"]}
            sys_text = None
        out.append(m)
    if sys_text:  # no following user turn — promote system to a user turn
        out.insert(0, {"role": "user", "content": sys_text})
    return out


def _load_jsonl(path: str):
    """Load a JSONL dataset. Each line should be either:
      {"messages": [{role, content}, ...]}            (preferred; chat format)
    or {"prompt": "...", "completion": "..."}          (will be wrapped into ChatML).
    Returns a list of dicts with a 'messages' key (ChatML-ready)."""
    import json

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                rows.append({"messages": obj["messages"]})
            elif "prompt" in obj and "completion" in obj:
                rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": obj["prompt"]},
                            {"role": "assistant", "content": obj["completion"]},
                        ]
                    }
                )
            else:
                # tolerate the raw-moves schema by skipping non-conformant rows
                continue
    return rows


# ==================================================================================
# 1) SFT — Unsloth FastLanguageModel 4-bit + TRL SFTTrainer LoRA (prompt-masked, ChatML)
# ==================================================================================

@app.function(
    image=train_image,
    gpu="A10G",
    timeout=3 * HOURS,
    volumes=VOLUMES,
    secrets=HF_SECRETS,
)
def sft(
    base_model: str = DEFAULT_BASE,
    run_name: str = "qwen-sft-v1",
    dataset_path: str = "/root/data/sft.jsonl",
    epochs: int = 2,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_len: int = CHATML_MAX_SEQ,
    per_device_bs: int = 4,
    grad_accum: int = 4,
):
    """LoRA SFT on distilled reasoning/move traces.

    `dataset_path` can point at a file inside a mounted volume (e.g.
    /artifacts/sft.jsonl or /ckpts/...) OR at the image-mounted /root/data/sft.jsonl.
    Saves the LoRA adapter + a merged 16-bit model under /ckpts/sft/<run_name> and copies
    the adapter to /artifacts for convenience.
    """
    _require_heavy()
    import os as _os

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"[sft] base={base_model} run={run_name} data={dataset_path} epochs={epochs}")

    # ---- load model 4-bit (QLoRA) -------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        load_in_4bit=True,    # QLoRA; buys VRAM headroom on the 24GB A10G
        dtype=None,           # auto (bf16 on Ampere+)
    )

    # ---- attach LoRA (all-linear target = best quality) ---------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ---- chat template (family-aware; consistent end-to-end SFT->GRPO->GGUF) -------
    tmpl, instr_part, resp_part = _chat_cfg(base_model)
    if tmpl is not None:
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=tmpl)
        except Exception as e:
            print(f"[sft] get_chat_template({tmpl}) -> native fallback: {e}")
    else:
        print(f"[sft] using native chat template for {base_model} (family={_family(base_model)})")

    # ---- dataset -> rendered ChatML text -----------------------------------------
    if not _os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"SFT dataset not found at {dataset_path}. Upload it to a volume "
            f"(e.g. /artifacts/sft.jsonl) or place it in train-mounted /root/data/."
        )
    rows = _load_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"No usable rows parsed from {dataset_path}")
    print(f"[sft] loaded {len(rows)} examples")

    def _format(ex):
        msgs = _prep_messages(ex["messages"], base_model)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = Dataset.from_list(rows).map(_format, remove_columns=["messages"])

    # ---- trainer (TRL SFTConfig) --------------------------------------------------
    out_dir = f"/ckpts/sft/{run_name}"
    cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,   # effective batch 16
        warmup_ratio=0.05,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_len,
        packing=False,        # keep traces atomic
    )
    # TRL renamed `tokenizer` -> `processing_class`; prefer the new kwarg, fall back for
    # older TRL / Unsloth-patched signatures.
    try:
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=ds, args=cfg
        )
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)

    # ---- prompt masking: train on the assistant response only ---------------------
    # Family-aware markers (Qwen ChatML / Gemma <start_of_turn> / Llama headers); this masks
    # everything up to the assistant turn so the model learns the reasoning+move (stack §2).
    if instr_part and resp_part:
        try:
            trainer = train_on_responses_only(
                trainer, instruction_part=instr_part, response_part=resp_part,
            )
        except Exception as e:
            print(f"[sft] train_on_responses_only skipped ({e}); training on full text")
    else:
        print("[sft] no response markers for this family; training on full text")

    trainer.train()

    # ---- save LoRA + merged 16-bit ------------------------------------------------
    model.save_pretrained(out_dir)                 # LoRA adapter
    tokenizer.save_pretrained(out_dir)
    merged_dir = f"/artifacts/{run_name}-merged"
    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[sft] merged 16-bit -> {merged_dir}")
    except Exception as e:
        print(f"[sft] merged save skipped ({e})")

    # also drop the adapter under /artifacts for easy retrieval
    model.save_pretrained(f"/artifacts/{run_name}-lora")
    ckpts.commit()
    artifacts.commit()
    print(f"[sft] DONE. LoRA -> {out_dir} ; merged -> {merged_dir}")
    return {"lora": out_dir, "merged": merged_dir}


# ==================================================================================
# 2) GRPO — continue SFT LoRA, Unsloth/TRL GRPO with colocated vLLM + proxy reward
# ==================================================================================

@app.function(
    image=train_image,
    gpu="A10G",
    timeout=8 * HOURS,
    volumes=VOLUMES,
    secrets=HF_SECRETS,
)
def grpo(
    base_model: str = DEFAULT_BASE,
    sft_path: str = "/ckpts/sft/qwen-sft-v1",
    run_name: str = "qwen-grpo-v1",
    max_steps: int = 300,
    lr: float = 1e-5,
    num_generations: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_prompt_len: int = 1024,
    max_completion_len: int = 1024,
    gpu_mem_util: float = 0.55,
):
    """GRPO RL post-training against the proxy verifier reward.

    Loads the base with Unsloth fast_inference (in-process vLLM), continues the SFT LoRA,
    and runs TRL GRPOTrainer with reward_funcs = [proxy_env.reward, proxy_env.format_reward]
    over rendered proxy task prompts (tasks.build_curriculum()). Saves the GRPO LoRA to
    /ckpts/grpo/<run_name>.
    """
    _require_heavy()
    import os as _os
    import random
    import sys

    # NCCL/vLLM weight-sync robustness on single GPU (training_stack.md §3 watch-items).
    _os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    _os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    _os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")

    # make the mounted proxy package importable
    if PROXY_REMOTE not in sys.path:
        sys.path.insert(0, PROXY_REMOTE)

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import GRPOConfig, GRPOTrainer

    import proxy_env   # our reward backend (Toffoli x width verifier)
    import tasks        # curriculum + prompt rendering

    print(f"[grpo] base={base_model} sft={sft_path} run={run_name} steps={max_steps}")

    # ---- model + rollout backend (family-aware) -----------------------------------
    # Gemma-4 E-series is NOT vLLM-supported (2026) -> fast_inference=False (HF generate).
    use_vllm = _vllm_ok(base_model)
    fp_kwargs = dict(
        model_name=base_model,
        max_seq_length=max_prompt_len + max_completion_len,
        load_in_4bit=True,
        max_lora_rank=lora_r,
    )
    if use_vllm:
        fp_kwargs.update(fast_inference=True, gpu_memory_utilization=gpu_mem_util)
    else:
        fp_kwargs.update(fast_inference=False)
        print(f"[grpo] {base_model}: vLLM unsupported -> fast_inference=False (slower HF rollouts)")
    model, tokenizer = FastLanguageModel.from_pretrained(**fp_kwargs)

    # match the SFT chat template so RL prompts are formatted identically (transfer)
    tmpl, _instr, _resp = _chat_cfg(base_model)
    if tmpl is not None:
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=tmpl)
        except Exception as e:
            print(f"[grpo] get_chat_template({tmpl}) -> native fallback: {e}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ---- continue from the SFT LoRA if present ------------------------------------
    if _os.path.isdir(sft_path):
        try:
            model.load_adapter(sft_path, adapter_name="default")
            print(f"[grpo] loaded SFT LoRA from {sft_path}")
        except Exception as e:
            print(f"[grpo] could not load SFT adapter ({e}); starting from base+fresh LoRA")
    else:
        print(f"[grpo] no SFT path at {sft_path}; starting GRPO from base + fresh LoRA")

    # ---- dataset = rendered proxy task prompts -----------------------------------
    # tasks.build_curriculum() returns {band: [Instance, ...]}; each Instance carries a
    # rendered .prompt and a .task_spec that holds a LIVE python callable `f` (so it is not
    # JSON-serializable). We therefore keep the instances in memory and pass only an integer
    # `task_id` through the HF dataset; the reward fn recovers the full spec by index.
    from datasets import Dataset

    rng = random.Random(3407)
    curric = tasks.build_curriculum(rng=rng)
    instances = [inst for band in sorted(curric) for inst in curric[band]]
    if not instances:
        raise RuntimeError("tasks.build_curriculum() produced no instances")
    print(f"[grpo] curriculum: {len(instances)} task instances across "
          f"{len(curric)} bands")

    def _wrap(p):
        msgs = _prep_messages(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            base_model,
        )
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    dataset = Dataset.from_dict({
        "prompt": [_wrap(inst.prompt) for inst in instances],
        "task_id": list(range(len(instances))),
    })

    def _proxy_reward(prompts=None, completions=None, **kw):
        # map each rollout's task_id back to its in-memory task_spec (with live `f`)
        ids = kw.get("task_id")
        specs = [instances[int(i)].task_spec for i in ids] if ids is not None else None
        return proxy_env.reward(prompts=prompts, completions=completions, task_spec=specs)

    def _fmt_reward(prompts=None, completions=None, **kw):
        return proxy_env.format_reward(prompts=prompts, completions=completions)

    reward_funcs = [_proxy_reward, _fmt_reward]

    out_dir = f"/ckpts/grpo/{run_name}"
    grpo_kwargs = {}
    if use_vllm:
        grpo_kwargs.update(use_vllm=True, vllm_mode="colocate")  # single-GPU colocated rollouts
    else:
        grpo_kwargs.update(use_vllm=False)                       # Gemma E-series: HF generate
    cfg = GRPOConfig(
        output_dir=out_dir,
        **grpo_kwargs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=num_generations,  # group size (cost scales with this)
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=max_steps,
        logging_steps=2,
        save_steps=max(50, max_steps // 4),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        reward_weights=[1.0, 0.2],       # cost reward dominates; format is a light nudge
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=dataset,
    )
    trainer.train()

    # ---- save GRPO LoRA -----------------------------------------------------------
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(f"/artifacts/{run_name}-grpo-lora")
    ckpts.commit()
    artifacts.commit()
    print(f"[grpo] DONE. GRPO LoRA -> {out_dir}")
    return {"grpo_lora": out_dir}


# ==================================================================================
# 3) EXPORT — merge LoRA -> fp16 -> GGUF (q4_k_m) into /artifacts
# ==================================================================================

@app.function(
    image=train_image,
    gpu="A10G",
    timeout=2 * HOURS,
    volumes=VOLUMES,
    secrets=HF_SECRETS,
)
def export_gguf(
    base_model: str = DEFAULT_BASE,
    ckpt_path: str = "/ckpts/grpo/qwen-grpo-v1",
    run_name: str = "ecdsa-coder-1.5b",
    quant: str = "q4_k_m",
    lora_r: int = 16,
):
    """Merge the (GRPO or SFT) LoRA into the base, then export a quantized GGUF.

    Uses Unsloth's one-call `save_pretrained_gguf` which merges LoRA -> fp16 and runs
    llama.cpp convert + quantize internally. Output GGUF lands in /artifacts.
    """
    _require_heavy()
    import os as _os

    from unsloth import FastLanguageModel

    print(f"[export] base={base_model} ckpt={ckpt_path} quant={quant} run={run_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=CHATML_MAX_SEQ,
        load_in_4bit=False,    # need fp16 for a clean merge before GGUF conversion
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r, lora_alpha=2 * lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    if _os.path.isdir(ckpt_path):
        model.load_adapter(ckpt_path, adapter_name="default")
        print(f"[export] loaded LoRA from {ckpt_path}")
    else:
        print(f"[export] WARNING: no LoRA at {ckpt_path}; exporting base model GGUF")

    out_dir = f"/artifacts/{run_name}"
    _os.makedirs(out_dir, exist_ok=True)
    # one-call merge -> fp16 -> gguf (q4_k_m). Unsloth runs llama.cpp under the hood.
    model.save_pretrained_gguf(out_dir, tokenizer, quantization_method=quant)

    # locate the produced .gguf
    gguf = None
    for root, _dirs, files in _os.walk(out_dir):
        for fn in files:
            if fn.endswith(".gguf"):
                gguf = _os.path.join(root, fn)
    artifacts.commit()
    print(f"[export] DONE. GGUF -> {gguf}")
    print(f"[export] download with: modal volume get ecdsa-artifacts {run_name} ./{run_name}")
    return {"gguf": gguf, "out_dir": out_dir}


# ==================================================================================
# Local entrypoint — dispatch subcommands (smoke / sft / grpo / export)
# ==================================================================================

@app.local_entrypoint()
def main(
    stage: str = "smoke",
    base_model: str = DEFAULT_BASE,
    run_name: str = "",
    dataset_path: str = "/root/data/sft.jsonl",
    sft_path: str = "/ckpts/sft/qwen-sft-v1",
    ckpt_path: str = "/ckpts/grpo/qwen-grpo-v1",
    epochs: int = 2,
    max_steps: int = 300,
    quant: str = "q4_k_m",
    max_seq_len: int = CHATML_MAX_SEQ,
    per_device_bs: int = 4,
):
    """Dispatch a pipeline stage. Defaults to the cheap `smoke` light-path check.

    Examples:
        modal run train/modal_app.py                       # -> smoke
        modal run train/modal_app.py --stage sft   --run-name qwen-sft-v1
        modal run train/modal_app.py --stage grpo  --run-name qwen-grpo-v1
        modal run train/modal_app.py --stage export --run-name ecdsa-coder-1.5b
    """
    stage = stage.lower()
    if stage in ("sft", "grpo", "export") and not BUILD_HEAVY:
        raise SystemExit(
            f"stage '{stage}' needs the heavy CUDA/unsloth image. Relaunch with the env "
            f"var set, e.g.:\n    ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py "
            f"--stage {stage} --run-name <name>"
        )
    if stage == "smoke":
        result = smoke.remote()
        print(f"[main] smoke result: {result}")
    elif stage == "sft":
        rn = run_name or "qwen-sft-v1"
        # 7B needs more VRAM than the A10G -> route it to an L40S at call time.
        fn = sft.with_options(gpu="L40S") if "7b" in base_model.lower() else sft
        result = fn.remote(
            base_model=base_model, run_name=rn,
            dataset_path=dataset_path, epochs=epochs,
            max_seq_len=max_seq_len, per_device_bs=per_device_bs,
        )
        print(f"[main] sft result: {result}")
    elif stage == "grpo":
        rn = run_name or "qwen-grpo-v1"
        result = grpo.remote(
            base_model=base_model, sft_path=sft_path,
            run_name=rn, max_steps=max_steps,
        )
        print(f"[main] grpo result: {result}")
    elif stage == "export":
        rn = run_name or "ecdsa-coder-1.5b"
        result = export_gguf.remote(
            base_model=base_model, ckpt_path=ckpt_path,
            run_name=rn, quant=quant,
        )
        print(f"[main] export result: {result}")
    else:
        raise SystemExit(
            f"unknown stage '{stage}' (expected: smoke | sft | grpo | export)"
        )
