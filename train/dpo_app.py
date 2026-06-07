"""
dpo_app.py — STABLE post-training app for the ECDSA-fail coder model.

This replaces the COLLAPSED GRPO run documented in `eval/EVAL_REPORT.md`. The PoC's GRPO
combined a proxy (validity) reward with a *soft* format reward (weight 0.2); over 30 epochs
on a tiny 22-task curriculum the policy learned to hack the lenient format term — emitting
short repetitive "op-ish" token spam that scored format ≈ 1.0 while the validity reward stayed
≈ 0. Completion length collapsed 320 -> ~60 tokens, KL drifted to 0.13, and the merged model
produced garbage (T-CFG 0/0/0). The SFT model was shipped instead.

This app provides TWO stable replacements plus the smoke plumbing. It deliberately reuses the
exact heavy-image / volume / helper pattern from `train/modal_app.py` (same CUDA+unsloth image,
ECDSA_BUILD_HEAVY gate, hf-cache/ckpts/artifacts volumes, _family/_chat_cfg/_prep_messages/
_load_jsonl helpers, add_local_dir of ../proxy and ../data) so SFT->{DPO,GRPO}->export stays
format-consistent end to end.

  smoke    LIGHT cpu image — proves Modal plumbing works (no torch/GPU build).
  dpo      TRL DPOTrainer (Unsloth FastLanguageModel + LoRA) on JSONL of
           {prompt, chosen, rejected}. chosen = cheaper VALID circuit, rejected = costlier or
           INVALID circuit, ranked by the proxy verifier. DPO has NO reward-hacking surface:
           it only ever sees a fixed preference between two fixed completions, so there is no
           scalar for the policy to game — it cannot invent a degenerate string that scores
           high, it can only move probability mass toward `chosen` and away from `rejected`.
  grpo_v2  FIXED GRPO. ONE validity-gated reward (see grpo_v2 docstring): format is a HARD
           parse-gate INSIDE the reward (never a soft additive term), plus an anti-degeneration
           penalty, higher KL (beta), and max_steps tuned for ~1 epoch over a LARGE task set so
           no task is re-seen. This closes every hole that collapsed the PoC run.

Bases (training_stack.md §1, all Apache-2.0 / ungated -> no HF token needed):
  * unsloth/Qwen2.5-Coder-1.5B-Instruct  -> A10G  (primary, fastest rollouts)
  * unsloth/Qwen2.5-Coder-7B-Instruct    -> L40S  (bigger base, Apache-2.0, 48GB headroom)
  GPU is auto-selected from the base name (7B -> L40S, else A10G); override with --gpu.

Gemma-4 is NOT handled here: its custom Gemma4ClippableLinear layers need the vanilla
HF+PEFT path, which already lives in `train/sft_gemma_standalone.py` (target_modules=["linear"]).
Do NOT duplicate it — run that file for the Gemma arm. (A Gemma DPO arm would extend that
standalone file with TRL DPOTrainer, not this Unsloth app.)

Run the cheap light path (builds ONLY the slim image, CPU container — no heavy build):

    modal run train/dpo_app.py::smoke

Heavy stages (build the multi-GB CUDA image + spend GPU $$). They REQUIRE ECDSA_BUILD_HEAVY=1
so the CUDA/unsloth/vllm image is actually constructed (without it the heavy functions reuse
the light image and refuse to run, exactly like modal_app.py):

    ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py::dpo \
        --base-model unsloth/Qwen2.5-Coder-1.5B-Instruct \
        --run-name qwen-dpo-v1 --dataset-path /artifacts/dpo_pairs.jsonl
    ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py::grpo_v2 \
        --base-model unsloth/Qwen2.5-Coder-7B-Instruct --run-name qwen7b-grpo-v2

Or dispatch via the local entrypoint:

    modal run train/dpo_app.py --stage smoke
    ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py --stage dpo     --run-name qwen-dpo-v1
    ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py --stage grpo_v2 --run-name qwen-grpo-v2
"""

import os
from pathlib import Path

import modal

# ----------------------------------------------------------------------------------
# App + shared resources (mirrors train/modal_app.py)
# ----------------------------------------------------------------------------------

app = modal.App("ecdsa-dpo")

MINUTES = 60  # seconds, for readable timeouts
HOURS = 60 * 60

HERE = Path(__file__).parent.resolve()                 # .../ecdsa-model/train
PROXY_DIR = (HERE.parent / "proxy").resolve()          # .../ecdsa-model/proxy
DATA_DIR = (HERE.parent / "data").resolve()            # .../ecdsa-model/data

# Default bases (Apache-2.0, ungated). 7B is the bigger arm on L40S.
DEFAULT_BASE = "unsloth/Qwen2.5-Coder-1.5B-Instruct"   # primary, A10G
QWEN_7B = "unsloth/Qwen2.5-Coder-7B-Instruct"          # bigger arm, L40S, Apache-2.0
# Gemma is handled by train/sft_gemma_standalone.py (vanilla HF+PEFT) — NOT this app.
GEMMA_E4B = "unsloth/gemma-4-E4B-it"

# Shared system prompt — MUST match data/build_sft.py + modal_app.py SYSTEM so SFT, DPO, and
# GRPO see the same persona/format (train/inference consistency; mismatch hurts transfer).
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
# Volumes (shared with modal_app.py + sft_gemma_standalone.py)
# ----------------------------------------------------------------------------------

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpts = modal.Volume.from_name("ecdsa-ckpts", create_if_missing=True)
artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/ckpts": ckpts,
    "/artifacts": artifacts,
}


def _hf_secrets():
    """Return [Secret] iff a 'huggingface' secret exists in this workspace, else [].
    The Qwen bases are Apache-2.0 + ungated so no token is needed; we only attach the
    secret if it actually exists (lazy hydrate, same trick as modal_app.py)."""
    try:
        s = modal.Secret.from_name("huggingface")
        s.hydrate()
        return [s]
    except Exception:
        return []


HF_SECRETS = _hf_secrets()

# ----------------------------------------------------------------------------------
# LIGHT image — slim Python. Used by smoke() to validate plumbing cheaply.
# ----------------------------------------------------------------------------------

light_image = modal.Image.from_registry("python:3.12-slim")

# ----------------------------------------------------------------------------------
# HEAVY training image — CUDA + python 3.12 + Unsloth/TRL/vLLM. Identical pins to
# modal_app.py so the two apps share the same build cache and version surface.
#
# Gated behind ECDSA_BUILD_HEAVY: `modal run dpo_app.py::smoke` resolves the whole app and
# eagerly builds EVERY image attached to any function. We only construct the multi-GB CUDA
# image when ECDSA_BUILD_HEAVY=1; otherwise heavy functions fall back to the light image
# (they are never invoked during smoke, so this is safe and keeps the smoke build tiny).
# ----------------------------------------------------------------------------------

BUILD_HEAVY = os.environ.get("ECDSA_BUILD_HEAVY", "0") == "1"


def _build_train_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "cmake", "curl")
        .pip_install(
            "torch==2.8.0",
            index_url="https://download.pytorch.org/whl/cu129",
        )
        .pip_install(
            # --- Unsloth training core (SFT + DPO + GRPO + GGUF export) ---
            "unsloth==2026.1.4",
            "unsloth_zoo==2026.1.4",
            # --- HuggingFace RL/PEFT stack (inside unsloth's declared version windows;
            #     trl 0.23.0 ships both DPOTrainer and GRPOTrainer) ---
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
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "UNSLOTH_VLLM_STANDBY": "1",
                "UNSLOTH_RETURN_LOGITS": "0",
                "TOKENIZERS_PARALLELISM": "false",
                # Carry the flag so _require_heavy() passes INSIDE the container (the local
                # ECDSA_BUILD_HEAVY env is not propagated into Modal containers; it only
                # governs local image selection).
                "ECDSA_BUILD_HEAVY": "1",
            }
        )
        # Mount the proxy reward package so `import proxy_env` / `import tasks` work in GRPO.
        .add_local_dir(str(PROXY_DIR), remote_path="/root/proxy")
        # Mount the local data dir as a fallback source for the DPO pairs file.
        .add_local_dir(str(DATA_DIR), remote_path="/root/data")
    )


train_image = _build_train_image() if BUILD_HEAVY else light_image

PROXY_REMOTE = "/root/proxy"


def _require_heavy():
    """Fail fast (before any GPU spin-up) if a heavy stage was launched without the
    CUDA/unsloth image (i.e. ECDSA_BUILD_HEAVY unset)."""
    if not BUILD_HEAVY:
        raise RuntimeError(
            "This stage needs the heavy CUDA/unsloth image, which is only built when "
            "ECDSA_BUILD_HEAVY=1 is set. Relaunch e.g.:\n"
            "    ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py::dpo --run-name <name>"
        )


def _gpu_for(base_model: str) -> str:
    """Pick the GPU from the base size. 7B needs the 48GB L40S; 1.5B fits the 24GB A10G
    (colocated train+vLLM). Anything bigger than ~3B -> L40S."""
    b = base_model.lower()
    if "7b" in b or "8b" in b or "14b" in b:
        return "L40S"
    return "A10G"


# ==================================================================================
# Shared helpers (copied verbatim from modal_app.py to keep format end-to-end consistent)
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
    """(chat_template_name_or_None, instruction_part, response_part) per model family."""
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
    """Gemma-4 E-series is NOT vLLM-supported (2026) -> GRPO must use HF generate.
    (Not reachable here since Gemma routes to the standalone file, but kept for parity.)"""
    b = base_model.lower()
    if "gemma" in b and ("e2b" in b or "e4b" in b):
        return False
    return True


def _prep_messages(messages, base_model):
    """Make a messages list safe for the model's chat template (Gemma folds system into
    the first user turn). Other families pass through unchanged."""
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
    if sys_text:
        out.insert(0, {"role": "user", "content": sys_text})
    return out


def _load_jsonl(path: str):
    """Load a JSONL chat-style dataset (used for the SFT-shaped fallbacks). Each line is
    {"messages": [...]} or {"prompt","completion"}. Returns rows with a 'messages' key."""
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
                continue
    return rows


def _load_pref_jsonl(path: str):
    """Load a DPO preference JSONL. Each line is one of:

      {"prompt": "...", "chosen": "...", "rejected": "..."}          (raw text triple)
      {"messages": [...], "chosen": "...", "rejected": "..."}        (chat prompt + text)

    Returns a list of {"prompt": str_or_messages, "chosen": str, "rejected": str}. The
    prompt is left as-is here; dpo() renders it through the chat template per family.
    chosen MUST be the cheaper-valid completion and rejected the costlier/invalid one — that
    ranking is produced upstream by the proxy verifier (see data/build_dpo.py if present)."""
    import json

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "chosen" not in obj or "rejected" not in obj:
                continue
            if "prompt" in obj:
                prompt = obj["prompt"]
            elif "messages" in obj:
                prompt = obj["messages"]
            else:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": obj["chosen"],
                    "rejected": obj["rejected"],
                }
            )
    return rows


# ==================================================================================
# 0) SMOKE — LIGHT image, CPU only. Proves Modal plumbing without torch / GPU spend.
# ==================================================================================

@app.function(image=light_image, timeout=5 * MINUTES)
def smoke():
    """CPU-only sanity check for the DPO/GRPO-v2 app. Imports nothing heavy; confirms a
    Modal container spins up, runs our code, and that the GPU auto-selection logic +
    anti-degeneration penalty (the core fix) behave as designed. Safe to run for free."""
    import platform
    import sys

    # exercise the pure-python pieces that don't need torch so the smoke actually tests logic
    gpu_map = {
        DEFAULT_BASE: _gpu_for(DEFAULT_BASE),
        QWEN_7B: _gpu_for(QWEN_7B),
        GEMMA_E4B: _gpu_for(GEMMA_E4B),
    }
    fam_map = {b: _family(b) for b in (DEFAULT_BASE, QWEN_7B, GEMMA_E4B)}

    # mini self-test of the anti-degeneration penalty: spammy/short streams must be punished,
    # a healthy varied stream must not be. This is the exact knob that the collapsed PoC lacked.
    spam = "X q0\nX q0\nX q0\nX q0\nX q0\nX q0\n"
    healthy = "X q0\nCX q0 q1\nCCX q0 q1 q2\nSWAP q1 q2\nCZ q0 q1\n"
    pen_spam = _anti_degen_penalty(spam)
    pen_healthy = _anti_degen_penalty(healthy)

    info = {
        "message": "Modal plumbing OK — light image container ran ecdsa-dpo.smoke()",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu_selection": gpu_map,
        "family_detection": fam_map,
        "anti_degen_penalty": {"spam": pen_spam, "healthy": pen_healthy},
        "anti_degen_ok": pen_spam < pen_healthy,  # spam must be punished more (more negative)
        "build_heavy": BUILD_HEAVY,
    }
    print("=" * 70)
    print("[smoke] " + info["message"])
    print(f"[smoke] python   : {info['python']}")
    print(f"[smoke] platform : {info['platform']}")
    print(f"[smoke] gpu map  : 1.5B->{gpu_map[DEFAULT_BASE]}  7B->{gpu_map[QWEN_7B]}")
    print(f"[smoke] anti-degen: spam={pen_spam:.3f} < healthy={pen_healthy:.3f} "
          f"-> {'OK' if info['anti_degen_ok'] else 'BAD'} (spam must score lower)")
    print("[smoke] (no torch imported here — heavy image is built lazily on GPU calls)")
    print("=" * 70)
    return info


# ==================================================================================
# Anti-degeneration penalty — the structural fix that the collapsed PoC GRPO lacked.
#
# Pure-python (no torch) so it is importable in the light smoke image AND inside grpo_v2.
# Returns a NON-POSITIVE penalty (0.0 = clean; negative = degenerate). It punishes exactly
# the failure modes the PoC reward-hack exhibited (320->60 token collapse, repeated "op-ish"
# spam): too few distinct ops, repeated-line spam, and below-minimum length.
# ==================================================================================

def _anti_degen_penalty(
    text: str,
    min_lines: int = 3,
    min_distinct_ops: int = 2,
    max_repeat_frac: float = 0.5,
) -> float:
    """Penalize degenerate op-streams. Looks ONLY at structure, never at validity (validity
    is the verifier's job). Components (each clamped, summed, floored at -1.0):

      below-min length : fewer than `min_lines` real op lines  -> up to -0.4
      too few ops      : fewer than `min_distinct_ops` distinct op KINDS -> -0.3
      repeated-line spam: most-common identical line dominates -> up to -0.5

    A varied multi-op stream returns 0.0. The PoC's `X q0 \\n X q0 \\n ...` spam returns a
    strong negative, so the policy can never park on it for free reward."""
    lines = []
    op_kinds = set()
    for raw in text.splitlines():
        ln = raw.split("#", 1)[0].strip()
        if not ln:
            continue
        lines.append(ln)
        op_kinds.add(ln.split()[0] if ln.split() else "")

    n = len(lines)
    if n == 0:
        return -1.0  # empty/comment-only output is maximally degenerate

    pen = 0.0

    # 1) below-min length: ramp from -0.4 (1 line) to 0 (>= min_lines lines)
    if n < min_lines:
        pen -= 0.4 * (min_lines - n) / min_lines

    # 2) too few distinct op kinds (e.g. all `X`)
    if len(op_kinds) < min_distinct_ops:
        pen -= 0.3

    # 3) repeated-line spam: if the single most common line is > max_repeat_frac of all lines
    from collections import Counter

    most_common = Counter(lines).most_common(1)[0][1]
    repeat_frac = most_common / n
    if repeat_frac > max_repeat_frac:
        # ramp from 0 at the threshold to -0.5 at fully-repeated (frac == 1.0)
        over = (repeat_frac - max_repeat_frac) / (1.0 - max_repeat_frac)
        pen -= 0.5 * over

    return max(-1.0, pen)


# ==================================================================================
# 1) DPO — TRL DPOTrainer on verifier-ranked {prompt, chosen, rejected} pairs.
#
# Why DPO has NO reward-hacking surface (the whole point of this file):
#   GRPO optimizes a SCALAR reward over the policy's OWN samples, so the policy can search
#   for any string that maximizes that scalar — and it found degenerate token-spam. DPO never
#   computes a reward over policy samples. It only ever sees a FIXED (chosen > rejected)
#   preference between two FIXED completions and maximizes the log-prob margin of chosen over
#   rejected, anchored to the SFT reference by `beta`. There is no scalar to game and no free
#   completion to drift toward — the only gradient is "prefer the cheaper-valid circuit over
#   the costlier/invalid one," exactly the signal we want.
# ==================================================================================

@app.function(
    image=train_image,
    gpu="A10G",  # overridden per-base at call time via .with_options when launching 7B
    timeout=4 * HOURS,
    volumes=VOLUMES,
    secrets=HF_SECRETS,
)
def dpo(
    base_model: str = DEFAULT_BASE,
    run_name: str = "qwen-dpo-v1",
    dataset_path: str = "/root/data/dpo_pairs.jsonl",
    sft_path: str = "",
    beta: float = 0.1,
    lr: float = 5e-6,
    epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_prompt_len: int = 1024,
    max_completion_len: int = 1024,
    per_device_bs: int = 1,
    grad_accum: int = 8,
):
    """LoRA DPO on verifier-ranked preference pairs.

    Dataset (`dataset_path`) is JSONL of {prompt, chosen, rejected} where the proxy verifier
    decided chosen = cheaper VALID circuit and rejected = costlier-or-INVALID circuit. We load
    the base 4-bit with Unsloth, optionally continue the SFT LoRA (`sft_path`), and run TRL
    DPOTrainer with beta~0.1, lr~5e-6, 1-2 epochs, bf16, LoRA r16. Saves the LoRA + a merged
    16-bit model under /ckpts/dpo/<run> and /artifacts/<run>-merged.
    """
    _require_heavy()
    import os as _os

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    print(f"[dpo] base={base_model} run={run_name} data={dataset_path} "
          f"beta={beta} lr={lr} epochs={epochs}")

    max_seq = max_prompt_len + max_completion_len

    # ---- load model 4-bit (QLoRA) -------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq,
        load_in_4bit=True,
        dtype=None,
        max_lora_rank=lora_r,
    )

    # ---- chat template (family-aware; consistent SFT->DPO->GGUF) ------------------
    tmpl, _instr, _resp = _chat_cfg(base_model)
    if tmpl is not None:
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=tmpl)
        except Exception as e:
            print(f"[dpo] get_chat_template({tmpl}) -> native fallback: {e}")

    # ---- attach LoRA --------------------------------------------------------------
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

    # ---- continue from the SFT LoRA if provided -----------------------------------
    if sft_path and _os.path.isdir(sft_path):
        try:
            model.load_adapter(sft_path, adapter_name="default")
            print(f"[dpo] loaded SFT LoRA from {sft_path} (DPO continues the SFT policy)")
        except Exception as e:
            print(f"[dpo] could not load SFT adapter ({e}); DPO from base + fresh LoRA")
    else:
        print(f"[dpo] no SFT path; DPO from base + fresh LoRA (sft_path={sft_path!r})")

    # ---- preference dataset -------------------------------------------------------
    if not _os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"DPO preference file not found at {dataset_path}. Provide a JSONL of "
            f"{{prompt, chosen, rejected}} on a volume (e.g. /artifacts/dpo_pairs.jsonl) "
            f"or in train-mounted /root/data/."
        )
    pref_rows = _load_pref_jsonl(dataset_path)
    if not pref_rows:
        raise ValueError(f"No usable {{prompt,chosen,rejected}} rows in {dataset_path}")
    print(f"[dpo] loaded {len(pref_rows)} preference pairs")

    def _render_prompt(p):
        # p is either raw user text or a messages list; render to a chat-template string with
        # the assistant generation prompt so chosen/rejected complete the assistant turn.
        if isinstance(p, str):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": p}]
        else:
            msgs = list(p)
            if not any(m.get("role") == "system" for m in msgs):
                msgs = [{"role": "system", "content": SYSTEM}] + msgs
        msgs = _prep_messages(msgs, base_model)
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    ds = Dataset.from_list(
        [
            {
                "prompt": _render_prompt(r["prompt"]),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in pref_rows
        ]
    )

    # ---- DPO trainer --------------------------------------------------------------
    out_dir = f"/ckpts/dpo/{run_name}"
    cfg = DPOConfig(
        output_dir=out_dir,
        beta=beta,                                   # KL anchor to the SFT reference (~0.1)
        learning_rate=lr,                            # DPO wants a small lr (~5e-6)
        num_train_epochs=epochs,                     # 1-2 epochs (preference data is sharp)
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        max_length=max_seq,
        max_prompt_length=max_prompt_len,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )
    # Unsloth/PEFT DPO: pass ref_model=None so TRL uses the disabled-adapter base as the
    # implicit reference (the standard PEFT-DPO pattern; no second model copy needed).
    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=cfg,
            train_dataset=ds,
            processing_class=tokenizer,
        )
    except TypeError:
        # older TRL signature used `tokenizer=`
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=cfg,
            train_dataset=ds,
            tokenizer=tokenizer,
        )

    trainer.train()

    # ---- save LoRA + merged 16-bit ------------------------------------------------
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(f"/artifacts/{run_name}-dpo-lora")
    merged_dir = f"/artifacts/{run_name}-merged"
    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"[dpo] merged 16-bit -> {merged_dir}")
    except Exception as e:
        print(f"[dpo] merged save skipped ({e})")

    ckpts.commit()
    artifacts.commit()
    print(f"[dpo] DONE. LoRA -> {out_dir} ; merged -> {merged_dir}")
    return {"dpo_lora": out_dir, "merged": merged_dir}


# ==================================================================================
# 2) GRPO v2 — FIXED GRPO that cannot collapse the way the PoC did.
# ==================================================================================

@app.function(
    image=train_image,
    gpu="A10G",  # overridden per-base at call time when launching 7B
    timeout=10 * HOURS,
    volumes=VOLUMES,
    secrets=HF_SECRETS,
)
def grpo_v2(
    base_model: str = DEFAULT_BASE,
    sft_path: str = "/ckpts/sft/qwen-sft-v1",
    run_name: str = "qwen-grpo-v2",
    max_steps: int = 0,            # 0 -> auto: ~1 epoch over the large task set (no re-seeing)
    n_per_family: int = 12,        # curriculum size knob: instances per (band, family)
    lr: float = 1e-6,              # lower than the PoC's 1e-5 — stay near the SFT reference
    beta: float = 0.1,            # HIGHER KL than the PoC (~0.04 default) — anti-drift
    num_generations: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_prompt_len: int = 1024,
    max_completion_len: int = 1024,
    gpu_mem_util: float = 0.55,
    grad_accum: int = 4,
):
    """FIXED GRPO post-training against the proxy verifier — closes every hole that
    collapsed the PoC run (see eval/EVAL_REPORT.md).

    ================================ REWARD DESIGN ================================
    The PoC collapsed because it summed a validity reward with a *soft* format reward
    (weight 0.2). The policy hacked the lenient format term with repeated "op-ish" token
    spam (format ~1.0, validity ~0), completion length collapsed 320->60, KL drifted, and
    the model degenerated. THE FIX: ONE reward function, with format as a HARD parse-GATE
    INSIDE it — never a separately-weighted soft term that can be farmed.

    For each completion:
      1. HARD PARSE GATE. Run the proxy verifier (proxy_env.verify), which first parses +
         validates every op line against the EXACT-semantics DSL. If the stream does not
         parse/validate as a well-formed circuit, the verifier reports invalid. Format is
         therefore a *precondition* for ANY positive reward, not an additive bonus — there
         is no lenient format scalar to maximize independently of validity.
      2. VALIDITY-GATED REWARD. The single base reward is exactly proxy_env.reward's layered
         shaping over the verifier Report:
            invalid/parse-fail/over-width : ~ -1.0  (+ tiny parse_progress credit)
            reversible-but-wrong          : ~  0.0  (+ small frac_correct credit)
            correct-but-dirty ancilla     : ~  0.3
            FULLY VALID                   : ~  1.0 + cost_bonus  (cheaper-than-ref -> higher)
         Reward rises ONLY by producing circuits that are actually MORE correct / cheaper —
         the only axis we want optimized. There is no orthogonal term to exploit.
      3. ANTI-DEGENERATION PENALTY (added INSIDE the same reward, value <= 0). Punishes the
         exact degenerate shapes the PoC exhibited: too-few-distinct-ops, repeated-line spam,
         and below-min length (see _anti_degen_penalty). This makes the spam attractor that
         won last time strictly negative, so the policy cannot park on it even transiently.
    The reward is the SUM of (2) + (3): validity-driven gain minus degeneration penalty, as a
    SINGLE reward_func. No reward_weights, no second soft function -> no hackable surface.

    ============================== STABILITY KNOBS ===============================
      * HIGHER KL (beta ~0.1, vs the PoC's ~0.04 default): keeps the policy near the SFT
        reference so it cannot wander into the degenerate region.
      * LOWER lr (1e-6): smaller, safer GRPO steps.
      * ~1 EPOCH over a LARGE task set: max_steps is auto-computed from the curriculum size so
        every prompt is seen ~once and NO task is re-seen (the PoC re-saw 22 tasks x30 epochs,
        which is what let it overfit the hack). Grow the set with --n-per-family.
      * num_generations 8 (group size).
    """
    _require_heavy()
    import math
    import os as _os
    import random
    import sys

    _os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    _os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    _os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")

    if PROXY_REMOTE not in sys.path:
        sys.path.insert(0, PROXY_REMOTE)

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    import proxy_env   # faithful verifier (800/800 bit-identical to the Rust sim)
    import tasks        # curriculum + prompt rendering

    print(f"[grpo_v2] base={base_model} sft={sft_path} run={run_name}")

    # ---- model + rollout backend (family-aware) -----------------------------------
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
        print(f"[grpo_v2] {base_model}: vLLM unsupported -> fast_inference=False")
    model, tokenizer = FastLanguageModel.from_pretrained(**fp_kwargs)

    tmpl, _instr, _resp = _chat_cfg(base_model)
    if tmpl is not None:
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=tmpl)
        except Exception as e:
            print(f"[grpo_v2] get_chat_template({tmpl}) -> native fallback: {e}")

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

    if _os.path.isdir(sft_path):
        try:
            model.load_adapter(sft_path, adapter_name="default")
            print(f"[grpo_v2] loaded SFT LoRA from {sft_path}")
        except Exception as e:
            print(f"[grpo_v2] could not load SFT adapter ({e}); GRPO from base+fresh LoRA")
    else:
        print(f"[grpo_v2] no SFT path at {sft_path}; GRPO from base + fresh LoRA")

    # ---- LARGE curriculum (no re-seeing) ------------------------------------------
    # The PoC re-saw 22 tasks across 30 epochs. We build a much larger set with
    # n_per_family instances per (band, family) and size max_steps to ~1 pass over it.
    rng = random.Random(3407)
    curric = tasks.build_curriculum(rng=rng, n_per_family=n_per_family)
    instances = [inst for band in sorted(curric) for inst in curric[band]]
    if not instances:
        raise RuntimeError("tasks.build_curriculum() produced no instances")
    rng.shuffle(instances)
    print(f"[grpo_v2] curriculum: {len(instances)} task instances across "
          f"{len(curric)} bands (n_per_family={n_per_family})")

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

    # ---- THE SINGLE validity-gated reward (format = hard gate; + anti-degen) -------
    def _gated_reward(prompts=None, completions=None, **kw):
        ids = kw.get("task_id")
        specs = [instances[int(i)].task_spec for i in ids] if ids is not None else None
        # proxy_env.reward parses+validates (HARD gate) then returns the layered validity/cost
        # shaping. There is NO separate soft format term anywhere.
        base = proxy_env.reward(prompts=prompts, completions=completions, task_spec=specs)
        out = []
        for r, comp in zip(base, completions or []):
            text = proxy_env._extract_text(comp)
            out.append(float(r) + _anti_degen_penalty(text))
        return out

    reward_funcs = [_gated_reward]   # ONE function — no reward_weights, no hackable surface

    # ---- ~1 epoch sizing ----------------------------------------------------------
    # effective prompts/step = per_device_bs * grad_accum (one rollout group per prompt).
    per_device_bs = 1
    prompts_per_step = per_device_bs * grad_accum
    auto_steps = max(1, math.ceil(len(instances) / prompts_per_step))
    steps = max_steps if max_steps and max_steps > 0 else auto_steps
    print(f"[grpo_v2] ~1 epoch: {len(instances)} prompts / {prompts_per_step} per step "
          f"-> {auto_steps} steps (using max_steps={steps})")

    out_dir = f"/ckpts/grpo/{run_name}"
    grpo_kwargs = {}
    if use_vllm:
        grpo_kwargs.update(use_vllm=True, vllm_mode="colocate")
    else:
        grpo_kwargs.update(use_vllm=False)
    cfg = GRPOConfig(
        output_dir=out_dir,
        **grpo_kwargs,
        learning_rate=lr,
        beta=beta,                                   # HIGHER KL (anti-drift) — the key fix
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=steps,
        logging_steps=2,
        save_steps=max(50, steps // 4),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        # NOTE: deliberately NO reward_weights / NO second format reward_func — a single
        # validity-gated reward is the whole anti-hack design.
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(f"/artifacts/{run_name}-grpo-lora")
    ckpts.commit()
    artifacts.commit()
    print(f"[grpo_v2] DONE. GRPO-v2 LoRA -> {out_dir}")
    return {"grpo_v2_lora": out_dir, "steps": steps, "n_tasks": len(instances)}


# ==================================================================================
# Local entrypoint — dispatch subcommands (smoke / dpo / grpo_v2)
# ==================================================================================

@app.local_entrypoint()
def main(
    stage: str = "smoke",
    base_model: str = DEFAULT_BASE,
    run_name: str = "",
    dataset_path: str = "/root/data/dpo_pairs.jsonl",
    sft_path: str = "",
    gpu: str = "",
    epochs: int = 1,
    max_steps: int = 0,
    n_per_family: int = 12,
):
    """Dispatch a stable-training stage. Defaults to the cheap `smoke` light-path check.

    Examples:
        modal run train/dpo_app.py                                  # -> smoke
        ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py --stage dpo \
            --run-name qwen-dpo-v1 --dataset-path /artifacts/dpo_pairs.jsonl
        ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py --stage grpo_v2 \
            --base-model unsloth/Qwen2.5-Coder-7B-Instruct --run-name qwen7b-grpo-v2
    """
    stage = stage.lower()

    if _family(base_model) == "gemma":
        raise SystemExit(
            "Gemma-4 uses the vanilla HF+PEFT path in train/sft_gemma_standalone.py "
            "(its Gemma4ClippableLinear layers need target_modules=['linear']). This "
            "Unsloth DPO/GRPO app does not handle Gemma — run that file for the Gemma arm."
        )

    if stage in ("dpo", "grpo_v2") and not BUILD_HEAVY:
        raise SystemExit(
            f"stage '{stage}' needs the heavy CUDA/unsloth image. Relaunch with the env "
            f"var set, e.g.:\n    ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py "
            f"--stage {stage} --run-name <name>"
        )

    # GPU: explicit override wins, else auto from base size (7B -> L40S, else A10G).
    chosen_gpu = gpu or _gpu_for(base_model)

    if stage == "smoke":
        result = smoke.remote()
        print(f"[main] smoke result: {result}")
    elif stage == "dpo":
        rn = run_name or "qwen-dpo-v1"
        result = dpo.with_options(gpu=chosen_gpu).remote(
            base_model=base_model, run_name=rn,
            dataset_path=dataset_path, sft_path=sft_path, epochs=epochs,
        )
        print(f"[main] dpo result: {result}")
    elif stage == "grpo_v2":
        rn = run_name or "qwen-grpo-v2"
        result = grpo_v2.with_options(gpu=chosen_gpu).remote(
            base_model=base_model,
            sft_path=sft_path or "/ckpts/sft/qwen-sft-v1",
            run_name=rn, max_steps=max_steps, n_per_family=n_per_family,
        )
        print(f"[main] grpo_v2 result: {result}")
    else:
        raise SystemExit(
            f"unknown stage '{stage}' (expected: smoke | dpo | grpo_v2)"
        )
