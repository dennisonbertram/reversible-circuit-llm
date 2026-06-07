"""Standalone Gemma-4 SFT on Modal using VANILLA HuggingFace + PEFT + TRL + bitsandbytes
(NO Unsloth — its pinned 2026.1.4 predates Gemma-4, and latest unsloth_zoo conflicts with
latest trl). This is the robust path; slightly slower than Unsloth but conflict-free.

  ECDSA_BUILD_HEAVY=1 modal run train/sft_gemma_standalone.py --base-model unsloth/gemma-4-E4B-it --run-name gemma4-e4b-sft-v1

Saves a merged 16-bit model to /artifacts/<run_name>-merged (download + `ollama create`).
"""
import os
from pathlib import Path
import modal

app = modal.App("ecdsa-gemma")
HOURS = 3600
HERE = Path(__file__).parent.resolve()
DATA_DIR = (HERE.parent / "data").resolve()

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpts = modal.Volume.from_name("ecdsa-ckpts", create_if_missing=True)
artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/ckpts": ckpts, "/artifacts": artifacts}

BUILD_HEAVY = os.environ.get("ECDSA_BUILD_HEAVY", "0") == "1"


def _gemma_image():
    return (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .apt_install("git", "build-essential", "cmake", "curl")
        .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu129")
        .pip_install(
            # vanilla HF stack — latest (mutually compatible, Gemma-4-capable), no unsloth.
            # torch stays pinned to the cu129 wheel above; HF libs are co-released compatible.
            "transformers", "trl", "peft", "datasets", "accelerate",
            "bitsandbytes", "huggingface_hub[hf_transfer]",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
              "ECDSA_BUILD_HEAVY": "1"})
        .add_local_dir(str(DATA_DIR), remote_path="/root/data")
    )


gemma_image = _gemma_image() if BUILD_HEAVY else modal.Image.from_registry("python:3.12-slim")


def _hf_secrets():
    try:
        s = modal.Secret.from_name("huggingface"); s.hydrate(); return [s]
    except Exception:
        return []


@app.function(image=gemma_image, gpu="L40S", timeout=3 * HOURS, volumes=VOLUMES, secrets=_hf_secrets())
def sft_gemma(base_model: str = "unsloth/gemma-4-E4B-it",
              run_name: str = "gemma4-e4b-sft-v1",
              dataset_path: str = "/root/data/sft_train.jsonl",
              epochs: int = 2, lr: float = 2e-4, lora_r: int = 16, lora_alpha: int = 32,
              max_seq_len: int = 2048):
    if not BUILD_HEAVY:
        raise RuntimeError("Relaunch with ECDSA_BUILD_HEAVY=1 to build the Gemma image.")
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"[gemma-sft] base={base_model} run={run_name} epochs={epochs}")
    tok = AutoTokenizer.from_pretrained(base_model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    model = prepare_model_for_kbit_training(model)
    # Gemma-4 wraps each proj in a custom Gemma4ClippableLinear(.linear=Linear4bit); stock PEFT
    # can't target the wrapper but CAN target the inner `.linear` (a Linear4bit it supports).
    lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["linear"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def _fold(messages):  # Gemma has no system role -> fold into first user turn
        out, sys_text = [], None
        for m in messages:
            if m["role"] == "system":
                sys_text = m["content"]; continue
            if sys_text and m["role"] == "user":
                m = {"role": "user", "content": sys_text + "\n\n" + m["content"]}; sys_text = None
            out.append(m)
        if sys_text:
            out.insert(0, {"role": "user", "content": sys_text})
        return out

    rows = [json.loads(l) for l in open(dataset_path) if l.strip()]
    texts = [tok.apply_chat_template(_fold(r["messages"]), tokenize=False, add_generation_prompt=False)
             for r in rows if "messages" in r]
    print(f"[gemma-sft] {len(texts)} examples")
    ds = Dataset.from_dict({"text": texts})

    out_dir = f"/ckpts/sft/{run_name}"
    cfg = SFTConfig(output_dir=out_dir, per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    warmup_ratio=0.05, num_train_epochs=epochs, learning_rate=lr,
                    lr_scheduler_type="cosine", optim="paged_adamw_8bit", weight_decay=0.01,
                    bf16=True, max_grad_norm=1.0, logging_steps=5, save_strategy="no",
                    report_to="none", dataset_text_field="text", max_length=max_seq_len, packing=False)
    try:
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
    trainer.train()

    adapter_dir = f"{out_dir}/adapter"
    trainer.save_model(adapter_dir); tok.save_pretrained(adapter_dir)
    ckpts.commit()
    print("[gemma-sft] adapter saved; merging to fp16 ...")

    # reload base in bf16 (no quant) and merge the adapter for a clean servable model
    del model, trainer
    torch.cuda.empty_cache()
    base_fp = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    merged = PeftModel.from_pretrained(base_fp, adapter_dir).merge_and_unload()
    merged_dir = f"/artifacts/{run_name}-merged"
    merged.save_pretrained(merged_dir, safe_serialization=True); tok.save_pretrained(merged_dir)
    artifacts.commit()
    print(f"[gemma-sft] DONE. merged -> {merged_dir}")
    return {"merged": merged_dir}


@app.function(image=gemma_image, gpu="L40S", timeout=1800, volumes=VOLUMES)
def gen(model_dir: str = "/artifacts/gemma4-e4b-sft-v1-merged"):
    """Quick cross-family sanity: does the SFT'd Gemma emit op-streams + sensible knob moves?"""
    if not BUILD_HEAVY:
        raise RuntimeError("Relaunch with ECDSA_BUILD_HEAVY=1.")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    SYS = ("You are a reversible-circuit optimization specialist. Emit op-streams (X/CX/CCX/SWAP) "
           "and propose bounded DIALOG_* knob moves for the secp256k1 challenge.")
    prompts = [
        "TASK family=const_add band=B0 n_in=2. f(x)=(x+1) mod 3 in place on qubits [0,1], ancilla [2,3]. Emit ONLY op lines.",
        'Frontier: secp256k1 dialog-GCD point-add. Current set_default_env("DIALOG_GCD_COMPARE_BITS", "74"). Propose the next bounded move + validation plan.',
    ]
    for p in prompts:
        msgs = [{"role": "user", "content": SYS + "\n\n" + p}]  # gemma: fold system into user
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=200, do_sample=False)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n=== PROMPT ===\n", p[:90], "\n--- GEMMA-SFT OUTPUT ---\n", txt[:500])
    return "ok"


@app.local_entrypoint()
def main(base_model: str = "unsloth/gemma-4-E4B-it", run_name: str = "gemma4-e4b-sft-v1",
         dataset_path: str = "/root/data/sft_train.jsonl", epochs: int = 2):
    print(sft_gemma.remote(base_model=base_model, run_name=run_name,
                           dataset_path=dataset_path, epochs=epochs))
