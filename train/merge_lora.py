"""Merge a LoRA adapter into its base -> a clean fp16 HF model (vanilla HF+PEFT, no Unsloth).

Used to serve the GRPO model (and bypass Unsloth's flaky save_pretrained_gguf). The merged
dir downloads + `ollama create`s exactly like the SFT model.

  ECDSA_BUILD_HEAVY=1 modal run train/merge_lora.py \
      --base-model /artifacts/qwen-sft-v1-merged \
      --adapter /ckpts/grpo/qwen-grpo-v1 \
      --out-name qwen-grpo-v1-merged
"""
import os
from pathlib import Path
import modal

app = modal.App("ecdsa-merge")
HOURS = 3600
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
ckpts = modal.Volume.from_name("ecdsa-ckpts", create_if_missing=True)
artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/ckpts": ckpts, "/artifacts": artifacts}
BUILD_HEAVY = os.environ.get("ECDSA_BUILD_HEAVY", "0") == "1"

img = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
       .entrypoint([]).apt_install("git")
       .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("transformers", "peft", "accelerate", "safetensors", "huggingface_hub[hf_transfer]")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "ECDSA_BUILD_HEAVY": "1"})) if BUILD_HEAVY \
    else modal.Image.from_registry("python:3.12-slim")


@app.function(image=img, gpu="A10G", timeout=2 * HOURS, volumes=VOLUMES)
def merge(base_model: str, adapter: str, out_name: str):
    if not BUILD_HEAVY:
        raise RuntimeError("Relaunch with ECDSA_BUILD_HEAVY=1.")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"[merge] base={base_model} adapter={adapter} -> {out_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out = f"/artifacts/{out_name}"
    merged.save_pretrained(out, safe_serialization=True)
    # tokenizer: prefer the adapter's, fall back to base
    try:
        AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    except Exception:
        AutoTokenizer.from_pretrained(base_model).save_pretrained(out)
    artifacts.commit()
    print(f"[merge] DONE -> {out}")
    return {"merged": out}


@app.local_entrypoint()
def main(base_model: str, adapter: str, out_name: str):
    print(merge.remote(base_model=base_model, adapter=adapter, out_name=out_name))
