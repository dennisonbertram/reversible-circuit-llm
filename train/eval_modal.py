"""Evaluate a merged HF model on held-out proxy tasks ON MODAL (avoids huge local downloads).

best-of-N generation per held-out task, scored by the faithful proxy verifier (mounted).
Reports valid_rate per band — the deployment-realistic metric (verifier as inference oracle).

  ECDSA_BUILD_HEAVY=1 modal run train/eval_modal.py --model-dir /artifacts/qwen7b-sft-v4-merged \
      --per-band 3 --n-samples 16
"""
import os
from pathlib import Path
import modal

app = modal.App("ecdsa-eval")
HERE = Path(__file__).parent.resolve()
PROXY_DIR = (HERE.parent / "proxy").resolve()
BUILD_HEAVY = os.environ.get("ECDSA_BUILD_HEAVY", "0") == "1"

img = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
       .entrypoint([]).apt_install("git")
       .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("transformers", "accelerate", "numpy", "safetensors")
       .env({"ECDSA_BUILD_HEAVY": "1"})
       .add_local_dir(str(PROXY_DIR), remote_path="/root/proxy")) if BUILD_HEAVY \
    else modal.Image.from_registry("python:3.12-slim")

artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def _extract(text):
    import re
    KINDS = {"X", "Z", "CX", "CZ", "CCX", "CCZ", "SWAP", "R", "HMR", "NEG", "BIT_INVERT",
             "BIT_STORE0", "BIT_STORE1", "REGISTER", "APPEND_TO_REGISTER", "PUSH_CONDITION",
             "POP_CONDITION", "DEBUG_PRINT"}
    m = re.findall(r"```[a-zA-Z0-9_]*\n(.*?)```", text, re.S)
    body = (m[0] if m else text).replace(";", "\n")
    return "\n".join(ln.strip() for ln in body.splitlines()
                     if ln.strip() and not ln.strip().startswith("#")
                     and ln.strip().split()[0].upper() in KINDS)


@app.function(image=img, gpu="L40S", timeout=3600, volumes={"/artifacts": artifacts,
              "/root/.cache/huggingface": hf_cache})
def eval_model(model_dir: str, per_band: int = 3, n_samples: int = 16, temperature: float = 0.8):
    if not BUILD_HEAVY:
        raise RuntimeError("Relaunch with ECDSA_BUILD_HEAVY=1.")
    import sys
    import random
    from collections import defaultdict
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/root/proxy")
    import proxy_env
    import tasks
    SYSTEM = open("/root/proxy/system_prompt.txt").read().strip()

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # held-out instances: unseen seeds 60000+ per band + the generalization tasks
    insts = []
    per = defaultdict(int)
    s = 60000
    while s < 60400 and any(per[b] < per_band for b in tasks.BANDS):
        cur = tasks.build_curriculum(rng=random.Random(s))
        for band, lst in cur.items():
            for it in lst:
                if per[band] < per_band:
                    per[band] += 1
                    insts.append((band, it))
        s += 1
    for fn in ("build_heldout_fma", "build_heldout_sbox6"):
        try:
            insts.append(("HELDOUT", getattr(tasks, fn)(rng=random.Random(7))))
        except Exception:
            pass

    by_band = defaultdict(lambda: {"n": 0, "valid": 0})
    overall = {"n": 0, "valid": 0}
    for band, inst in insts:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": inst.prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=400, do_sample=True, temperature=temperature,
                                 top_p=0.95, num_return_sequences=n_samples, pad_token_id=tok.pad_token_id)
        best_valid = False
        for o in out:
            txt = tok.decode(o[plen:], skip_special_tokens=True)
            ops = _extract(txt)
            try:
                v = proxy_env.verify(ops, inst.task_spec)
            except Exception:
                v = {"valid": False}
            if v.get("valid"):
                best_valid = True
                break
        by_band[band]["n"] += 1
        by_band[band]["valid"] += int(best_valid)
        overall["n"] += 1
        overall["valid"] += int(best_valid)

    result = {"model_dir": model_dir, "n_samples": n_samples,
              "overall_valid_rate": round(overall["valid"] / max(1, overall["n"]), 3),
              "overall": overall,
              "by_band": {b: {**v, "valid_rate": round(v["valid"] / max(1, v["n"]), 3)}
                          for b, v in sorted(by_band.items())}}
    print("EVAL_RESULT", result)
    return result


@app.local_entrypoint()
def main(model_dir: str = "/artifacts/qwen7b-sft-v4-merged", per_band: int = 3,
         n_samples: int = 16, temperature: float = 0.8):
    print(eval_model.remote(model_dir=model_dir, per_band=per_band,
                            n_samples=n_samples, temperature=temperature))
