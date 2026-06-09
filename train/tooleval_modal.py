"""Evaluate a merged model DRIVING the ToolEnv (multi-turn) on Modal — avoids huge local downloads.

For each held-out task: run ToolEnv; each turn render the state, have the model emit ONE gate,
step it, until done or max_steps. Reports tool-use valid_rate per band. Framing matches
tooltrace_gen / tooluse_solve so train == eval.

  ECDSA_BUILD_HEAVY=1 modal run train/tooleval_modal.py --model-dir /artifacts/qwen3-8b-tool-v1-merged \
      --per-band 4 --max-steps 40
"""
import os
from pathlib import Path
import modal

app = modal.App("ecdsa-tooleval")
HERE = Path(__file__).parent.resolve()
PROXY_DIR = (HERE.parent / "proxy").resolve()
EVAL_DIR = (HERE.parent / "eval").resolve()
BUILD_HEAVY = os.environ.get("ECDSA_BUILD_HEAVY", "0") == "1"

img = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
       .entrypoint([]).apt_install("git")
       .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu129")
       .pip_install("transformers", "accelerate", "numpy", "safetensors")
       .env({"ECDSA_BUILD_HEAVY": "1", "OLLAMA_HOST": "http://127.0.0.1:11434"})
       .add_local_dir(str(PROXY_DIR), remote_path="/root/proxy")
       .add_local_dir(str(EVAL_DIR), remote_path="/root/eval")) if BUILD_HEAVY \
    else modal.Image.from_registry("python:3.12-slim")

artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(image=img, gpu="L40S", timeout=10800,
              volumes={"/artifacts": artifacts, "/root/.cache/huggingface": hf_cache})
def tooleval(model_dir: str, per_band: int = 4, max_steps: int = 40, n_restarts: int = 2,
             temperature: float = 0.4):
    if not BUILD_HEAVY:
        raise RuntimeError("Relaunch with ECDSA_BUILD_HEAVY=1.")
    import sys, random
    from collections import defaultdict
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, "/root/proxy"); sys.path.insert(0, "/root/eval")
    import tasks
    from tooluse import ToolEnv
    # protocol framing (match training); vendor to avoid Ollama import chain
    try:
        from tooluse_solve import TOOL_SYSTEM, extract_one_op
    except Exception:
        from tooltrace_gen import TOOL_SYSTEM  # type: ignore
        import re
        def extract_one_op(t):
            for ln in t.splitlines():
                s = ln.strip()
                if s and s.split()[0].upper() in {"X","Z","CX","CZ","CCX","CCZ","SWAP"}:
                    return s
            return t.strip().splitlines()[0].strip() if t.strip() else ""
    INTRO = "Here is the task. Reply with ONE op line each turn to drive mismatches to 0.\n\n"
    SUFFIX = "\n\nReply with ONE op line."

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # held-out gf2 instances at the test bands (B1 n=3, B2 n=4, B3-as-5bit, B4 n=6)
    insts = []
    for band, n in [("B1", 3), ("B2", 4), ("B3", 5), ("B4", 6)]:
        c = 0; s = 60000
        while c < per_band and s < 60400:
            rng = random.Random(s); s += 1
            try:
                M = tasks._rand_invertible_gf2(n, rng)
                inst = tasks.build_gf2_linear(n, M); inst.band = band
                inst.prompt = tasks.render_prompt(inst)
                insts.append((band, inst)); c += 1
            except Exception:
                pass

    def gen_gate(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=temperature,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    by_band = defaultdict(lambda: {"n": 0, "solved": 0})
    for band, inst in insts:
        spec = dict(inst.task_spec); spec["family"] = inst.family
        solved = False
        for _r in range(n_restarts):
            env = ToolEnv(spec)
            if env.done:
                solved = True; break
            msgs = [{"role": "system", "content": TOOL_SYSTEM},
                    {"role": "user", "content": INTRO + env.render()}]
            for _step in range(max_steps):
                raw = gen_gate(msgs)
                gate = extract_one_op(raw)
                msgs.append({"role": "assistant", "content": gate})
                res = env.step(gate)
                if res.get("done"):
                    solved = True; break
                msgs.append({"role": "user", "content": env.render() + SUFFIX})
                # keep context bounded: drop oldest user/assistant pair if too many turns
                if len(msgs) > 24:
                    msgs = [msgs[0]] + msgs[-22:]
            if solved:
                break
        by_band[band]["n"] += 1
        by_band[band]["solved"] += int(solved)

    res = {"model_dir": model_dir, "max_steps": max_steps,
           "by_band": {b: {**v, "solve_rate": round(v["solved"]/max(1, v["n"]), 3)}
                       for b, v in sorted(by_band.items())},
           "overall_solve_rate": round(sum(v["solved"] for v in by_band.values())
                                       / max(1, sum(v["n"] for v in by_band.values())), 3)}
    print("TOOLEVAL_RESULT", res)
    return res


@app.local_entrypoint()
def main(model_dir: str = "/artifacts/qwen3-8b-tool-v1-merged", per_band: int = 4,
         max_steps: int = 40, n_restarts: int = 2, temperature: float = 0.4):
    print(tooleval.remote(model_dir=model_dir, per_band=per_band, max_steps=max_steps,
                          n_restarts=n_restarts, temperature=temperature))
