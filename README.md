# reversible-circuit-llm — a verifier-grounded specialist for reversible-circuit optimization

Training a small, open-source model to do the *kind of work* the
[ECDSA.fail](https://ecdsa.fail) secp256k1 point-addition challenge demands:
**verifier-guided, cost-minimizing synthesis/optimization of reversible quantum circuits under
hard correctness constraints** (cost = average Toffoli count × peak qubit width; lower is better;
every circuit must be classically correct, reversible, phase-clean, and ancilla-clean).

The thesis: with a **microsecond-exact verifier**, this is not ordinary fine-tuning — it's
**search + self-improvement with an LLM as the policy**. The repo is the machinery for that.

## Headline result
Held-out reversible-circuit **synthesis** (the model must emit a circuit that passes all four
validity gates), `valid_rate` = fraction solved with best-of-16 (the verifier is a free inference
oracle):

| model | held-out valid_rate | mean reward |
|---|---|---|
| base Qwen2.5-Coder-1.5B | **0%** (writes Python) | −1.00 |
| v1 SFT (bloated MMD targets) | **0%** | −0.59 |
| **v4 SFT (24.5k optimal targets + bug fix)** | **4.8%** (solves the easiest band) | **+0.07** |

**Key finding:** a 7B trained identically scores the *same* 4.8% → this is **not a capacity problem**.
Reversible synthesis of unseen tasks is *algorithmic*; pure imitation plateaus at the easiest band at
every scale. The levers that target the harder bands — **reasoning/long-CoT** and **RL** — are
implemented here (`proxy/reason_gen.py`, `train/dpo_app.py::grpo_v2`).

What drove the jump from the PoC:
1. **A verifier-as-search data factory** (`proxy/synth.py`): training targets at **0.54× the Toffoli
   cost** of the textbook (MMD) references — the PoC was training the model to imitate bloat.
2. **A fundamental bug fix**: the S-box / GF(2) task prompts omitted their truth-table/matrix, making
   those tasks *underspecified* (unlearnable, contradictory targets) and capping diversity at ~2,300.
   Fixing it unlocked **31,718 distinct tasks (14×)**.
3. **Best-of-N with the verifier oracle** — the honest deployment mode.

## What's here
- **`proxy/`** — the environment. `proxy_env.py` is a faithful proxy verifier, **bit-identical
  (800/800)** to the real Rust simulator (vendored in `proxy_rs/`). `tasks.py` is the task curriculum
  (7 families × bands B0–B6 + held-out generalization). `synth.py` is the verifier-gated search engine
  (greedy-delete + peephole + simulated annealing + IDA*-optimal for tiny n). `reason_gen.py` produces
  verified algorithmic chain-of-thought traces.
- **`data/`** — dataset builders (`build_sft.py`, generators) + the mined optimization-move corpus
  (`moves_raw.jsonl`, 264 accepted ECDSA.fail submissions). Large generated `.jsonl` are git-ignored
  (regenerate via the scripts / Modal fan-out `train/gen_curriculum.py`).
- **`train/`** — Modal training apps: `modal_app.py` (SFT), `dpo_app.py` (DPO + collapse-proof
  validity-gated GRPO), `gen_curriculum.py` (CPU fan-out optimal-target generation), `eval_modal.py`
  (GPU-side eval), `sft_gemma_standalone.py` (Gemma-4 arm).
- **`eval/`** — `eval_proxy.py` (held-out synthesis, best-of-N), `agentic_solve.py` (verifier-feedback
  repair loop), `eval_cfg.py` (real-challenge move quality), `run_harness.py` (the real ECDSA.fail
  harness wrapper). Full results in **`eval/EVAL_REPORT.md`**.
- **`recon/`** — the reconnaissance that grounded everything (harness semantics, the ~105 DIALOG knobs,
  mined reasoning episodes, proxy-env design, training-stack analysis).
- **`PLAN.md`**, **`artifacts/MODEL_CARD.md`** — plan and model card.

## Reproduce (Modal; the verifier being CPU-cheap is what makes this affordable)
```bash
python3 data/build_sft.py                                   # base SFT set from mined corpus
ECDSA_BUILD_HEAVY=1 modal run train/gen_curriculum.py --n-seeds 800 --n-per-family 8   # 30k optimal targets
ECDSA_BUILD_HEAVY=1 modal run train/modal_app.py --stage sft --dataset-path /root/data/sft_v4f.jsonl ...
ECDSA_BUILD_HEAVY=1 modal run train/dpo_app.py   --stage grpo_v2 --sft-path /ckpts/sft/qwen-sft-v4 ...
ECDSA_BUILD_HEAVY=1 modal run train/eval_modal.py --model-dir /artifacts/<merged>   # best-of-N valid_rate
```
See `train/README.md` and `eval/EVAL_REPORT.md` for the full pipeline and numbers.

## License
Apache-2.0. Base models used are Apache-2.0 (Qwen2.5-Coder, Gemma-4). The vendored
`proxy_rs/{circuit,sim}.rs` derive from the ECDSA.fail challenge harness (see its NOTICE).

🤖 Built autonomously with Claude Code.
