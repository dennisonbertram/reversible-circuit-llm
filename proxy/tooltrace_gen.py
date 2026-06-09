#!/usr/bin/env python3
"""
tooltrace_gen.py — EXPERT TOOL-USE trace generator for the state-externalizing ToolEnv.

Why this exists
---------------
`proxy/tooluse.py` (ToolEnv) is a tool that tracks the cumulative reversible-circuit state and
lets the model pick ONE gate per turn. Zero-shot, a 1.5B model thrashes: it cannot plan the
multi-step synthesis sequence and oscillates. The FIX in this file is to manufacture SFT data of
CORRECT single-step play, so the model learns the per-step policy ("given current-vs-target, emit
the one gate that fixes the first wrong row") and the hard multi-step synthesis decomposes into
single imitable steps.

What a trace is
---------------
For each sampled task we obtain an EXPERT op-stream:
  * gf2_linear -> the Gauss-Jordan CX/SWAP factoring from reason_gen.decompose_gf2 (pure, short).
  * other families -> synth.synthesize (a correct, often optimal op-stream).
Then we REPLAY that op-stream gate-by-gate through a FRESH ToolEnv, recording a multi-turn chat
trace whose framing matches eval/tooluse_solve.py EXACTLY so the SFT model sees the same thing at
inference time:
  messages[0] = {"role":"system",  "content": TOOL_SYSTEM}          # verbatim from tooluse_solve
  messages[1] = {"role":"user",    "content": INTRO + env.render()} # the first turn's framing
  then per expert gate g_t (state BEFORE the gate):
     {"role":"assistant","content": g_t}                            # EXACTLY one op line
     {"role":"user",     "content": env.render() + REPLY_SUFFIX}    # next state, driver framing
The trailing user turn (after the last gate) is dropped: the final assistant op completes the
circuit, so there is nothing more to ask. We CONFIRM env.done is True before keeping the task.

Matching tooluse_solve's framing (so train == eval):
  - system prompt   = tooluse_solve.TOOL_SYSTEM (imported, not copied).
  - first user turn = "Here is the task. Reply with ONE op line each turn to drive mismatches
                       to 0.\n\n" + render()        (tooluse_solve's `intro`).
  - later user turns = render() + "\n\nReply with ONE op line."  (tooluse_solve's per-turn user
                       message; we omit the dynamic no-progress NUDGE, which only fires when the
                       live model stalls — expert play never stalls).
  - assistant turns  = the single op line, EXACTLY as ToolEnv emitted/parsed it (so the driver's
                       extract_one_op recovers it identically).

Output: data/sft_tooltrace.jsonl — one JSON object per task = one full conversation
{"messages":[...]}. A companion data/sft_tooltrace.meta.jsonl records family/band/turns for
analysis (not needed to train).

Run:  /usr/bin/python3 tooltrace_gen.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EVAL_DIR = os.path.join(os.path.dirname(HERE), "eval")
sys.path.insert(0, EVAL_DIR)

import proxy_env as pe  # noqa: E402
import tasks  # noqa: E402
import synth  # noqa: E402
from tooluse import ToolEnv  # noqa: E402
from reason_gen import decompose_gf2  # noqa: E402

# Pull the EXACT protocol strings from the eval driver so train == eval. tooluse_solve imports
# eval_proxy (which may require an Ollama host constant) at module load; that import is harmless
# here (it only reads a host string), so we import the driver directly to reuse TOOL_SYSTEM and
# extract_one_op verbatim. If that import fails for any reason, fall back to a vendored copy that
# is kept character-identical (asserted at startup).
try:
    from tooluse_solve import TOOL_SYSTEM, extract_one_op  # noqa: E402
    _HAVE_DRIVER = True
except Exception as _e:  # pragma: no cover - defensive; the driver should import fine
    _HAVE_DRIVER = False
    _DRIVER_IMPORT_ERR = _e


# The first user turn's framing (tooluse_solve.tooluse_solve builds this as `intro`).
INTRO_PREFIX = ("Here is the task. Reply with ONE op line each turn to drive mismatches to 0.\n\n")
# The per-turn user-message suffix (tooluse_solve appends this to res["render"]).
REPLY_SUFFIX = "\n\nReply with ONE op line."

DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
OUT_PATH = os.path.join(DATA_DIR, "sft_tooltrace.jsonl")
META_PATH = os.path.join(DATA_DIR, "sft_tooltrace.meta.jsonl")

# Skip tasks whose expert op-stream is longer than this (render + history would blow context).
MAX_GATES = 40

# Training seeds are DISJOINT from eval (>=60000) and build_heldout_* fixed seeds.
_SEED_LO = 1
_SEED_HI = 50000


# ----------------------------------------------------------------------------------
# Expert op-stream sources -> a clean list of single-op lines.
# ----------------------------------------------------------------------------------
def _opstream_lines(text: str) -> List[str]:
    """Split an op-stream into non-empty, non-comment op lines (one op each)."""
    out: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def expert_gates_gf2(inst: tasks.Instance) -> Optional[List[str]]:
    """Gauss-Jordan CX/SWAP factoring (pure, 0 Toffoli) from reason_gen — the bulk source."""
    n = inst.params["n"]
    M = inst.params["M"]
    try:
        gates, _narr = decompose_gf2(M, n, compact=False)
    except ValueError:
        return None
    if not gates:
        return None  # identity map -> nothing to learn
    lines: List[str] = []
    for kind, a, b in gates:
        if kind == "CX":
            lines.append(f"CX q{a} q{b}")
        else:
            lines.append(f"SWAP q{a} q{b}")
    return lines


def expert_gates_synth(inst: tasks.Instance, seed: int,
                       time_budget_s: float = 1.5) -> Optional[List[str]]:
    """A correct (often optimal) op-stream from synth.synthesize for non-gf2 families."""
    res = synth.synthesize(inst.task_spec, inst.reference_ops,
                           time_budget_s=time_budget_s, seed=seed)
    if not res.get("valid"):
        return None
    lines = _opstream_lines(res["opstream"])
    return lines or None


# ----------------------------------------------------------------------------------
# Replay an expert op-stream through a fresh ToolEnv, recording the multi-turn chat trace.
# ----------------------------------------------------------------------------------
def build_trace(inst: tasks.Instance, gates: List[str]) -> Optional[Dict]:
    """Replay `gates` through ToolEnv(inst.task_spec + family) and emit a chat-format trace.

    Returns {"messages": [...], "meta": {...}} or None if the stream errors, is too long, an
    assistant turn is not a single parseable op line, or the final state is not done==True.
    """
    if not gates or len(gates) > MAX_GATES:
        return None

    spec = dict(inst.task_spec)
    spec["family"] = inst.family  # enables the gf2 RESIDUAL view (matches eval)
    env = ToolEnv(spec)

    # Skip trivially-solved tasks (empty circuit already correct): no single-step lesson there.
    if env.done:
        return None

    messages: List[Dict[str, str]] = [{"role": "system", "content": TOOL_SYSTEM}]
    first_user = INTRO_PREFIX + env.render()

    for idx, g in enumerate(gates):
        # The user turn shows the state BEFORE this gate. First turn uses the intro framing.
        if idx == 0:
            messages.append({"role": "user", "content": first_user})
        else:
            messages.append({"role": "user", "content": env.render() + REPLY_SUFFIX})

        # Assistant emits EXACTLY one op line. It must:
        #  (1) be exactly the single op (no prose), and
        #  (2) survive the driver's extract_one_op -> the SAME token the tool would step on.
        if "\n" in g or ";" in g:
            return None  # not a single op line
        if _HAVE_DRIVER:
            parsed = extract_one_op(g)
            if parsed != g:
                # The driver would re-tokenize/normalize this op (e.g. casing); skip to keep the
                # assistant target byte-identical to what inference parses. Expert ops from
                # reason_gen / synth are already canonical, so this practically never triggers.
                return None
        messages.append({"role": "assistant", "content": g})

        # Apply the gate; it must be accepted (expert ops are always valid for this env).
        res = env.step(g)
        if res["error"]:
            return None
        # After the final gate, do NOT append another user turn (nothing left to ask). Earlier
        # gates' next-state user turn is emitted at the top of the next loop iteration.

    if not env.done:
        return None

    meta = {
        "family": inst.family,
        "band": inst.band,
        "n_gates": len(gates),
        "n_turns": len(gates),               # one assistant op per turn
        "n_messages": len(messages),
        "cost": env.cost,
        "toffoli": (env._cache["toffoli"] if env._cache else None),
        "peak_width": env.peak_width,
        "params": dict(inst.params),
    }
    return {"messages": messages, "meta": meta}


# ----------------------------------------------------------------------------------
# Instance samplers (training-namespace RNG; DISJOINT from eval seeds >= 60000).
# ----------------------------------------------------------------------------------
def _gf2_instance(n: int, band: str, rng: random.Random) -> Optional[tasks.Instance]:
    """Build a gf2_linear Instance at width n (skip the degenerate identity matrix)."""
    M = tasks._rand_invertible_gf2(n, rng)
    if M == [1 << i for i in range(n)]:
        return None
    inst = tasks.build_gf2_linear(n, M)
    inst.band = band
    inst.prompt = tasks.render_prompt(inst)
    return inst


# gf2_linear bands -> register width (mirrors reason_gen's gf2_width).
_GF2_WIDTH = {"B1": 3, "B2": 4, "B3": 5, "B4": 6}

# Per-(family, band) quotas for KEPT traces. gf2_linear is the bulk (unbounded random invertible
# matrices). Arithmetic families: only the bands whose synth op-streams stay <= MAX_GATES are
# worth including (see the survey in tooltrace_gen's build notes):
#   const_add B1 (median 6.5), B2 (median 18.5)  -> good, short.
#   controlled_addsub B1 (median 12)             -> good, short.
#   reg_add B2 (median ~384) and ctl_addsub B4 (median ~105) are EXCLUDED (too long).
#
# Measured throughput (per trace, incl. the MMD-ref build inside build_gf2_linear):
#   B1 n=3 ~0.7ms, B2 n=4 ~2.4ms, B3 n=5 ~7.7ms, B4 n=6 ~39ms. gf2 is the cheap, unbounded
#   bulk (random invertible matrices), so we place the mass at n=4/n=5 and a solid tail at n=6.
_GF2_QUOTA: Dict[str, int] = {
    "B1": 150,     # n=3: ~|GL(3,2)|-1 = 167 reachable distinct -> near the cap (~0.1s)
    "B2": 5000,    # n=4: |GL(4,2)| = 20160 distinct; cheap -> the bulk (~12s)
    "B3": 4000,    # n=5: huge distinct space; cheap (~31s)
    "B4": 2000,    # n=6: huge distinct space; ~39ms/trace (~78s)
}
# Arithmetic families have SMALL distinct spaces per band (measured): const_add B1 ~10 distinct,
# const_add B2 ~22, controlled_addsub B1 ~20. Quotas are set at/above those so the slot fills
# its whole reachable space and then stops on no-progress (rather than over-quota'ing).
_ARITH_QUOTA: Dict[Tuple[str, str], int] = {
    ("const_add", "B1"): 12,
    ("const_add", "B2"): 24,
    ("controlled_addsub", "B1"): 24,
}
# synth time budget per arithmetic instance (short streams; 0.5s is plenty and bounds the
# worst-case slot time to no-progress-limit * 0.5s).
_SYNTH_BUDGET_S = 0.5

# Stop a slot after this many consecutive samples add nothing new (distinct space exhausted, or
# every candidate exceeds MAX_GATES). The arithmetic distinct spaces are tiny, so this caps the
# wasted synth calls once a slot saturates. gf2 bulk slots hit quota long before this.
_NO_PROGRESS_LIMIT = 200


# ----------------------------------------------------------------------------------
# Dataset builder.
# ----------------------------------------------------------------------------------
def build_dataset(verbose: bool = True) -> Dict:
    assert _HAVE_DRIVER, (
        f"could not import tooluse_solve (TOOL_SYSTEM/extract_one_op): {_DRIVER_IMPORT_ERR!r}. "
        "The traces MUST reuse the eval driver's exact system prompt; refusing to proceed.")
    os.makedirs(DATA_DIR, exist_ok=True)

    kept: List[Dict] = []
    per_fb: Dict[Tuple[str, str], int] = {}
    seen_prompt: set = set()

    def add(inst: tasks.Instance, gates: Optional[List[str]]) -> bool:
        fam, band = inst.family, inst.band
        if inst.prompt in seen_prompt:
            return False
        if gates is None:
            return False
        tr = build_trace(inst, gates)
        if tr is None:
            return False
        seen_prompt.add(inst.prompt)
        per_fb[(fam, band)] = per_fb.get((fam, band), 0) + 1
        kept.append(tr)
        return True

    # ---- gf2_linear (bulk): sample random invertible matrices directly per band/width. ----
    for i, band in enumerate(["B1", "B2", "B3", "B4"]):
        quota = _GF2_QUOTA[band]
        n = _GF2_WIDTH[band]
        wrng = random.Random(4200 + i)
        no_progress = 0
        while per_fb.get(("gf2_linear", band), 0) < quota and no_progress < _NO_PROGRESS_LIMIT:
            inst = _gf2_instance(n, band, wrng)
            ok = False
            if inst is not None:
                ok = add(inst, expert_gates_gf2(inst))
            no_progress = 0 if ok else no_progress + 1
        if verbose:
            print(f"  filled gf2_linear/{band} (n={n}): "
                  f"{per_fb.get(('gf2_linear', band), 0)}/{quota}", flush=True)

    # ---- arithmetic families: sample via the curriculum, expert from synth. ----
    import zlib
    for (fam, band), quota in _ARITH_QUOTA.items():
        # deterministic per-slot training seed cursor (stable across runs; disjoint from eval
        # seeds >= 60000 since it lives in [_SEED_LO, _SEED_LO+997)).
        seed = _SEED_LO + (zlib.adler32(f"{fam}/{band}".encode()) % 997)
        no_progress = 0
        while per_fb.get((fam, band), 0) < quota and no_progress < _NO_PROGRESS_LIMIT \
                and seed < _SEED_HI:
            rng = random.Random(seed)
            cur_seed = seed
            seed += 1
            try:
                inst = tasks.sample_instance(fam, band, rng)
            except Exception:
                no_progress += 1
                continue
            ok = add(inst, expert_gates_synth(inst, seed=cur_seed,
                                              time_budget_s=_SYNTH_BUDGET_S))
            no_progress = 0 if ok else no_progress + 1
        if verbose:
            print(f"  filled {fam}/{band}: {per_fb.get((fam, band), 0)}/{quota}", flush=True)

    # write the dataset (chat format) + companion meta.
    with open(OUT_PATH, "w") as fh:
        for row in kept:
            fh.write(json.dumps({"messages": row["messages"]}) + "\n")
    with open(META_PATH, "w") as fh:
        for row in kept:
            fh.write(json.dumps(row["meta"]) + "\n")

    return {
        "out_path": OUT_PATH,
        "meta_path": META_PATH,
        "total": len(kept),
        "per_family_band": {f"{f}/{b}": c for (f, b), c in sorted(per_fb.items())},
        "kept": kept,
    }


# ----------------------------------------------------------------------------------
# Validation: re-drive a fresh ToolEnv from the recorded assistant gates and confirm done==True.
# ----------------------------------------------------------------------------------
def _rebuild_spec(meta: dict) -> dict:
    """Reconstruct the EXACT verifying task_spec (+ family flag) from a meta record."""
    fam = meta["family"]
    p = meta["params"]
    if fam == "gf2_linear":
        spec = dict(tasks.build_gf2_linear(p["n"], p["M"]).task_spec)
    elif fam == "const_add":
        spec = dict(tasks.build_const_add(p["n"], p["m"], p["a"]).task_spec)
    elif fam == "controlled_addsub":
        spec = dict(tasks.build_controlled_addsub(p["n"], p["m"], p["a"], sub=p["sub"]).task_spec)
    elif fam == "reg_add":
        spec = dict(tasks.build_reg_add(p["n"], p["m"]).task_spec)
    else:
        raise ValueError(f"cannot rebuild spec for family {fam}")
    spec["family"] = fam
    return spec


def revalidate(k: int = 300, seed: int = 17) -> dict:
    """Re-drive a random sample of written traces: feed the recorded assistant ops in order to a
    FRESH ToolEnv and confirm it reaches done==True. Also confirm the system prompt matches the
    eval driver's and every assistant turn is a single parseable op line."""
    data_lines = open(OUT_PATH).read().splitlines()
    meta_lines = open(META_PATH).read().splitlines()
    assert len(data_lines) == len(meta_lines), "data/meta length mismatch"
    rng = random.Random(seed)
    idxs = rng.sample(range(len(data_lines)), min(k, len(data_lines)))

    passed = 0
    sys_ok = 0
    single_op_ok = 0
    failures: List[dict] = []
    for i in idxs:
        row = json.loads(data_lines[i])
        meta = json.loads(meta_lines[i])
        msgs = row["messages"]

        # (a) system prompt identical to the eval driver's TOOL_SYSTEM.
        if msgs[0]["role"] == "system" and msgs[0]["content"] == TOOL_SYSTEM:
            sys_ok += 1

        # (b) every assistant turn is a single op line the driver parses to itself.
        ops = [m["content"] for m in msgs if m["role"] == "assistant"]
        ok_single = all(
            ("\n" not in op and ";" not in op and extract_one_op(op) == op) for op in ops)
        if ok_single:
            single_op_ok += 1

        # (c) re-drive a fresh ToolEnv with those ops -> done.
        spec = _rebuild_spec(meta)
        env = ToolEnv(spec)
        err = None
        for op in ops:
            r = env.step(op)
            if r["error"]:
                err = r["error"]
                break
        if err is None and env.done:
            passed += 1
        else:
            failures.append({"idx": i, "family": meta["family"], "band": meta["band"],
                             "err": err, "done": env.done})
    return {
        "checked": len(idxs),
        "passed": passed,
        "fraction": passed / len(idxs) if idxs else 0.0,
        "sys_prompt_match": sys_ok,
        "single_op_turns": single_op_ok,
        "failures": failures,
    }


def _median(xs: List[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def one_full_sample(seed: int = 7) -> dict:
    """Return one complete trace (prefer a small gf2_linear) for display."""
    data_lines = open(OUT_PATH).read().splitlines()
    meta_lines = open(META_PATH).read().splitlines()
    rng = random.Random(seed)
    order = list(range(len(data_lines)))
    rng.shuffle(order)
    pick = None
    for i in order:
        meta = json.loads(meta_lines[i])
        if meta["family"] == "gf2_linear" and meta["params"]["n"] in (3, 4) \
                and 2 <= meta["n_gates"] <= 5:
            pick = i
            break
    if pick is None:
        pick = order[0]
    return {"row": json.loads(data_lines[pick]), "meta": json.loads(meta_lines[pick])}


def report(result: Dict) -> None:
    print("=" * 76)
    print("TOOL-USE EXPERT TRACE BUILD REPORT  (data/sft_tooltrace.jsonl)")
    print("=" * 76)
    print(f"file            : {result['out_path']}")
    print(f"meta            : {result['meta_path']}")
    print(f"total traces    : {result['total']}")

    # per family/band + by-family/by-band rollups
    by_fam: Dict[str, int] = {}
    by_band: Dict[str, int] = {}
    print("per family/band :")
    for key, c in result["per_family_band"].items():
        fam, band = key.split("/")
        by_fam[fam] = by_fam.get(fam, 0) + c
        by_band[band] = by_band.get(band, 0) + c
        print(f"    {key:28s} {c}")
    print("by family       :", by_fam)
    print("by band         :", by_band)

    # median #turns (over all kept)
    turns = [row["meta"]["n_turns"] for row in result["kept"]]
    print(f"median #turns   : {_median(turns):.1f}  (min {min(turns) if turns else 0}, "
          f"max {max(turns) if turns else 0})")

    # ---- 100% re-drive validation on a random 300 ----
    print("-" * 76)
    rv = revalidate(k=300, seed=17)
    print(f"RE-DRIVE VALIDATION (random {rv['checked']}): "
          f"{rv['passed']}/{rv['checked']} reach done==True "
          f"({100.0 * rv['fraction']:.1f}%)")
    print(f"  system-prompt matches eval driver : {rv['sys_prompt_match']}/{rv['checked']}")
    print(f"  every assistant turn single op    : {rv['single_op_turns']}/{rv['checked']}")
    if rv["failures"]:
        print("  FAILURES:", rv["failures"][:10])

    # ---- one full sample trace ----
    print("=" * 76)
    print("ONE FULL SAMPLE TRACE (system + first user + a few user/assistant turns + final)")
    print("=" * 76)
    s = one_full_sample(seed=7)
    msgs = s["row"]["messages"]
    print(f"[meta] {s['meta']}")
    print("\n----- messages[0] SYSTEM -----")
    print(msgs[0]["content"])
    print("\n----- messages[1] USER (first turn: intro + render) -----")
    print(msgs[1]["content"])
    # then assistant/user turns
    for j in range(2, len(msgs)):
        m = msgs[j]
        if m["role"] == "assistant":
            print(f"\n----- messages[{j}] ASSISTANT (one op line) -----")
            print(m["content"])
        else:
            print(f"\n----- messages[{j}] USER (next state + reply suffix) -----")
            print(m["content"])
    print("\n(final assistant op completes the circuit; no trailing user turn.)")


if __name__ == "__main__":
    res = build_dataset()
    report(res)
