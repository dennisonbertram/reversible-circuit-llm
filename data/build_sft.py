#!/usr/bin/env python3
"""Deterministic SFT dataset builder for the ECDSA-fail reversible-circuit specialist.

Three sources, all built WITHOUT any LLM generation (robust, reproducible):
  1. proxy synthesis  : (task prompt -> a VALID reference op-stream) from proxy/tasks.py
                        teaches the harness op-stream DSL + reversible synthesis (GRPO warm-start).
  2. real-challenge moves : (current DIALOG knob state -> next bounded move) from data/moves_raw.jsonl
                        teaches the (tighten knob)+(re-find Fiat-Shamir nonce) atomic move pattern.
  3. reasoning episodes : (situation -> Tony/Anton audit) from recon/reasoning_episodes.md
                        teaches the verifier-guided, smallest-bounded-change METHOD.

Output: data/sft_train.jsonl, data/sft_val.jsonl (chat format {"messages":[...]}),
        per-source files, and data/DATASET.md.
"""
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))            # .../ecdsa-model/data
BASE = os.path.dirname(ROOT)                                  # .../ecdsa-model
sys.path.insert(0, os.path.join(BASE, "proxy"))
import tasks  # noqa: E402

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

MAX_OPSTREAM_LINES = 140   # cap proxy-synthesis targets so SFT stays within sequence budget

examples = []  # list of (source, messages)

# ----------------------------------------------------------------------------- 1) proxy synthesis
def opstream_lines(s):
    return [ln for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("#")]

per_band_target = {"B0": 120, "B1": 120, "B2": 120, "B3": 110, "B4": 90, "B5": 60, "B6": 20}
per_band = {b: [] for b in per_band_target}
seen_prompts = set()

n_seeds = 90
for seed in range(n_seeds):
    cur = tasks.build_curriculum(rng=random.Random(1000 + seed))
    for band, insts in cur.items():
        if band not in per_band_target:
            continue
        if len(per_band[band]) >= per_band_target[band]:
            continue
        for inst in insts:
            ops = inst.reference_ops
            if not isinstance(ops, str) or not ops.strip():
                continue
            if len(opstream_lines(ops)) > MAX_OPSTREAM_LINES:
                continue
            key = inst.prompt
            if key in seen_prompts:
                continue
            seen_prompts.add(key)
            per_band[band].append(inst)
            assistant = ops.strip()
            examples.append((
                "proxy_synth",
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": inst.prompt},
                 {"role": "assistant", "content": assistant}],
            ))
            if len(per_band[band]) >= per_band_target[band]:
                break

n_proxy = sum(len(v) for v in per_band.values())
print(f"[proxy_synth] {n_proxy} examples; per-band:", {b: len(v) for b, v in per_band.items()})

# ----------------------------------------------------------------------------- 2) real-challenge moves
MOVES = os.path.join(ROOT, "moves_raw.jsonl")
KNOB_RE = re.compile(r'set_default_env\(\s*"([A-Z0-9_]+)"\s*,\s*"([^"]+)"\s*\)')
SCORE_RE = re.compile(r'score[^0-9]{0,12}([0-9][0-9,]{6,})')
SKIP_KNOB = re.compile(r'DUMMY|TRACE|DEBUG|STOP_AFTER')

def move_type(knob):
    if "TAIL_NONCE" in knob or "REROLL" in knob:
        return "Fiat-Shamir tail-nonce island selection (zero score effect; correctness selector)"
    if "COMPARE_BITS" in knob:
        return "tighten comparator bits (cuts executed Toffoli; op-stream-altering)"
    if "WIDTH" in knob:
        return "width-envelope tune (trades peak qubits vs convergence support)"
    if "CARRY" in knob or "TRUNC" in knob:
        return "carry-truncation widen/trim (cuts Toffoli but risks hard-input correctness)"
    if "ACTIVE_ITERATIONS" in knob or "ITERATION" in knob:
        return "GCD active-iteration count (fewer steps = less Toffoli AND lower peak, risks non-convergence)"
    if "HOST" in knob or "BORROW" in knob or "SCRATCH" in knob:
        return "value-exact scratch/carry hosting (reuse a provably-|0> idle qubit; no new failure mode)"
    if "FUSE" in knob or "FUSED" in knob or "DEDUP" in knob or "COMPRESS" in knob:
        return "Toffoli fusion / dedup lever"
    return "bounded DIALOG knob adjustment"

moves_added = 0
nonce_added = 0
seen_moves = set()
for line in open(MOVES):
    r = json.loads(line)
    if not r.get("is_small_move") or not r.get("small_diff"):
        continue
    diff = r["small_diff"]
    removed = {k: v for k, v in KNOB_RE.findall("\n".join(
        ln[1:] for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")))}
    added = {k: v for k, v in KNOB_RE.findall("\n".join(
        ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")))}
    score_hits = SCORE_RE.findall(diff)
    score_txt = (" Observed score after the validated move: " + score_hits[-1] + ".") if score_hits else ""
    for knob, new in added.items():
        if SKIP_KNOB.search(knob):
            continue
        old = removed.get(knob)
        if old is None or old == new:
            continue
        mtype = move_type(knob)
        sig = (knob, old, new)
        if sig in seen_moves:
            continue
        seen_moves.add(sig)
        if "TAIL_NONCE" in knob or "REROLL" in knob:
            user = ("We just altered the emitted op-stream of the secp256k1 dialog-GCD point-add and "
                    "the previous Fiat-Shamir input island no longer validates (classical mismatches / "
                    "phase batches on some of the 9024 shots). The relevant selector is "
                    f'`set_default_env("{knob}", "{old}")`. What is the next move?')
            assistant = (
                f'Move: {knob} {old} -> {new}. This is {mtype}. Nonce/reroll changes have ZERO effect on '
                "the score (Toffoli x qubits) — they only select a clean input island so all 9024 Fiat-Shamir "
                "shots validate. Use the classical GCD pre-filter to find a survivor nonce cheaply, set it, then "
                "re-validate 0 classical / 0 phase / 0 ancilla over all 9024 shots. If many pre-filtered "
                "survivors still die at full validation, the underlying change is STRUCTURAL (a real value "
                f"error), not island-limited — revert it rather than grinding nonces.{score_txt}")
            nonce_added += 1
        else:
            user = ("Frontier circuit: the secp256k1 dialog-GCD point-add, scored by avg executed Toffoli x "
                    f'peak qubits (lower is better). Current setting `set_default_env("{knob}", "{old}")`. '
                    "Propose the next single bounded optimization move and the validation plan.")
            assistant = (
                f'Move (smallest bounded change): {knob} {old} -> {new}. Type: {mtype}. Expected effect: lowers '
                "the executed Toffoli count (or peak width) while the four validity gates must still hold. "
                "Because this changes the emitted op-stream, the prior Fiat-Shamir input island is voided — "
                "re-find a clean DIALOG_TAIL_NONCE (and, if needed, retune DIALOG_REROLL / "
                "DIALOG_POST_SUB_REROLL) and re-validate 0 classical / 0 phase / 0 ancilla over all 9024 shots "
                f"before submitting. If hard inputs break, back off toward the previous value.{score_txt}")
            moves_added += 1
        examples.append((
            "moves",
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user},
             {"role": "assistant", "content": assistant}],
        ))

print(f"[moves] {moves_added} knob-moves + {nonce_added} nonce-island examples")

# ----------------------------------------------------------------------------- 3) reasoning episodes
EP = os.path.join(BASE, "recon", "reasoning_episodes.md")
text = open(EP).read()

# constant harness context block
ctx_m = re.search(r"## Problem / Harness context.*?\n(.*?)\n# ERA", text, re.S)
CTX = ctx_m.group(1).strip() if ctx_m else ""
CTX_SHORT = "\n".join(CTX.splitlines()[:16])

def split_blocks(prefix):
    # returns list of (title, body) for headers '## <prefix>...'
    pat = re.compile(r"^## (" + prefix + r".*?)\n(.*?)(?=^## |\Z)", re.S | re.M)
    return [(m.group(1).strip(), m.group(2).strip()) for m in pat.finditer(text)]

episodes = split_blocks(r"E\d+\.")
patterns = split_blocks(r"Pattern ")

rea_added = 0
for title, body in episodes:
    # strip the outcome tag in parens from the title for the "candidate idea"
    idea = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    idea = re.sub(r"^E\d+\.\s*", "", idea)
    user = (f"Harness context:\n{CTX_SHORT}\n\nCandidate idea / situation: {idea}\n\n"
            "Audit this in the smallest-bounded-change style: cite the exact lever, quantify the expected "
            "Toffoli / qubit / phase impact, classify the risk (structural value-error vs Fiat-Shamir "
            "tail-nonce island vs convergence/width floor vs measurement noise), and give the single "
            "smallest next step (or say why it is a dead end).")
    examples.append(("reasoning",
                     [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user},
                      {"role": "assistant", "content": body}]))
    rea_added += 1
    # a second framing keyed on the Lesson, when present
    lesson_m = re.search(r"\*\*Lesson\*\*:\s*(.*)", body, re.S)
    valid_m = re.search(r"\*\*Validation\*\*:\s*(.*?)(?:\n- |\Z)", body, re.S)
    if lesson_m and valid_m:
        user2 = (f"During the grind we tried: {idea}. The validation came back: "
                 f"{valid_m.group(1).strip()[:300]} What is the generalizable lesson, and does this move "
                 "belong in the frontier or should we move on?")
        assistant2 = "Lesson: " + lesson_m.group(1).strip()
        examples.append(("reasoning",
                         [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user2},
                          {"role": "assistant", "content": assistant2}]))
        rea_added += 1

pat_clean = re.compile(r"^Pattern [A-E]\s*[—-]\s*")
for title, body in patterns:
    method_name = pat_clean.sub("", title)
    user = "Describe your working method: " + method_name + "."
    examples.append(("reasoning",
                     [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user},
                      {"role": "assistant", "content": body}]))
    rea_added += 1

print(f"[reasoning] {rea_added} examples ({len(episodes)} episodes, {len(patterns)} patterns)")

# ----------------------------------------------------------------------------- assemble + split
# dedup by (user, assistant)
uniq = []
seen = set()
for src, msgs in examples:
    k = (msgs[1]["content"], msgs[2]["content"])
    if k in seen:
        continue
    seen.add(k)
    uniq.append((src, msgs))

rng = random.Random(42)
# stratified 90/10 split by source
by_src = {}
for src, msgs in uniq:
    by_src.setdefault(src, []).append(msgs)
train, val = [], []
for src, items in by_src.items():
    rng.shuffle(items)
    n_val = max(1, round(len(items) * 0.10))
    val += [(src, m) for m in items[:n_val]]
    train += [(src, m) for m in items[n_val:]]
rng.shuffle(train)
rng.shuffle(val)

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for src, msgs in rows:
            f.write(json.dumps({"messages": msgs}) + "\n")

write_jsonl(os.path.join(ROOT, "sft_train.jsonl"), train)
write_jsonl(os.path.join(ROOT, "sft_val.jsonl"), val)
# per-source for inspection
for src in by_src:
    write_jsonl(os.path.join(ROOT, f"sft_{src}.jsonl"),
                [(s, m) for s, m in uniq if s == src])

# validate every line parses + has the 3-role structure
bad = 0
for p in ("sft_train.jsonl", "sft_val.jsonl"):
    for ln in open(os.path.join(ROOT, p)):
        try:
            o = json.loads(ln)
            assert [m["role"] for m in o["messages"]] == ["system", "user", "assistant"]
            assert all(m["content"].strip() for m in o["messages"])
        except Exception as e:
            bad += 1
            print("BAD LINE:", e)

counts = {src: len(v) for src, v in by_src.items()}
print(f"\nTOTAL unique: {len(uniq)} | train {len(train)} | val {len(val)} | bad {bad}")
print("by source:", counts)

with open(os.path.join(ROOT, "DATASET.md"), "w") as f:
    f.write("# ECDSA-fail Specialist — SFT Dataset\n\n")
    f.write("Built deterministically by `build_sft.py` (no LLM generation). Chat format: each line is "
            "`{\"messages\":[{system},{user},{assistant}]}`.\n\n")
    f.write(f"- Total unique examples: **{len(uniq)}** (train {len(train)} / val {len(val)})\n")
    f.write("- Sources:\n")
    for src, n in counts.items():
        f.write(f"  - `{src}`: {n}\n")
    f.write("\n## Sources\n")
    f.write("1. **proxy_synth** — (task prompt -> valid reference op-stream) from `proxy/tasks.py` "
            "curriculum (bands B0-B6, 7 families). Teaches the harness op-stream DSL + reversible "
            f"synthesis. Op-streams capped at {MAX_OPSTREAM_LINES} lines.\n")
    f.write("2. **moves** — (current DIALOG knob -> next bounded move + validation plan) from the 146 "
            "small high-signal accepted-submission diffs in `data/moves_raw.jsonl` (dummy/trace knobs "
            "filtered). Teaches the (tighten knob)+(re-find Fiat-Shamir nonce) atomic move.\n")
    f.write("3. **reasoning** — (situation -> Tony/Anton smallest-bounded-change audit) from the 27 "
            "episodes + 5 patterns in `recon/reasoning_episodes.md`. Teaches the transferable method.\n")
    f.write("\nShared system prompt establishes the reversible-circuit-optimization specialist persona "
            "and the validity-first cost metric (Toffoli x peak width).\n")

print("wrote sft_train.jsonl, sft_val.jsonl, per-source files, DATASET.md")
