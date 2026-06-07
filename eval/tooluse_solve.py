#!/usr/bin/env python3
"""tooluse_solve.py — drive an Ollama model through the STATE-EXTERNALIZING ToolEnv.

This is the test rig for the externalization hypothesis. Instead of asking the model to emit a
WHOLE circuit (which requires it to execute the synthesis algorithm and track cumulative state),
we put a TOOL between the model and the problem:

  * The tool (proxy/tooluse.ToolEnv) tracks the cumulative circuit state and, each turn, shows
    the model the TARGET vs the CURRENT effect vs the remaining MISMATCHES.
  * The model only has to do SINGLE-STEP reasoning: "given this current-vs-target, what ONE
    gate moves me closer?" It replies with exactly one op line. The tool applies it, recomputes
    the state, and shows the new mismatches. Repeat until mismatches -> 0 (and the circuit is a
    valid full-state bijection with clean ancillas + phase 0, i.e. proxy_env.verify valid=True).

The state is NEVER tracked by the model — that is the whole point.

Returns: {solved, opstream, steps, cost, transcript}.

Reuses eval_proxy's Ollama client (ollama_chat_messages from agentic_solve) + SYSTEM constant
is replaced here by a protocol-specific system prompt with one worked few-shot example.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ecdsa-model
sys.path.insert(0, os.path.join(BASE, "proxy"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # eval/

import proxy_env  # noqa: E402
from tooluse import ToolEnv  # noqa: E402
from eval_proxy import OLLAMA  # noqa: E402  (reuse the Ollama host)


# ----------------------------------------------------------------------------------
# Multi-turn Ollama chat helper (reused shape from agentic_solve, kept local so this file
# is self-contained for the experiment).
# ----------------------------------------------------------------------------------
def ollama_chat_messages(model, messages, temperature=0.2, seed=None, timeout=180,
                         num_predict=64):
    options = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None:
        options["seed"] = int(seed)
    payload = {"model": model, "messages": messages, "stream": False, "options": options}
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    return obj["message"]["content"]


# ----------------------------------------------------------------------------------
# Protocol system prompt: explain the one-gate-at-a-time tool loop + ONE worked example.
# ----------------------------------------------------------------------------------
TOOL_SYSTEM = """You synthesize a REVERSIBLE circuit ONE GATE AT A TIME. A TOOL tracks the
circuit state for you — you do NOT track it yourself.

Each turn the tool shows you:
  - TARGET: the function f the circuit must compute (truth table x->f(x)).
  - For linear (GF2) tasks, a RESIDUAL: each output bit written in terms of the ORIGINAL inputs,
    next to the target row. Rows marked "<-- WRONG" still differ.
  - EMITTED: the gates you have placed so far.
  - STATUS: Toffoli cost, peak width, ancilla/phase dirty flags.
  - MISMATCHES: the inputs x where the current circuit's output differs from f(x).

You reply with EXACTLY ONE op line. The tool applies it and shows you the new state. Goal:
drive MISMATCHES to 0 with the FEWEST Toffolis (CCX/CCZ are the only ops that cost; X/CX/SWAP
are free).

Op syntax (the LAST qubit token is the target):
  X q2            flip qubit 2
  CX q1 q2        qubit2 ^= qubit1          (free; for GF2: row2 ^= row1)
  CCX q0 q1 q2    qubit2 ^= qubit0 & qubit1 (costs 1 Toffoli)
  SWAP q0 q1      exchange qubits 0 and 1   (free)
You may also reply `UNDO` to drop your last gate.

REPLY FORMAT: output ONLY one op line (e.g. `CX q0 q2`) or `UNDO`. No prose, no backticks,
no explanation. Just the one line.

------- WORKED EXAMPLE (a 2-bit GF2 map solved in 2 CX steps) -------
TARGET: y0 = x0 ^ x1 ; y1 = x1 ^ x0    (i.e. swap-ish XOR map)
Turn 1 tool shows:
  current y0 = x0          | target y0 = x0 ^ x1   <-- WRONG
  current y1 = x1          | target y1 = x0 ^ x1   <-- WRONG
You reason (silently): row0 needs x1 added -> CX q1 q0. You reply:
CX q1 q0
Turn 2 tool shows:
  current y0 = x0 ^ x1     | target y0 = x0 ^ x1
  current y1 = x1          | target y1 = x0 ^ x1   <-- WRONG
You reason: row1 needs x0 added. But x0 now lives where? q0 currently = x0^x1, q1 = x1, so
adding q0 then q1 gives x0. Simpler: row1 needs x0; add the original x0. Reply:
CX q0 q1
Now both rows match and the tool reports CIRCUIT IS COMPLETE.
--------------------------------------------------------------------

Always pick the single op that fixes one WRONG row / reduces the mismatch list. Reply with one
op line now."""


# Extract one op line (or UNDO) from a possibly-chatty model reply.
_OP_HEAD = re.compile(
    r"^\s*(X|Z|CX|CZ|CCX|CCZ|SWAP|NEG)\b", re.IGNORECASE)


def extract_one_op(text: str):
    """Pull the FIRST plausible op line (or UNDO) from the model's reply."""
    # strip code fences
    t = text
    fenced = re.findall(r"```[a-zA-Z0-9_]*\n(.*?)```", t, re.S)
    if fenced:
        t = fenced[0]
    for raw in t.replace(";", "\n").splitlines():
        s = raw.strip().strip("`").strip()
        if not s:
            continue
        if s.upper() == "UNDO":
            return "UNDO"
        # accept lines like "CX q0 q2" possibly with trailing comment/prose
        m = _OP_HEAD.match(s)
        if m:
            # keep only the op + qubit tokens (drop trailing prose)
            toks = s.split()
            kept = [toks[0].upper()]
            for tk in toks[1:]:
                if tk.startswith("q") and tk[1:].isdigit():
                    kept.append(tk)
                elif tk.startswith("b") and tk[1:].isdigit():
                    kept.append(tk)
                elif tk in ("if",):
                    kept.append(tk)
                else:
                    break
            return " ".join(kept)
    return ""  # nothing op-like found


# ----------------------------------------------------------------------------------
# The main driver.
# ----------------------------------------------------------------------------------
def tooluse_solve(model, task_spec, prompt=None, max_steps=40, temperature=0.2,
                  seed=4242, verbose=False, max_retries_per_step=2):
    """Drive `model` through ToolEnv(task_spec) one gate at a time.

    Args:
      model      Ollama model tag.
      task_spec  proxy_env task_spec dict (optionally carrying 'family' for gf2 residual).
      prompt     unused for the tool protocol except as optional extra task context (the tool's
                 render() already contains the full target); kept for signature parity.
      max_steps  max number of ACCEPTED gates (the budget; loop also bounded by 2*max_steps turns).
      temperature, seed   sampling controls.
      max_retries_per_step  on a parse/validate error, how many times to re-ask before counting
                 the turn as a wasted step.

    Returns: {solved, opstream, steps, cost, peak_width, toffoli, transcript}.
    """
    env = ToolEnv(task_spec)
    transcript = []

    # If already trivially solved (empty circuit is correct), short-circuit.
    if env.done:
        return {"solved": True, "opstream": env.solved_opstream(), "steps": 0,
                "cost": env.cost, "peak_width": env.peak_width, "toffoli": 0.0,
                "transcript": transcript}

    messages = [{"role": "system", "content": TOOL_SYSTEM}]
    # First user turn: a short framing + the initial tool render.
    intro = ("Here is the task. Reply with ONE op line each turn to drive mismatches to 0.\n\n"
             + env.render())
    messages.append({"role": "user", "content": intro})

    steps = 0           # accepted gates
    turn_budget = max_steps * 2 + 4
    turn = 0
    last_n_mismatch = env.n_mismatch
    no_progress = 0     # consecutive accepted steps that did not reduce mismatches
    # Oscillation guard: remember recently-visited circuit states (by their residual signature)
    # so we can reject an op that just returns us to a state we already left — the 1.5B model's
    # dominant failure mode is a 2-cycle. The tool still tracks state; we only refuse to let the
    # model burn its whole budget toggling.
    def _state_sig():
        outs, _ph, _pw, _ew = env._current_outputs()
        return tuple(outs)
    visited = {_state_sig(): 0}

    while steps < max_steps and turn < turn_budget and not env.done:
        turn += 1
        # vary seed per turn so retries/loops don't repeat verbatim
        try:
            reply = ollama_chat_messages(model, messages, temperature=temperature,
                                         seed=seed + turn)
        except Exception as e:
            reply = f"__error__ {e}"
        op = extract_one_op(reply)
        transcript.append({"turn": turn, "kind": "model", "raw": reply.strip()[:200],
                           "op": op})
        if verbose:
            print(f"  [turn {turn}] model -> {op!r}")

        if op == "":
            # nothing op-like; nudge and continue (counts toward turn budget, not steps)
            messages.append({"role": "assistant", "content": reply.strip()[:200]})
            messages.append({"role": "user", "content":
                             "I could not find an op line in your reply. Reply with EXACTLY one "
                             "op line like `CX q0 q2` or `CCX q0 q1 q2`, nothing else.\n\n"
                             + env.render()})
            continue

        if op == "UNDO":
            res = env.undo()
            messages.append({"role": "assistant", "content": "UNDO"})
            messages.append({"role": "user", "content":
                             ("Undone.\n\n" if not res["error"] else res["error"] + "\n\n")
                             + res["render"] + "\n\nReply with ONE op line."})
            transcript.append({"turn": turn, "kind": "tool", "applied": "UNDO",
                               "n_mismatch": res["n_mismatch"], "done": res["done"]})
            continue

        # try to apply the op
        res = env.step(op)
        messages.append({"role": "assistant", "content": op})
        if res["error"]:
            # parse/validate/width error -> show error, let it retry (does NOT consume a step)
            transcript.append({"turn": turn, "kind": "tool", "applied": op,
                               "error": res["error"], "n_mismatch": res["n_mismatch"],
                               "done": res["done"]})
            messages.append({"role": "user", "content":
                             f"REJECTED: {res['error']}\nThe circuit is unchanged. Try a "
                             f"different single op line.\n\n" + res["render"]
                             + "\n\nReply with ONE op line."})
            if verbose:
                print(f"            REJECTED: {res['error'][:60]}")
            continue

        # Oscillation guard: if this op returned us to an already-visited state (and we're not
        # done), undo it and tell the model to try a DIFFERENT op. Keeps the budget productive.
        sig = _state_sig()
        if not res["done"] and sig in visited and len(env.emitted) > visited[sig]:
            env.undo()
            transcript.append({"turn": turn, "kind": "tool", "applied": op,
                               "rejected_cycle": True, "n_mismatch": env.n_mismatch,
                               "done": False})
            messages[-1] = {"role": "assistant", "content": op}
            messages.append({"role": "user", "content":
                             f"That op ({op}) just returns the circuit to a state you already "
                             f"visited (it undoes recent progress). I reverted it. Pick a "
                             f"DIFFERENT op that fixes the FIRST `<-- WRONG` row.\n\n"
                             + env.render() + "\n\nReply with ONE op line."})
            if verbose:
                print(f"            CYCLE rejected ({op}); reverted.")
            no_progress += 1
            if no_progress >= 8:
                break
            continue

        # accepted gate
        steps += 1
        visited[sig] = len(env.emitted)
        if res["n_mismatch"] >= last_n_mismatch:
            no_progress += 1
        else:
            no_progress = 0
        last_n_mismatch = res["n_mismatch"]
        transcript.append({"turn": turn, "kind": "tool", "applied": op,
                           "n_mismatch": res["n_mismatch"], "cost": res["cost"],
                           "peak_width": res["peak_width"], "done": res["done"]})
        if verbose:
            print(f"            applied. n_mismatch={res['n_mismatch']} "
                  f"cost={res['cost']} done={res['done']}")

        if res["done"]:
            break

        # progress nudge if the model is stuck XORing the same rows back and forth
        nudge = ""
        if no_progress >= 4:
            nudge = ("\nNOTE: the mismatch count has not dropped in several steps. Look at the "
                     "FIRST `<-- WRONG` row (or first mismatched input) and pick the op that "
                     "fixes exactly that one. You may UNDO a step that made things worse.")
        messages.append({"role": "user", "content": res["render"] + nudge
                         + "\n\nReply with ONE op line."})

        # keep the message list from growing unbounded: retain system + last ~8 turns
        if len(messages) > 18:
            messages = [messages[0]] + messages[-16:]

    solved = env.done
    return {
        "solved": bool(solved),
        "opstream": env.solved_opstream(),
        "steps": steps,
        "cost": env.cost if solved else None,
        "peak_width": env.peak_width,
        "toffoli": (env._cache["toffoli"] if env._cache else None),
        "n_mismatch": env.n_mismatch,
        "transcript": transcript,
    }


if __name__ == "__main__":
    import argparse
    import random
    import tasks  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ecdsa-coder-1.5b-v4")
    ap.add_argument("--family", default="gf2_linear")
    ap.add_argument("--band", default="B1")
    ap.add_argument("--seed", type=int, default=60000)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.2)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    inst = tasks.sample_instance(a.family, a.band, rng)
    spec = dict(inst.task_spec)
    spec["family"] = inst.family
    print(f"[smoke] {inst.family} {inst.band} ref_cost={inst.task_spec['reference_cost']:.0f} "
          f"params={inst.params}")
    res = tooluse_solve(a.model, spec, inst.prompt, max_steps=a.max_steps,
                        temperature=a.temperature, verbose=True)
    print(f"[smoke] solved={res['solved']} steps={res['steps']} cost={res['cost']}")
    print("opstream:\n" + res["opstream"])
