#!/usr/bin/env python3
"""agentic_solve.py — VERIFIER-IN-THE-LOOP inference harness for the proxy reversible-circuit env.

The capability multiplier: instead of one shot, let the model ITERATE against real verifier
feedback (proxy_env.verify — bit-identical to the real ECDSA-fail simulator). This flips the
held-out valid_rate off zero WITHOUT retraining.

Strategy (agentic_solve):
  Turn 0:  sample n_samples completions (varying temperature + seed), extract op-streams,
           verify each. If any are VALID, keep the cheapest (lowest cost) and stop.
  If none valid: take the BEST partial attempt, ranked by the verifier's own progress signals
           (frac_correct, then frac_ancilla_clean, then parse_progress). Build a SPECIFIC
           feedback message from verify()'s reason + witness:
             * parse/validate error      -> "emit ONLY op lines; line N failed: <err>"
             * reset/HMR banned           -> "this band forbids R/HMR; use X/CX/CCX/SWAP only"
             * peak_width over cap         -> "peak_width P > cap C; reuse ancillas, drop qubit"
             * not reversible (collision)  -> "inputs a and b both map to <out>; the map must
                                              be a bijection — that pair of ops collides"
             * classical_mismatch          -> "wrong on input x: got G, want W (per declared
                                              output qubits); fix the bits that differ"
             * ancilla_garbage             -> "ancilla q# left dirty on some input; uncompute it
                                              back to |0> (mirror the compute in reverse)"
             * phase_garbage               -> "global phase != 0 on some input; remove the
                                              CZ/CCZ/Z/NEG that injects phase or pair it"
             * forward_reverse_identity    -> "ops not self-inverse as written; check operand
                                              order / aliasing"
             * cost > ref (valid case)     -> handled as a SUCCESS but the feedback path can ask
                                              for a strictly cheaper revision when desired.
           Append the model's last attempt + that feedback as a NEW user turn, re-sample, verify.
           Loop up to max_turns; keep the best-ever VALID circuit (cheapest), else best partial.

Returns: {best_opstream, valid, cost, turns, transcript}.

Reuses the Ollama /api/chat client, extract_opstream, and SYSTEM prompt from eval_proxy.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ecdsa-model
sys.path.insert(0, os.path.join(BASE, "proxy"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # eval/

import proxy_env  # noqa: E402

# Reuse the model client, op-stream extractor, and system prompt from the one-shot eval.
from eval_proxy import extract_opstream, SYSTEM, OLLAMA  # noqa: E402


# ----------------------------------------------------------------------------------
# Ollama chat with full message history (multi-turn) + per-call seed/temperature.
# ----------------------------------------------------------------------------------
def ollama_chat_messages(model, messages, temperature=0.2, seed=None, timeout=180):
    """Single /api/chat call over an explicit message list (system + dialog turns)."""
    options = {"temperature": temperature, "num_predict": 1024}
    if seed is not None:
        options["seed"] = int(seed)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read())
    return obj["message"]["content"]


# ----------------------------------------------------------------------------------
# Candidate scoring: rank partial (non-valid) attempts by the verifier's progress signals.
# Higher is better. Valid candidates are ranked separately by lowest cost.
# ----------------------------------------------------------------------------------
def _partial_key(v):
    """Sort key for NON-valid candidates: prefer more correct, then cleaner ancillas, then
    more of the stream parsing. All come straight from verify()."""
    return (
        float(v.get("frac_correct") or 0.0),
        float(v.get("frac_ancilla_clean") or 0.0),
        float(v.get("parse_progress") or 0.0),
    )


def _verify_safe(ops_text, spec):
    if not ops_text.strip():
        return {"valid": False, "reason": "empty (no op lines extracted)",
                "frac_correct": 0.0, "frac_ancilla_clean": 0.0, "parse_progress": 0.0,
                "cost": None, "witness": None, "peak_width": None, "toffoli": None,
                "vs_ref_pct": None}
    try:
        return proxy_env.verify(ops_text, spec)
    except Exception as e:  # never crash the loop
        return {"valid": False, "reason": f"verify-exc: {e}",
                "frac_correct": 0.0, "frac_ancilla_clean": 0.0, "parse_progress": 0.0,
                "cost": None, "witness": None, "peak_width": None, "toffoli": None,
                "vs_ref_pct": None}


# ----------------------------------------------------------------------------------
# Decode the declared output-register layout so feedback can speak in INPUT-register and
# OUTPUT-register values (what the model sees in the prompt), not raw qubit bitmasks.
# ----------------------------------------------------------------------------------
def _ancilla_qubits(spec):
    width = int(spec.get("width", 0))
    out_set = set(spec.get("out_qubits", []))
    return [q for q in range(width) if q not in out_set]


# ----------------------------------------------------------------------------------
# Build a SPECIFIC feedback user-turn from verify()'s reason + witness.
# This is the heart of the harness: turn the verifier's machine signal into a concrete,
# actionable instruction the 1.5B model can act on.
# ----------------------------------------------------------------------------------
def build_feedback(v, spec, last_opstream):
    reason = str(v.get("reason") or "")
    witness = v.get("witness")
    ref_cost = spec.get("reference_cost")
    fc = v.get("frac_correct")
    fac = v.get("frac_ancilla_clean")
    anc = _ancilla_qubits(spec)
    out_q = spec.get("out_qubits", [])
    in_q = spec.get("in_qubits", [])

    lines = []
    lines.append("Your previous op-stream was REJECTED by the verifier. Here it is:")
    lines.append("----- your attempt -----")
    lines.append(last_opstream.strip() if last_opstream.strip()
                 else "(no parseable op lines were found in your reply)")
    lines.append("------------------------")

    # --- diagnose the specific failure ---
    if reason.startswith("parse") or reason.startswith("validate") or reason.startswith("spec") \
            or reason.startswith("analyze") or reason.startswith("verify-exc") \
            or reason.startswith("empty"):
        lines.append(f"PARSE/FORMAT ERROR: {reason}.")
        lines.append("Emit ONLY op lines, one operation per line, no prose, no backticks, "
                     "no numbering. Each line is e.g. `CCX q0 q1 q2` or `CX q0 q1` or `X q0`. "
                     "The LAST qubit token on a line is the target.")
    elif reason.startswith("reset/HMR"):
        lines.append("BANNED OP: this task band forbids R and HMR (reset). Use only reversible "
                     "ops X / CX / CCX / SWAP / Z / CZ / CCZ. Replace any reset with an "
                     "explicit uncompute (apply the same gates in reverse).")
    elif reason.startswith("peak_width") or reason.startswith("enumeration"):
        cap = spec.get("max_width")
        pw = v.get("peak_width")
        lines.append(f"WIDTH CAP EXCEEDED: peak_width={pw} but the hard cap is {cap}. "
                     f"Do not reference qubit indices >= {cap}. Reuse ancilla qubits "
                     f"(uncompute and recompute) instead of allocating new high-index qubits.")
    elif reason == "not reversible (collision)":
        if isinstance(witness, dict):
            a = witness.get("input_a"); b = witness.get("input_b"); o = witness.get("output")
            lines.append(f"NOT REVERSIBLE: two different full-state inputs {a} and {b} both map "
                         f"to output state {o}. The circuit must be a bijection on ALL "
                         f"{1 << int(spec.get('width', 0))} states. You likely overwrote a qubit "
                         f"without preserving information (e.g. an X/CX that is not undone, or a "
                         f"target that aliases what you needed). Make every step invertible.")
        else:
            lines.append("NOT REVERSIBLE: two inputs collide to the same output. The circuit "
                         "must be a full-state bijection; make every step invertible.")
    elif reason == "classical_mismatch":
        if isinstance(witness, dict):
            x = witness.get("input"); got = witness.get("got"); want = witness.get("expected")
            nbits = len(out_q)
            gb = format(got & ((1 << nbits) - 1), f"0{nbits}b") if got is not None else "?"
            wb = format(want & ((1 << nbits) - 1), f"0{nbits}b") if want is not None else "?"
            diff = (got ^ want) if (got is not None and want is not None) else None
            lines.append(f"WRONG OUTPUT (correct on {fc:.0%} of inputs): on input x={x} the "
                         f"circuit produced {got} (bits {gb}) on the output register {out_q}, "
                         f"but the target function requires {want} (bits {wb}).")
            if diff:
                wrong_pos = [i for i in range(nbits) if (diff >> i) & 1]
                wrong_qubits = [out_q[i] for i in wrong_pos if i < len(out_q)]
                lines.append(f"The output bits that are WRONG are positions {wrong_pos} "
                             f"(qubits {wrong_qubits}). Fix only the logic feeding those qubits "
                             f"for input x={x}; keep the inputs you already get right.")
        else:
            lines.append(f"WRONG OUTPUT: correct on only {fc:.0%} of inputs. Re-derive the "
                         f"boolean function for the output qubits {out_q}.")
    elif reason == "ancilla_garbage":
        lines.append(f"DIRTY ANCILLA: ancilla qubits {anc} must return to |0> for EVERY input, "
                     f"but some input leaves one set to 1 (clean on only {fac:.0%} of inputs). "
                     f"You computed into an ancilla and never uncomputed it. After using an "
                     f"ancilla, apply the SAME gates that set it, in REVERSE order, to reset it "
                     f"to |0>. Do not leave scratch qubits dirty.")
    elif reason == "phase_garbage":
        lines.append("PHASE GARBAGE: global phase must be 0 on every input, but a Z/CZ/CCZ/NEG "
                     "left it nonzero on some input. Remove the phase-injecting op, or pair it "
                     "with its inverse so the net phase is 0.")
    elif reason == "forward_reverse_identity_failed":
        lines.append("NOT SELF-INVERSE: applying your ops then their reverse did not return the "
                     "identity. Check operand order on each line (target is the LAST qubit) and "
                     "that no op aliases its own control (target != control1 != control2).")
    else:
        lines.append(f"REJECTED: {reason}. frac_correct={fc}, frac_ancilla_clean={fac}.")

    # --- always restate the hard contract + the cost target ---
    lines.append("")
    lines.append("REQUIREMENTS (all hard): compute the target function on the input register "
                 f"{in_q} for every input; be reversible (full-state bijection); keep phase=0; "
                 f"return ancilla qubits {anc} to |0>. No operand aliasing.")
    if ref_cost:
        lines.append(f"Then minimize cost = avg_Toffoli x peak_width; beat the reference "
                     f"cost {ref_cost:.0f} (strictly cheaper wins).")
    lines.append("Emit the corrected op-stream now — ONLY op lines, nothing else.")
    return "\n".join(lines)


def build_cost_feedback(v, spec, last_opstream):
    """Feedback when an attempt is VALID but not cheaper than the reference — ask for a strictly
    cheaper revision while preserving correctness (used only when we already banked a valid)."""
    ref_cost = spec.get("reference_cost")
    cost = v.get("cost")
    tof = v.get("toffoli")
    pw = v.get("peak_width")
    lines = [
        "Your op-stream is VALID and correct — good. Now make it CHEAPER without breaking it.",
        "----- your valid attempt -----",
        last_opstream.strip(),
        "------------------------------",
        f"Current cost = {cost:.0f} (avg_Toffoli={tof} x peak_width={pw}). "
        f"Reference to beat = {ref_cost:.0f}." if cost is not None else
        f"Reference to beat = {ref_cost:.0f}.",
        "Only CCX/CCZ cost (X/CX/SWAP/Z are free). Remove redundant Toffolis, merge gates, or "
        "shrink peak_width. Keep correctness, reversibility, phase=0, clean ancillas. "
        "Emit ONLY op lines.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# The main entry point.
# ----------------------------------------------------------------------------------
def agentic_solve(model, task_spec, prompt, n_samples=8, max_turns=4,
                  temperature=0.7, base_temperature=0.4, seek_cheaper=False,
                  verbose=False):
    """Verifier-in-the-loop solve for one task.

    Args:
      model           Ollama model tag (e.g. 'ecdsa-coder-1.5b-sft').
      task_spec       the proxy_env task_spec dict (in_qubits/out_qubits/f/width/... ).
      prompt          the model-facing prompt text (tasks.render_prompt(inst)).
      n_samples       completions sampled at turn 0 (best-of-N seed).
      max_turns       max repair turns AFTER turn 0 (so up to max_turns+1 sampling rounds).
      temperature     sampling temperature for repair turns (turn>=1).
      base_temperature temperature for the turn-0 best-of-N spread (varied per sample).
      seek_cheaper    if True, keep iterating after finding a valid circuit to try to beat ref.

    Returns dict:
      {best_opstream, valid, cost, turns, transcript}
      where transcript is a list of per-round records (role/content/verify summary).
    """
    transcript = []
    best_valid = None          # (cost, opstream, verify_dict)
    best_partial = None        # (partial_key, opstream, verify_dict)

    def consider(ops, v):
        nonlocal best_valid, best_partial
        if v.get("valid"):
            c = v.get("cost")
            c = float(c) if c is not None else float("inf")
            if best_valid is None or c < best_valid[0]:
                best_valid = (c, ops, v)
        else:
            k = _partial_key(v)
            if best_partial is None or k > best_partial[0]:
                best_partial = (k, ops, v)

    # ---- Turn 0: best-of-N seed ----
    turn0_samples = []
    for i in range(n_samples):
        # Spread temperature across samples for diversity; distinct seed per sample.
        t = base_temperature + (i / max(1, n_samples - 1)) * 0.6  # base..base+0.6
        seed = 1000 + i
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
        try:
            comp = ollama_chat_messages(model, msgs, temperature=t, seed=seed)
        except Exception as e:
            comp = f"__error__ {e}"
        ops = extract_opstream(comp)
        v = _verify_safe(ops, task_spec)
        consider(ops, v)
        turn0_samples.append({
            "sample": i, "temperature": round(t, 3), "seed": seed,
            "raw": comp, "opstream": ops,
            "valid": bool(v.get("valid")), "reason": str(v.get("reason"))[:80],
            "cost": v.get("cost"), "frac_correct": v.get("frac_correct"),
            "frac_ancilla_clean": v.get("frac_ancilla_clean"),
        })
        if verbose:
            print(f"  [t0 s{i} T={t:.2f}] valid={bool(v.get('valid'))} "
                  f"reason={str(v.get('reason'))[:40]} fc={v.get('frac_correct')}")
    transcript.append({"turn": 0, "kind": "best_of_n", "n_samples": n_samples,
                       "samples": turn0_samples})

    turns_used = 0
    # If turn-0 already produced a valid circuit and we don't want to chase a cheaper one, done.
    if best_valid is not None and not seek_cheaper:
        return _result(best_valid, best_partial, turns_used, transcript)

    # ---- Repair turns ----
    # Carry a 2-turn dialog: the original prompt, the model's last attempt, the feedback.
    for turn in range(1, max_turns + 1):
        # Choose what to give feedback on: the best valid (cost-improve) or best partial (repair).
        if best_valid is not None and seek_cheaper:
            base_ops = best_valid[1]
            base_v = best_valid[2]
            feedback = build_cost_feedback(base_v, task_spec, base_ops)
        elif best_partial is not None:
            base_ops = best_partial[1]
            base_v = best_partial[2]
            feedback = build_feedback(base_v, task_spec, base_ops)
        else:
            # nothing parseable at all yet — re-ask with a format nudge
            base_ops = ""
            base_v = {"reason": "empty"}
            feedback = build_feedback(base_v, task_spec, "")

        # The model's prior attempt is replayed as an assistant turn so the dialog is coherent.
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": base_ops if base_ops else "(no op lines)"},
            {"role": "user", "content": feedback},
        ]
        seed = 2000 + turn
        try:
            comp = ollama_chat_messages(model, msgs, temperature=temperature, seed=seed)
        except Exception as e:
            comp = f"__error__ {e}"
        ops = extract_opstream(comp)
        v = _verify_safe(ops, task_spec)
        consider(ops, v)
        turns_used = turn
        transcript.append({
            "turn": turn, "kind": "repair",
            "feedback": feedback, "raw": comp, "opstream": ops,
            "valid": bool(v.get("valid")), "reason": str(v.get("reason"))[:80],
            "cost": v.get("cost"), "frac_correct": v.get("frac_correct"),
            "frac_ancilla_clean": v.get("frac_ancilla_clean"),
        })
        if verbose:
            print(f"  [turn {turn}] valid={bool(v.get('valid'))} "
                  f"reason={str(v.get('reason'))[:40]} fc={v.get('frac_correct')} "
                  f"cost={v.get('cost')}")

        # Stop as soon as we have a valid circuit, unless we are explicitly chasing cheaper.
        if best_valid is not None and not seek_cheaper:
            break

    return _result(best_valid, best_partial, turns_used, transcript)


def _result(best_valid, best_partial, turns_used, transcript):
    if best_valid is not None:
        cost, ops, v = best_valid
        return {
            "best_opstream": ops,
            "valid": True,
            "cost": (None if cost == float("inf") else cost),
            "toffoli": v.get("toffoli"),
            "peak_width": v.get("peak_width"),
            "vs_ref_pct": v.get("vs_ref_pct"),
            "reason": v.get("reason"),
            "turns": turns_used,
            "transcript": transcript,
        }
    if best_partial is not None:
        _k, ops, v = best_partial
        return {
            "best_opstream": ops,
            "valid": False,
            "cost": v.get("cost"),
            "toffoli": v.get("toffoli"),
            "peak_width": v.get("peak_width"),
            "vs_ref_pct": v.get("vs_ref_pct"),
            "reason": v.get("reason"),
            "frac_correct": v.get("frac_correct"),
            "frac_ancilla_clean": v.get("frac_ancilla_clean"),
            "turns": turns_used,
            "transcript": transcript,
        }
    return {
        "best_opstream": "", "valid": False, "cost": None, "reason": "no_samples",
        "turns": turns_used, "transcript": transcript,
    }


if __name__ == "__main__":
    # Smoke test on one held-out task.
    import argparse
    import tasks  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ecdsa-coder-1.5b-sft")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=4)
    a = ap.parse_args()

    inst = tasks.build_heldout_fma(n=2, m=3)
    print(f"[smoke] task={inst.family} ref_cost={inst.task_spec['reference_cost']:.0f}")
    res = agentic_solve(a.model, inst.task_spec, inst.prompt,
                        n_samples=a.n_samples, max_turns=a.max_turns, verbose=True)
    print(f"[smoke] valid={res['valid']} cost={res['cost']} turns={res['turns']}")
    print(res["best_opstream"][:400])
