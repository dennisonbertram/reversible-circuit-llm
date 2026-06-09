# Teaching a small model to drive a verifier-backed tool — a process log

*A living lab notebook. Honest about the dead-ends, because those are the load-bearing
lessons. Source material for a later blog post / paper. Last updated: 2026-06-08.*

---

## TL;DR (so far)

We set out to train a small, open-source model that can do **reversible-circuit synthesis** —
the kind of work the [ECDSA.fail](https://ecdsa.fail) challenge requires (find a cheap reversible
quantum circuit for a hard function). We can't train directly on the 256-bit secp256k1 point-add
frontier, so we built a faithful **proxy**: synthesize a reversible circuit for a random GF(2)
linear map on *n* bits, graded by width (B1=n3 … B4=n6), and scored by a simulator that is
800/800 bit-identical to the real Rust reference.

The arc of findings:

1. **Without a tool, the task is a wall.** A 1.5B and a 7B model both one-shot-synthesize correct
   circuits ~4.8% of the time. *Identical.* The bottleneck is not model capacity — it's
   **symbolic execution**: the model can't track the circuit state in its head.
2. **A state-externalizing tool removes that wall — but exposes a second one.** Give the model a
   tool that tracks circuit state and shows the residual each turn, and trained models start
   solving held-out tasks. But zero-shot tool use thrashes: the new bottleneck is **sequential
   planning**, not state-tracking.
3. **With the tool, scale matters** (it didn't without). A trained 1.5B caps at n=4; a trained 8B
   reaches n=5. Each ~5× of parameters buys roughly one more bit of width.
4. **Self-harvest expert iteration did NOT improve the model — a clean negative result.** Under a
   fair best-of-5 eval, base ≈ iter-1 ≈ iter-2 (all ~58% overall). An earlier "B4 cracked 0→7.5%"
   excitement turned out to be a *best-of-2 sampling artifact*: the base already solves n=6 at 5%
   with enough attempts. The flywheel re-learns the model's own distribution; it can't push past
   the ceiling that 1200 optimal expert demos already reached.
5. **Measurement discipline is THE story.** Two phantom results died to better eval: an 8-task eval
   inflated the base to 62.5% (real: 51%), and a best-of-2 eval manufactured a B4 "breakthrough"
   that best-of-5 erased. Under-sampling at low solve rates invents progress that isn't there.

Net: we built a real base model (~58% best-of-5 on held-out reversible-circuit synthesis up to
n=5) and rigorously showed that **self-harvest iteration plateaus immediately** on this task. The
honest next lever is not "more iterations" — see §10.

---

## 1. The problem and why it's hard

**ECDSA.fail** asks for a reversible circuit computing secp256k1 point addition with minimal
cost (avg Toffoli count × peak qubits). It's a real, open optimization problem — no efficient
classical method gives the optimum. We wanted a *small open model* that could learn to do this
kind of synthesis, as a stepping stone, and ideally generalize the skill.

Training directly on the 256-bit frontier is infeasible (each evaluation is seconds of Rust sim;
the search space is astronomical). So we built a **proxy task** that is:

- **Real** — verified by a simulator bit-identical to the reference (`proxy/proxy_env.py`, 800/800).
- **Gradeable** — difficulty = register width *n* (B1=3, B2=4, B3=5, B4=6 bits).
- **Cheap** — a bit-packed classical-reversible simulator, microseconds per check.
- **Honestly hard** — the model must emit a gate sequence (`CX`, `CCX`/Toffoli, `SWAP`) that
  transforms identity into a target GF(2) linear map, in place.

The family we focus on is `gf2_linear`: given an invertible *n×n* GF(2) matrix, synthesize a
reversible circuit that applies it. (Classically solvable by Gaussian elimination — which is
exactly how we generate optimal *expert* demonstrations via `proxy/synth.py`. The point is not to
beat Gaussian elimination; it's to see whether a small model can *learn tool-driven synthesis* and
*self-improve*, as a transferable method.)

---

## 2. Finding 1 — without a tool, it's a symbolic-execution wall (capacity is not the bottleneck)

First experiment: ask the model to one-shot emit the whole circuit (`OP-STREAM` of gates), then
verify.

| Model | One-shot synthesis (held-out) |
|------|------|
| Qwen2.5-Coder-1.5B | 4.8% |
| Qwen2.5-Coder-7B | 4.8% |

**Identical.** Scaling the model 5× did *nothing*. That's the signature of a bottleneck that isn't
capacity: the model cannot mentally simulate the running circuit state across many gates, so it
can't tell whether its partial sequence is on track. It's flying blind.

> **Lesson:** before reaching for a bigger model, find out *which* wall you're hitting. Here, more
> parameters were worthless because the limiting skill (symbolic execution of a growing circuit)
> isn't what scale improves.

---

## 3. The intervention — externalize the state into a tool

We built **`ToolEnv`** (`proxy/tooluse.py`): a stateful environment the model drives one gate per
turn. After each gate it re-renders the **current state vs. the target** — for `gf2_linear`, the
*residual rows* (`current y0 = x0 | target y0 = x0^x2^x4  <-- WRONG`). The model no longer has to
simulate in its head; it reacts to an externalized, always-correct view. This mirrors how a human
would actually do it: row-reduce, looking at the board.

Two immediate sub-findings:

- **Zero-shot tool use thrashes.** Untrained models emit ill-formed ops, undo their own progress,
  loop. Having the state isn't enough — you have to know what to *do* with it.
- **So the bottleneck moved: from state-tracking to sequential planning.** The tool is necessary
  but not sufficient. (This is itself a result — see our earlier note
  [[tooluse-state-externalization]]: state-externalization alone does *not* beat the reasoning
  ceiling; the model must be *trained* to plan in the tool.)

We then built data factories for tool-driven traces (`proxy/tooltrace_gen.py`): expert
demonstrations of the *turn-by-turn* play (one canonical op per turn), framed identically to how
the model will be evaluated (train == eval). A subtlety that cost us real diversity early on:
**omitting the truth table from sbox/gf2 prompts made targets underspecified/contradictory** and
capped distinct tasks at ~2,300; including it unlocked 31,718 distinct tasks (14×).

---

## 4. Finding 2 — with the tool, scale matters (and buys ~1 bit per 5×)

Training models on the tool-driven traces and evaluating them *driving the tool* on held-out tasks:

| Model (trained, tool-driven) | Top solvable band |
|------|------|
| 1.5B | B2 (n=4) — caps here; **0% at B3 even when trained on B3 data** |
| 8B | B3 (n=5) — ~25-36% |

So *with* the tool, scale does what it refused to do without it: the 8B reaches a width the 1.5B
provably cannot, even when the 1.5B is trained directly on that width. The 1.5B's failure at n=5 is
a genuine **capacity ceiling**, not a data problem.

> **Lesson:** "does scale help?" has no context-free answer. The *same* two models were identical
> without the tool and a full band apart with it. Tooling can convert a capacity-insensitive task
> into a capacity-sensitive one.

We also learned the small-*n* task space **saturates**: there are only ~168 invertible 3×3 GF(2)
matrices total, so the expert set already nearly exhausts B1/B2. The flywheel can only add *new*
coverage at n≥5 (n=5 ≈ 10M matrices) — which is exactly where only the 8B can play. This is why the
1.5B is the wrong base for self-improvement, and the 8B is the real attempt.

---

## 5. The flywheel — expert iteration in the tool

The core idea (STaR / expert-iteration flavored): the model drives the tool over fresh training
tasks; we **keep every verifier-confirmed solution** as new training data; retrain; repeat. The
model's own successes become the next round's curriculum.

Components (`flywheel/`, `train/flywheel_harvest.py`):
- **Harvest** — drive the latest model over fresh tasks (best-of-N sampling), record solved
  trajectories in the exact eval framing, keep the cheapest play per task.
- **Combine** — dedup expert + harvested solutions per task; this is the replay buffer.
- **Retrain → merge → eval** — fresh SFT on the cumulative set; score held-out.

### 5a. Dead-end → fix: the harvester didn't scale
First real harvests took **~3.3 hours** because *stuck rollouts ground out the full turn budget* —
a task the model can't solve still ran all 84 turns × 5 restarts. Fix: **drop-on-no-progress**
(track the residual `n_mismatch`; if it doesn't improve for ~10 turns, abandon the rollout). Plus
the render is **Markov** (the current residual fully specifies the remaining problem), so we cut
the live context window from 22 turns to 8 — ~2.5× less prefill at no loss. Harvest dropped from
~3.3h toward minutes-to-~1h depending on width.

> **Lesson:** an unsolvable rollout is pure waste; give it an exit. And don't pay to re-feed history
> the task representation already encodes.

### 5b. The 8B base, from expert demos alone
Trained Qwen3-8B on a B4-rich compact trace set (1200 demos/band for the hard bands). Held-out:

| Eval | B1 | B2 | B3 | B4 | Overall |
|------|----|----|----|----|---------|
| 8-task/band (noisy) | 100% | 87.5% | 62.5% | 0% | 62.5% |
| **40-task/band (clean)** | **95%** | **85%** | **25%** | **0%** | **51.2%** |

Already strong on n≤4 straight from imitation. **n=6 = 0%** despite 1200 optimal demos — the hole.

### 5c. Dead-end → lesson: the 8-task eval lied to us
We ran one flywheel iteration. Its **8-task** eval read B3 25% (down from the base's 62.5%) and
overall 56.2% (down from 62.5%) — looked like a **regression**, so we killed the run to avoid
compounding it.

Then the **40-task clean eval** told the truth: the base's "62.5%" was a **lucky 5/8 draw on B3**;
the real base B3 is ~25%. Iter-1's B3 (20-25%) **matched** it. There was **no regression** — we'd
compared against an over-measured baseline. We killed the run a bit early.

> **Lesson:** at 8 samples/band a single band swings ±25 points on luck. We built a **fixed
> 40-task/band held-out set, same seeds, identical protocol, scored on the base and every
> iteration**, so the curve is signal, not noise. Cheap instrumentation discipline would have saved
> a wrong decision.

### 5d. Finding 3 — "5 self-solutions beat 1200 expert demos on n=6"  ⚠️ LATER OVERTURNED (see §5h)
> **Read this section as a snapshot of what we believed at this point.** The best-of-2 eval below
> made it look like the flywheel cracked n=6. The fair best-of-5 eval in §5h dissolves it — the base
> already solves n=6 at 5%; best-of-2 under-sampled it to 0. Kept here intact because the *mistake*
> is the lesson.

The clean iter-1 eval, as measured at best-of-2:

| Band | Clean base | Clean iter-1 | Δ |
|------|-----------|-------------|---|
| B1 | 95% | 97.5% | +2.5 |
| B2 | 85% | 82.5% | −2.5 |
| B3 | 25% | 20% | −5 (noise) |
| **B4 (n=6)** | **0%** | **7.5%** | **+7.5** |
| Overall | 51.2% | 51.9% | +0.7 |

Overall is flat (B1/B2 saturated, B3 within noise) — but **B4 went 0 → 7.5%**: iter-1 solves 3/40
n=6 tasks the base solved *none* of.

The crucial detail: due to a cap bug (see 5e), iter-1 trained on **fewer** expert B4 demos than the
base (995 vs 1200) but **+5 harvested (model-found) B4 solutions** — and *that* cracked n=6. So **~5
of the model's own solutions taught it more about n=6 than 1200 synth-optimal expert
demonstrations did.** Expert iteration ≫ imitation on the hard band. *(This was the exciting read at
the time. It did not survive the fair eval — §5h. The base was already at 5% n=6; we were looking at
best-of-2 noise.)*

### 5e. Dead-end → fix: the cap was subtracting expert data
We capped the replay buffer at 1000 traces/band to bound SFT time — but the base trained on 1200
demos/band. So the "cumulative" set had *fewer* hard-band demos than the base: the flywheel was
net-*subtracting* data on exactly the bands that matter. Fix: cap raised to 1500 (nothing expert
dropped) **and** `combine.py` now *always* preserves harvested traces in the cap (they're pricier
than synth-optimal expert, so a naive cheapest-N cap silently discards the very signal we want).

### 5g. Finding 4 — "the harvest compounds"  ⚠️ LATER OVERTURNED (see §5h)
> **Another believed-then-refuted snapshot.** The harvest yield *did* rise round-over-round — but
> that's the model's capability on the *training* distribution. §5h shows it did **not** transfer to
> held-out tasks. Compounding in the harvest ≠ compounding in generalization.

When we resumed the flywheel from the iter-1 model, the harvest yield climbed — a better model
harvests more of its own hard-band solutions *on training tasks*:

| Band | harvested by **base** | harvested by **iter-1 model** |
|------|------|------|
| B2 | 84.2% | 84.2% (saturated) |
| B3 | 35.8% | **44.2%** |
| **B4 (n=6)** | **4.2% (5/120)** | **8.3% (10/120)** |

**B4 harvest doubled.** This is the self-reinforcing core of expert iteration: model improves on the
hard band → harvests more of its own solutions there → richer training data → (expected) further
improvement. The next SFT trains on ~28 model-found B4 solutions (18 seeded + 10 fresh) vs the 5
that first cracked it from 0. Whether this converts to a *higher eval* B4 is the open test.

### 5h. Finding 5 (THE NEGATIVE RESULT) — self-harvest expert iteration does NOT improve the model here
We suspected the flat best-of-2 curve might be the eval *underselling* real gains (the harvest B4
rate rose 4.2→8.3%). So we ran the fair test: **re-eval base / iter-1 / fly8b-iter-1 at best-of-5**
(matching how the model is actually used), 40 held-out tasks/band. The result is decisive and
overturns the earlier optimistic read:

| Model | B1 | B2 | B3 | B4 | Overall (best-of-5) |
|------|----|----|----|----|------|
| base | 95% | 92.5% | 40% | 5.0% | **58.1%** |
| iter-1 | 100% | 82.5% | 37.5% | 7.5% | 56.9% |
| fly8b iter-1 | 100% | 92.5% | 35% | 5.0% | **58.1%** |

**The three models are statistically identical. The flywheel produced zero held-out improvement.**

Two corrections fall out of this, and they matter:

1. **"B4 cracked 0 → 7.5%" was a measurement artifact.** At best-of-5 the *base already solves B4 at
   5%* (2/40). It was never truly 0 — with only 2 attempts (best-of-2) it got unlucky (0/40). The
   flywheel didn't crack n=6; the base was already there. The whole "5 self-solves beat 1200 expert
   demos" story dissolves under a fair eval. **Lesson, paid for in compute: at low solve rates,
   best-of-2 manufactures phantom gains; always anchor with enough attempts.**

2. **The harvest gains didn't generalize.** Harvest B3 rose 36→44→49% (training tasks, best-of-5),
   but held-out best-of-5 B3 is flat (40 → 37.5 → 35, if anything down). The model got better at
   the *training distribution*, not the task.

**Why self-harvest plateaus (the mechanism):** STaR-style self-training only adds signal when the
harvested solutions teach the model to solve things it *couldn't*. Here the harvest is dominated by
B2/B3 tasks the model *already* solves → redundant data, no new information. The rare frontier (B4)
solves are too few (~8%) and too noisy to shift the boundary. The base — trained on 1200 *optimal*
expert demos/band — has already extracted what imitation can give and sits at the 8B's capability
ceiling (~58% best-of-5) for this task. Iterating on its own outputs can't exceed that; it can only
re-learn the same distribution (and the slight B3 dip hints at mild over-fitting to self-solves).

**Decision:** plateau confirmed at the fair eval → run **killed** (no iter-3). The lever is no
longer "more iterations." See §10 for the pivot.

### 5f. The B4 ceiling probe
How many n=6 solutions can the 8B produce with a *wide* search? A B4-only harvest from the base
(400 fresh tasks × best-of-6) yields ~3.5% (14 solves). So n=6 is genuinely near the 8B's ceiling —
self-harvest is a *slow* lever there, and (per §5h) too slow to shift held-out capability at all.

---

## 6. The 1.5B validation run (does the loop even run end-to-end?)
A 3-iteration flywheel on the saturated 1.5B, purely to prove the plumbing: harvest → combine →
SFT → merge → eval, repeated. Result: noisy (overall 0.25 → 0.375 → 0.292 → 0.417 at **only 6
tasks/band**). At the time the apparent B3 0%→17% looked like a new capability; in hindsight, at 6
tasks/band that's 0/6→1/6 — exactly the kind of small-sample wobble §5h taught us not to trust. The
durable takeaway from this run is narrow: **the loop runs end-to-end across iterations without
breaking.** Nothing about its *effectiveness* should be read from 6-task numbers.

---

## 7. Running results table (clean 40-task/band held-out)

**best-of-2** (temp 0.4) — the protocol that misled us:
| Stage | B1 | B2 | B3 | B4 | Overall | Note |
|------|----|----|----|----|---------|------|
| 8B base | 95% | 85% | 25% | 0% | 51.2% | B4 0/40 — but this is best-of-2 *under-sampling* |
| 8B iter-1 | 97.5% | 82.5% | 20% | 7.5% | 51.9% | "B4 cracked" — artifact |
| 8B fly8b iter-1 | 97.5% | 85% | 22.5% | 2.5% | 51.9% | flat |

**best-of-5** (temp 0.7) — the fair eval, and the verdict:
| Stage | B1 | B2 | B3 | B4 | Overall |
|------|----|----|----|----|---------|
| 8B base | 95% | 92.5% | 40% | 5.0% | **58.1%** |
| 8B iter-1 | 100% | 82.5% | 37.5% | 7.5% | 56.9% |
| 8B fly8b iter-1 | 100% | 92.5% | 35% | 5.0% | **58.1%** |

**Verdict: flat. The flywheel did not improve held-out capability.** Target was ≥65% overall +
B4>0; the model sits at ~58% best-of-5 and does not move with iteration.

---

## 8. Lessons distilled (the parts worth a paper)

1. **Diagnose the wall before scaling.** 1.5B == 7B at 4.8% told us capacity wasn't the problem;
   it was symbolic execution. A tool, not a bigger model, was the fix.
2. **A tool can change what scale means.** Same two models: identical without the tool, a full band
   apart with it. Externalizing state converted a capacity-insensitive task into a sensitive one.
3. **Self-harvest expert iteration plateaus — it can't exceed the model's own distribution.**
   SFT on the model's own verified solutions re-teaches what it already does; the harvest is
   dominated by tasks it already solves, and the rare frontier solves are too few to move the
   boundary. base ≈ iter-1 ≈ iter-2 (~58% best-of-5). To push the frontier you need a method that
   rewards solving *currently-failed* tasks (RL) or makes the frontier learnable (curriculum) — not
   imitation of current successes. *(We initially believed the opposite — see lesson 4 for why.)*
4. **Measure with enough samples to survive luck — this was the most important discipline in the
   project.** Two separate phantom results died to better eval: an 8-task eval inflated the base to
   62.5% (real best-of-2: 51%), and a best-of-2 eval manufactured a "B4 breakthrough" that best-of-5
   erased (the base already solved n=6 at 5%). At low solve rates, under-sampling *invents*
   progress. A fixed, adequately-sampled, identical-protocol held-out set is the cheapest and
   highest-leverage instrument here. Report best-of-k, and anchor every comparison against the base
   at the *same* k.
5. **In a self-improvement loop, never let bookkeeping subtract signal.** A cap meant to bound SFT
   time silently dropped expert demos (and risked dropping harvested solves) on exactly the bands
   that matter. (This bug existed but, per finding 5, was not what caused the plateau — the plateau
   is intrinsic to self-harvest.)
6. **Give unsolvable work an exit.** Drop-on-no-progress turned a 3.3h harvest into minutes and is
   what made iterating affordable at all.

---

## 9. Questions — answered, and still open

**Answered by this work:**
- *Does self-harvest expert iteration improve the model?* **No** (on this task, this base). Flat
  across two iterations at the fair eval.
- *Did the flywheel reach ≥65% / crack B4?* **No.** ~58% best-of-5, B4 ~5% — and the base already
  sits there. The "crack" was a best-of-2 artifact.
- *Does scale matter with the tool?* **Yes** (1.5B caps n=4, 8B reaches n=5) — the one robust
  positive.

**Still genuinely open (the pivot, §10):**
- Would **RL (GRPO) with a verifier reward** push the frontier where self-harvest couldn't? (Untried
  in the right configuration.)
- Would a **frontier curriculum** (graded n=5/n=6) make the hard band learnable?
- Where does the band ceiling move with a **larger base** (14B/30B)? (Scale is implicated as the
  real lever.)
- Does any of this **transfer** to the real 256-bit ECDSA.fail frontier? (Never attempted — the
  large gap between the n≤6 proxy and the real challenge remains entirely unexplored.)

## 10. The pivot — what to try now that self-harvest plateaued

The negative result is specific: **SFT on the model's own verified solutions cannot exceed the
model's own solution distribution.** It re-teaches what the model already does. To push the
*frontier* (the tasks it currently fails) we need a lever that optimizes for currently-failed tasks,
not one that imitates current successes. Candidates, roughly in order of principle:

1. **RL with the verifier reward (GRPO).** This is the natural fix: a policy-gradient method
   rewards the model for *solving tasks it currently fails*, which is exactly what SFT-on-self-solves
   cannot do. Our earlier GRPO attempts collapsed via format-reward hacking; the fix is a
   strictly validity-gated, verifier-grounded reward (reward only on a *verified-correct* circuit,
   shaped by cost). Driving the **tool** under GRPO (so the credit assignment is per-gate against
   the residual) is the most promising untried configuration.
2. **A frontier curriculum.** Grade n=5/n=6 tasks by intrinsic difficulty (number of off-diagonal
   pivots / minimal solution length), and train easy→hard so the model climbs the frontier instead
   of being handed only the hardest instances. Pairs naturally with either SFT or RL.
3. **Accept the 8B ceiling and characterize it.** ~58% best-of-5 (reliable to n=5, n=6 at ~5-8%)
   may simply be this model's limit for tool-driven synthesis. If so, the contribution is the
   *method + the rigorously-bounded negative result*, and the scale lever (a larger base) is the
   only way up — consistent with "scale matters with the tool" (§4).

The honest meta-lesson for the writeup: **most of the apparent progress in this project was
measurement noise, and the discipline of a fixed, adequately-sampled held-out eval is what
separated the one real finding (tool removes the symbolic-execution wall; scale then matters) from
two phantom ones (the 62.5% base, the B4 "breakthrough").**

---

*This document is updated as each iteration's clean eval lands. Latest: best-of-5 verdict — flywheel
plateau confirmed (2026-06-08 ~00:40); self-harvest run killed; pivot under consideration.*
