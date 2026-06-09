# I tried to build a self-improving circuit-synthesis model. It plateaued. Here's the honest story.

*A short version. The full lab notebook, with every number and dead-end, is in
[`PROCESS_LOG.md`](./PROCESS_LOG.md).*

## What I was trying to do

The ECDSA.fail challenge wants a cheap *reversible circuit* for a hard function (secp256k1 point
addition). I wanted to know if a small, open model could learn to do that kind of synthesis — and,
more ambitiously, *improve itself* at it. I couldn't train on the real 256-bit problem directly, so
I built a faithful proxy: synthesize a reversible circuit for a random GF(2) linear map on *n* bits,
graded by width (n=3…6), checked by a simulator that's bit-for-bit identical to the real reference.

## The one thing that clearly worked

Asked to write the whole circuit in one shot, a 1.5B model and a 7B model both succeed ~4.8% of the
time — *identical*. That's the fingerprint of a bottleneck that isn't model size: the model can't
simulate the circuit's running state in its head. So I gave it a **tool** that tracks the state and
shows it what's still wrong after each gate. With the tool, trained models start solving held-out
tasks — and *now* scale matters: the 1.5B caps out at n=4, the 8B reaches n=5. A tool, not a bigger
model, was the fix; and the tool is what made scale matter at all. That result is solid.

## The thing I hoped would work, and didn't

Then the ambitious part: a **flywheel**. Let the model solve fresh tasks, keep the solutions the
verifier confirms, retrain on them, repeat — the model's own successes as its next curriculum. For a
while it looked like it was working: a metric jumped from 0% to 7.5% on the hardest band; the harvest
yield on n=6 doubled across rounds. I wrote that up as a breakthrough.

It wasn't. When I re-ran the evaluation with enough samples to be trustworthy, the gain evaporated.
The base model, the first iteration, and the second were **statistically identical** (~58%). The
"0% → 7.5%" was a sampling artifact — with only two attempts per task the base looked like it
*never* solved n=6, but with five attempts it solves it about as often as the "improved" models do.
The flywheel re-taught the model what it already knew. It never pushed the frontier, because
training a model on its own correct answers can't teach it to solve the problems it currently fails.

## What I actually learned

The most useful output of this project is a discipline, not a model: **most of my apparent progress
was measurement noise.** Two separate "wins" died the moment I measured them properly — an
8-sample eval that inflated the baseline by 14 points, and a 2-attempt eval that invented a
breakthrough. The cheapest, highest-leverage thing I built was a fixed, adequately-sampled held-out
test, run identically against every checkpoint.

So I'm stopping the self-improvement loop and calling it: a working base model (reliable
reversible-circuit synthesis through n=5) and a clean, mechanistically-explained negative result
(self-harvest plateaus). The honest next levers — reinforcement learning against the verifier
reward, a difficulty curriculum, or simply a bigger base — are real, but they're the *next*
experiment, not this one.

Negative results that you trust are worth more than positive results that you don't.
