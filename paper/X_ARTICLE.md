# Can a Small AI Learn to Break ECDSA? Notes from Eigen Labs' ECDSA.fail Challenge — and an Honest Negative Result

*A small open model, a verifier-backed tool, and a self-improvement loop that didn't pan out. Published as-is, including what failed.*

[ILLUSTRATION 1 — hero chart: held-out solve rate flat across two self-improvement rounds, below the target. (fig0_hero.png)]

## The backdrop: ECDSA.fail, Eigen Labs, and "Will you beat Google?"

[Eigen Labs](https://www.eigenlabs.org) runs a challenge called **ECDSA.fail** with a blunt question: *can you break ECDSA — and will you beat Google?* Concretely, it's a competition to construct a cheaper **quantum circuit** for attacking secp256k1 (the curve behind Bitcoin and Ethereum keys), scored on the **qubit × Toffoli product** — the rough cost of running the attack on a fault-tolerant quantum computer. The bar to beat is the best published circuit construction (Litinski) and, looming over all of it, Google Quantum AI's resource estimates for breaking elliptic-curve crypto.

It's a real, open optimization problem: there's no efficient classical recipe for the optimum, and every candidate circuit is expensive to check. That makes it a perfect place to ask a different question.

## The question we actually asked

Not "can *we* beat Google" — but: **can a small, open language model *learn* to do this kind of reversible-circuit synthesis at all, and can it teach *itself* to get better?** If yes, you'd have a reusable optimizer you could eventually point at the real ECDSA.fail circuit.

You can't train directly on the 256-bit circuit (too expensive to evaluate, search space astronomical), so we built a faithful, cheap **proxy**: synthesize a reversible circuit for a random linear map over GF(2), graded by width, scored by a simulator that matched a Rust reference on every case of an 800-case battery. Same shape as the real problem (reversible gates against a hard target, exact verifier), small enough to iterate on and — crucially — to *measure properly*.

[ILLUSTRATION 2 — the proxy task / tool: the model drives a tool one gate at a time, seeing what's still wrong. (fig2_base_by_band.png)]

## Finding 1: a verifier-backed tool beats raw scale

Ask a model to write the whole circuit in one shot and it fails almost always — and here's the tell: a **1.5B and a 7B model of the same family score *identically* (4.8%)**. Five times the parameters changed nothing. The bottleneck isn't knowledge; it's *symbolic execution* — keeping the running circuit state straight in your head.

Give the model a **tool** that tracks the state and shows what's still wrong after each gate, and trained models start solving real instances. Only *then* does scale matter (the 8B reaches widths the 1.5B can't). The lever was a tool, not a bigger model.

[ILLUSTRATION 3 — "the wall": 1.5B vs 7B both at 4.8% one-shot. (fig1_the_wall.png)]

## Finding 2: the self-improvement flywheel plateaued

The ambitious part was a flywheel: let the model solve fresh problems, keep the solutions a verifier confirms, retrain on them, repeat — its own successes as the next curriculum.

Measured honestly on a fixed held-out set: **no improvement.** Base, iteration 1, and iteration 2 land within a point of each other (~58%). Training a model on answers it can already produce mostly re-teaches what it already knows; the rare hard-case solutions are too few to move the frontier.

[ILLUSTRATION 4 — the flat flywheel curve: base ≈ iter-1 ≈ iter-2. (fig3_flywheel_flat.png)]

## What's honest to say

- This is a **negative result on a proxy**, reported as one — we did *not* beat Google, Litinski, or touch the real 256-bit ECDSA.fail circuit. It's a stepping-stone study.
- The durable positive: **tool use > scale** on this task, and scale only matters once the tool is in place.
- The path forward isn't "more self-harvest" — it's reinforcement learning against the verifier reward, a difficulty curriculum, or a bigger base.

Everything is open — the model, the code, the data, and the full write-up including the parts that failed:

🤗 Model: https://huggingface.co/dennisonb/reversible-circuit-8b-tool
💻 Code + paper: https://github.com/dennisonbertram/reversible-circuit-llm
🔗 The challenge: https://ecdsa.fail (an Eigen Labs project)

*Negative results you can trust are worth more than positive results you can't.*
