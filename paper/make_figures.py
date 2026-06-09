"""make_figures.py — honest charts for the writeup. Every number traces to the repo data
(flywheel/clean_eval_results.jsonl, flywheel/bo5_results.jsonl, docs/PROCESS_LOG.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#15202b"; MUTE = "#5b6b7b"; GRID = "#dfe6ee"
BLUE = "#2f6df0"; GREY = "#9aa7b4"; RED = "#e0563f"; GREEN = "#2ca06b"; AMBER = "#e8a33d"
plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlecolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
    "font.family": "sans-serif",
})

def _clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)

# ── Fig 0: HERO — the central result (flat best-of-5 curve vs target) ────────
fig, ax = plt.subplots(figsize=(9.2, 3.9))
stages = ["base", "iter-1", "iter-2"]
overall = [58.1, 56.9, 58.1]
ax.plot(stages, overall, "-o", color=BLUE, lw=3, ms=12, zorder=3)
for x, v in zip(stages, overall):
    ax.text(x, v+1.7, f"{v}%", ha="center", fontweight="bold", color=BLUE, fontsize=13)
ax.axhline(65, color=RED, lw=1.8, ls="--", zorder=2)
ax.text(2.04, 65, " target 65%", va="center", color=RED, fontsize=11, fontweight="bold")
ax.set_ylim(45, 70); ax.set_xlim(-0.3, 2.45)
ax.set_ylabel("held-out solve rate\n(best-of-5, n=40/band)")
ax.set_title("Two rounds of self-harvest expert iteration leave held-out capability unchanged",
             fontsize=13.5, fontweight="bold", pad=12)
_clean(ax); fig.tight_layout(); fig.savefig(f"{OUT}/fig0_hero.png"); plt.close(fig)

# ── Fig 1: THE WALL — capacity is not the bottleneck ─────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 4.2))
bars = ax.bar(["Qwen2.5-Coder\n1.5B", "Qwen2.5-Coder\n7B"], [4.8, 4.8],
              color=[GREY, BLUE], width=0.55)
for b, v in zip(bars, [4.8, 4.8]):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v}%", ha="center", fontweight="bold")
ax.set_ylim(0, 12); ax.set_ylabel("one-shot synthesis solve rate")
ax.set_title("Without a tool, 5× more parameters changes nothing\n(the wall is symbolic execution, not capacity)", fontsize=12.5)
_clean(ax); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_the_wall.png"); plt.close(fig)

# ── Fig 2: TOOL + SCALE — base model capability by band (best-of-5) ───────────
fig, ax = plt.subplots(figsize=(7.2, 4.3))
bands = ["B1\n(n=3)", "B2\n(n=4)", "B3\n(n=5)", "B4\n(n=6)"]
vals = [95, 92.5, 40, 5]
cols = [GREEN, GREEN, AMBER, RED]
bars = ax.bar(bands, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v}%", ha="center", fontweight="bold")
ax.set_ylim(0, 105); ax.set_ylabel("solve rate (best-of-5, 40 held-out tasks/band)")
ax.set_title("The 8B base, with the tool: reliable to n=5, a hard wall at n=6", fontsize=12.5)
ax.axhline(50, color=MUTE, lw=0.8, ls=":")
_clean(ax); fig.tight_layout(); fig.savefig(f"{OUT}/fig2_base_by_band.png"); plt.close(fig)

# ── Fig 3: THE FLYWHEEL DID NOT MOVE — overall, best-of-5 ─────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.3))
stages = ["base", "iter-1", "iter-2"]
overall = [58.1, 56.9, 58.1]
ax.plot(stages, overall, "-o", color=BLUE, lw=2.5, ms=9, label="overall (best-of-5)")
for x, v in zip(stages, overall):
    ax.text(x, v+1.4, f"{v}%", ha="center", fontweight="bold", color=BLUE)
ax.axhline(65, color=RED, lw=1.6, ls="--")
ax.text(2.02, 65, " target 65% (never reached)", va="center", color=RED, fontsize=10)
ax.set_ylim(40, 70); ax.set_ylabel("held-out solve rate")
ax.set_title("Self-harvest expert iteration: flat.\nThe flywheel did not improve held-out capability.", fontsize=12.5)
_clean(ax); fig.tight_layout(); fig.savefig(f"{OUT}/fig3_flywheel_flat.png"); plt.close(fig)

# ── Fig 4: THE MEASUREMENT LESSON — best-of-2 phantom vs best-of-5 truth (B4) ─
fig, ax = plt.subplots(figsize=(7.4, 4.4))
stages = ["base", "iter-1", "iter-2"]
b4_bo2 = [0.0, 7.5, 2.5]
b4_bo5 = [5.0, 7.5, 5.0]
ax.plot(stages, b4_bo2, "-o", color=RED, lw=2.5, ms=9, label="best-of-2 (under-sampled)")
ax.plot(stages, b4_bo5, "-s", color=GREEN, lw=2.5, ms=9, label="best-of-5 (fair)")
for x, v in zip(stages, b4_bo2): ax.text(x, v-1.3, f"{v}%", ha="center", color=RED, fontsize=10)
for x, v in zip(stages, b4_bo5): ax.text(x, v+1.0, f"{v}%", ha="center", color=GREEN, fontsize=10)
ax.annotate("phantom 'breakthrough':\nbase looked like 0%", xy=(0, 0), xytext=(0.25, 12),
            color=RED, fontsize=9.5, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_ylim(-2, 18); ax.set_ylabel("n=6 (B4) solve rate")
ax.set_title("Same models, two evals: under-sampling invented a B4 'breakthrough'\nthat a fair eval erased — the base already solved n=6", fontsize=12)
ax.legend(frameon=False, loc="upper right")
_clean(ax); fig.tight_layout(); fig.savefig(f"{OUT}/fig4_measurement_lesson.png"); plt.close(fig)

# ── Fig 5: HARVEST ROSE, GENERALIZATION DIDN'T ───────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 4.4))
models = ["base", "iter-1", "iter-2"]
harv_b3 = [35.8, 44.2, 49.2]      # training-task harvest yield (best-of-5)
eval_b3 = [40.0, 37.5, 35.0]      # held-out best-of-5
ax.plot(models, harv_b3, "-o", color=AMBER, lw=2.5, ms=9, label="B3 harvest yield (training tasks)")
ax.plot(models, eval_b3, "-s", color=BLUE, lw=2.5, ms=9, label="B3 held-out (best-of-5)")
for x, v in zip(models, harv_b3): ax.text(x, v+1.2, f"{v:.0f}%", ha="center", color=AMBER, fontsize=10)
for x, v in zip(models, eval_b3): ax.text(x, v-2.4, f"{v:.0f}%", ha="center", color=BLUE, fontsize=10)
ax.set_ylim(25, 58); ax.set_ylabel("n=5 (B3) solve rate")
ax.set_title("The trap: capability rose on TRAINING tasks (amber)\nbut held-out generalization stayed flat (blue)", fontsize=12)
ax.legend(frameon=False, loc="center right")
_clean(ax); fig.tight_layout(); fig.savefig(f"{OUT}/fig5_harvest_vs_eval.png"); plt.close(fig)

print("wrote 5 figures to", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
