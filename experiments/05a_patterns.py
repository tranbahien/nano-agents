"""Schematic: four ways to wire multiple agents together.

  - Ensemble / vote: independent agents, aggregate by majority (Post 3a's
    self-consistency, across agents).
  - Debate: agents see each other and revise (Post 4b's reflection, with peer
    critics).
  - Pipeline: agents in series, each a specialist stage (errors compound).
  - Orchestrator-workers: a manager decomposes and delegates, then integrates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.16))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9,
            color="#222")


def arrow(ax, x1, y1, x2, y2, color="#666", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=12, linewidth=1.4, color=color,
                 connectionstyle=f"arc3,rad={rad}"))


BLUE, RED, GREEN, ORANGE, GRAY = "#3a7ebf", "#c44e52", "#55a467", "#dd8452", "#888888"


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 1. Ensemble / vote
    ax = axes[0]; ax.set_title("Ensemble / vote", fontsize=11, color=GRAY)
    box(ax, 0.32, 0.86, 0.36, 0.12, "task", GRAY)
    for x in (0.04, 0.28, 0.52, 0.76):
        box(ax, x, 0.56, 0.18, 0.13, "agent", BLUE)
        arrow(ax, 0.5, 0.86, x + 0.09, 0.69)
        arrow(ax, x + 0.09, 0.56, 0.5, 0.42)
    box(ax, 0.30, 0.26, 0.40, 0.13, "majority vote", GREEN)
    arrow(ax, 0.5, 0.26, 0.5, 0.16)
    box(ax, 0.34, 0.02, 0.32, 0.12, "answer", GREEN)

    # 2. Debate
    ax = axes[1]; ax.set_title("Debate", fontsize=11, color=GRAY)
    box(ax, 0.18, 0.74, 0.28, 0.14, "agent A", BLUE)
    box(ax, 0.54, 0.74, 0.28, 0.14, "agent B", BLUE)
    box(ax, 0.18, 0.44, 0.28, 0.14, "agent C", BLUE)
    box(ax, 0.54, 0.44, 0.28, 0.14, "agent D", BLUE)
    arrow(ax, 0.46, 0.81, 0.54, 0.81, RED, 0.0)
    arrow(ax, 0.54, 0.75, 0.46, 0.75, RED, 0.0)
    arrow(ax, 0.32, 0.74, 0.32, 0.58, RED)
    arrow(ax, 0.68, 0.74, 0.68, 0.58, RED)
    arrow(ax, 0.46, 0.51, 0.54, 0.51, RED)
    ax.text(0.5, 0.30, "revise toward\neach other\n× rounds", ha="center",
            fontsize=8.5, color=RED)
    box(ax, 0.30, 0.06, 0.40, 0.13, "consensus", GREEN)

    # 3. Pipeline
    ax = axes[2]; ax.set_title("Pipeline (specialists)", fontsize=11, color=GRAY)
    labels = ["plan", "research", "write", "check"]
    cols = [BLUE, ORANGE, GREEN, RED]
    for i, (lab, c) in enumerate(zip(labels, cols)):
        y = 0.78 - i * 0.20
        box(ax, 0.28, y, 0.44, 0.13, lab, c)
        if i > 0:
            arrow(ax, 0.5, y + 0.20, 0.5, y + 0.13)
    ax.text(0.5, 0.02, "every stage must succeed → $p^L$", ha="center",
            fontsize=8.5, color=GRAY)

    # 4. Orchestrator-workers
    ax = axes[3]; ax.set_title("Orchestrator–workers", fontsize=11, color=GRAY)
    box(ax, 0.30, 0.80, 0.40, 0.14, "orchestrator", RED)
    for x in (0.06, 0.40, 0.74):
        box(ax, x, 0.44, 0.20, 0.14, "worker", BLUE)
        arrow(ax, 0.5, 0.80, x + 0.10, 0.58)
        arrow(ax, x + 0.10, 0.44, 0.5, 0.24, rad=0.0)
    box(ax, 0.30, 0.08, 0.40, 0.14, "integrate", GREEN)

    fig.suptitle(
        "Four ways to wire agents together. Ensemble aggregates independent "
        "votes; debate revises toward peers;\npipeline chains specialists; "
        "an orchestrator decomposes and delegates.",
        fontsize=11, y=1.04)
    fig.tight_layout()
    out = Path("figures/05a_patterns.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
