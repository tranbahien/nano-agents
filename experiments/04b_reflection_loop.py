"""Schematic: three ways to spend extra inference compute.

Single-shot generates once. Best-of-N samples in parallel and a verifier
picks the winner. Self-reflection generates, then critiques and revises
sequentially until the self-critic accepts. Reflection is the *adaptive,
sequential* sibling of best-of-N — it spends compute only where its own
judgement says more is needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, color, fc=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.6, edgecolor=color,
                       facecolor=color, alpha=0.16)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5,
            color="#222")


def arrow(ax, x1, y1, x2, y2, color="#555", text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.5, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, text, ha="center",
                fontsize=8.5, color=color)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- single-shot ---
    ax = axes[0]
    ax.set_title("Single-shot", fontsize=12, color="#888")
    box(ax, 0.30, 0.70, 0.40, 0.16, "prompt", "#888")
    box(ax, 0.30, 0.42, 0.40, 0.16, "generate", "#3a7ebf")
    box(ax, 0.30, 0.14, 0.40, 0.16, "answer", "#55a467")
    arrow(ax, 0.5, 0.70, 0.5, 0.58)
    arrow(ax, 0.5, 0.42, 0.5, 0.30)

    # --- best-of-N (parallel) ---
    ax = axes[1]
    ax.set_title("Best-of-N (parallel)", fontsize=12, color="#888")
    box(ax, 0.30, 0.82, 0.40, 0.13, "prompt", "#888")
    for i, x in enumerate([0.06, 0.30, 0.54, 0.78]):
        box(ax, x, 0.52, 0.16, 0.14, f"gen {i+1}", "#3a7ebf")
        arrow(ax, 0.5, 0.82, x + 0.08, 0.66, rad=0.0)
    box(ax, 0.28, 0.28, 0.44, 0.13, "verifier picks best", "#dd8452")
    for x in [0.14, 0.38, 0.62, 0.86]:
        arrow(ax, x, 0.52, 0.5, 0.41)
    box(ax, 0.30, 0.04, 0.40, 0.13, "answer", "#55a467")
    arrow(ax, 0.5, 0.28, 0.5, 0.17)

    # --- reflection (sequential) ---
    ax = axes[2]
    ax.set_title("Self-reflection (sequential)", fontsize=12, color="#888")
    box(ax, 0.28, 0.80, 0.44, 0.14, "generate", "#3a7ebf")
    box(ax, 0.28, 0.50, 0.44, 0.14, "self-critique", "#c44e52")
    box(ax, 0.06, 0.50, 0.16, 0.14, "revise", "#dd8452")
    box(ax, 0.28, 0.16, 0.44, 0.14, "accept → answer", "#55a467")
    arrow(ax, 0.5, 0.80, 0.5, 0.64)
    arrow(ax, 0.5, 0.50, 0.5, 0.30, text="ok")
    arrow(ax, 0.28, 0.57, 0.22, 0.57, text="wrong", color="#c44e52")
    arrow(ax, 0.14, 0.64, 0.30, 0.80, color="#dd8452", rad=-0.3)

    fig.suptitle(
        "Three ways to spend extra inference compute. Reflection is the adaptive, "
        "sequential sibling of best-of-N:\nit revises only when its own critic objects.",
        fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path("figures/04b_reflection_loop.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
