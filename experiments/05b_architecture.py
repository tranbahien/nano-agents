"""Schematic: the nanoAgent control loop.

generate an answer and a calibrated belief; then loop: for each capability,
estimate expected accuracy after acting minus its cost (value of
information); take the best action if it beats answering now, execute it,
update the belief; otherwise commit. The control layer is Bayesian; the
capabilities below it are black boxes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, RED, GREEN, ORANGE, PURPLE, GRAY = (
    "#3a7ebf", "#c44e52", "#55a467", "#dd8452", "#9467bd", "#888888")


def box(ax, x, y, w, h, text, color, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.02",
                 linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.16))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#222")


def arrow(ax, x1, y1, x2, y2, color="#555", rad=0.0, text=None, dx=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.5, color=color,
                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, text, fontsize=8.5,
                color=color, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="white", ec="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.04, 0.80, 0.20, 0.12, "question", GRAY)
    box(ax, 0.04, 0.58, 0.20, 0.12, "generate\n(answer, conf)", BLUE)
    box(ax, 0.04, 0.36, 0.20, 0.12, "belief b\n= calibrate(conf)", PURPLE)
    arrow(ax, 0.14, 0.80, 0.14, 0.70)
    arrow(ax, 0.14, 0.58, 0.14, 0.48)

    # value-of-information selector
    box(ax, 0.34, 0.36, 0.30, 0.34,
        "value of information\n\nfor each action a:\n"
        "gain = E[acc after a] − b − cost(a)\n\npick best gain > 0,\nelse ANSWER",
        RED, fs=9)
    arrow(ax, 0.24, 0.42, 0.34, 0.46, text="b, type")

    # capabilities (black boxes)
    caps = [("tool\n(3b)", ORANGE, 0.76), ("retrieve\n(4a)", BLUE, 0.58),
            ("reflect\n(4b)", RED, 0.40), ("vote\n(3a/5a)", PURPLE, 0.22)]
    for lab, c, y in caps:
        box(ax, 0.74, y, 0.20, 0.13, lab, c, fs=9)
        arrow(ax, 0.64, 0.53, 0.74, y + 0.065, rad=0.12)
        arrow(ax, 0.74, y + 0.02, 0.64, 0.40, rad=-0.12)
    ax.text(0.84, 0.10, "capabilities (black-box)", ha="center", fontsize=9,
            color=GRAY, style="italic")

    # update + loop
    arrow(ax, 0.49, 0.36, 0.49, 0.20, text="execute,\nupdate b", dx=0.0)
    box(ax, 0.34, 0.06, 0.30, 0.12, "update belief, loop", GREEN)
    arrow(ax, 0.34, 0.12, 0.14, 0.36, rad=-0.3, text="not done")

    # answer
    box(ax, 0.04, 0.06, 0.20, 0.12, "commit answer", GREEN)
    arrow(ax, 0.34, 0.40, 0.24, 0.14, rad=0.2, text="ANSWER")

    ax.text(0.32, 0.97, "nanoAgent: a Bayes-consistent controller",
            ha="center", fontsize=13, weight="bold")
    ax.text(0.32, 0.925,
            "act only when an action's expected gain beats its cost",
            ha="center", fontsize=9.5, color=GRAY)

    out = Path("figures/05b_architecture.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
