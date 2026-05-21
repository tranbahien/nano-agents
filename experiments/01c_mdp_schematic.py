"""Figure: MDP schematic.

Contrasts a multi-armed bandit with a small MDP. The point is that in an
MDP, actions don't just yield rewards — they change which state you're in
next, and that affects what you can do from there.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def draw_state(ax, xy, label, radius=0.18, color="#f5f5f7"):
    circle = patches.Circle(xy, radius, facecolor=color, edgecolor="#222",
                              linewidth=2.0, zorder=3)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=12, fontweight="bold", zorder=4)


def draw_action_arrow(ax, src, dst, label="", color="#3a7ebf",
                       curve_rad=0.0, label_offset=(0.0, 0.0), fontsize=9):
    style = "arc3," + f"rad={curve_rad}"
    ax.annotate("", xy=dst, xytext=src,
                 arrowprops=dict(arrowstyle="-|>", color=color,
                                  lw=1.8, connectionstyle=style,
                                  shrinkA=10, shrinkB=10))
    if label:
        # Place label along the line.
        mx = 0.5 * (src[0] + dst[0]) + label_offset[0]
        my = 0.5 * (src[1] + dst[1]) + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, color=color,
                ha="center", va="center", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                              gridspec_kw={"width_ratios": [1, 1.3]})

    # ----- Left: bandit -----
    ax = axes[0]
    ax.set_title("Multi-armed bandit (no state)", fontsize=12)
    draw_state(ax, (0.5, 0.5), "S", color="#e9ecef")
    arm_positions = [(0.15, 0.95), (0.5, 1.05), (0.85, 0.95)]
    for i, ap in enumerate(arm_positions):
        draw_action_arrow(ax, (0.5, 0.5), ap, label=f"a={i+1}",
                          color="#3a7ebf", curve_rad=0.0,
                          label_offset=(0.06, 0.0), fontsize=10)
        ax.text(ap[0], ap[1] + 0.07, rf"$r \sim P_{i+1}$",
                ha="center", fontsize=10, color="#444")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.0, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, 0.05, "Action returns reward; next state = same state",
            ha="center", fontsize=10, style="italic", color="#666")

    # ----- Right: MDP -----
    ax = axes[1]
    ax.set_title("Markov Decision Process", fontsize=12)
    # Four states laid out — diamond shape.
    positions = {
        "S0": (0.1, 0.6),
        "S1": (0.5, 1.05),
        "S2": (0.5, 0.15),
        "S3": (0.95, 0.6),
    }
    for name, pos in positions.items():
        draw_state(ax, pos, name)

    # Transitions for state S0:
    #   action 'a' → S1 w.p. 0.8, S2 w.p. 0.2
    #   action 'b' → S2 w.p. 1.0
    draw_action_arrow(ax, positions["S0"], positions["S1"],
                       label="a: 0.8", color="#3a7ebf",
                       curve_rad=0.15, label_offset=(-0.04, 0.06), fontsize=9)
    draw_action_arrow(ax, positions["S0"], positions["S2"],
                       label="a: 0.2", color="#3a7ebf",
                       curve_rad=-0.15, label_offset=(-0.04, -0.06), fontsize=9)
    draw_action_arrow(ax, positions["S0"], positions["S2"],
                       label="b: 1.0", color="#dd8452",
                       curve_rad=0.25, label_offset=(0.08, 0.04), fontsize=9)

    # S1 → S3 deterministic.
    draw_action_arrow(ax, positions["S1"], positions["S3"],
                       label="any: 1.0", color="#222", fontsize=9,
                       label_offset=(0.04, 0.06))

    # S2 → S3 with reward.
    draw_action_arrow(ax, positions["S2"], positions["S3"],
                       label="any: 1.0", color="#222", fontsize=9,
                       label_offset=(0.04, -0.06))

    # Reward annotations next to terminal/intermediate.
    ax.text(0.95, 0.85, "r = +1", ha="center", fontsize=10,
            fontweight="bold", color="#c44e52")
    ax.text(0.5, -0.05, "via S2: r = 0", ha="center", fontsize=9, color="#666")
    ax.text(0.5, 1.25, "via S1: r = 0", ha="center", fontsize=9, color="#666")

    ax.set_xlim(-0.1, 1.25)
    ax.set_ylim(-0.2, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.55, -0.15, "Action's next state depends on current state;\n"
            "future depends on past actions",
            ha="center", fontsize=10, style="italic", color="#666")

    fig.suptitle("Bandits assume one state. MDPs have many, "
                 "and actions move you between them.",
                 fontsize=13, y=1.04)
    fig.tight_layout()

    out = Path("figures/01c_mdp_schematic.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
