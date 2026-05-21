"""Figure: POMDP schematic.

Extends the MDP schematic from 1c by adding an observation layer. The
agent observes 'o', a noisy function of the true state s, rather than s
directly. The same dynamics underneath, but now learning is harder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_state(ax, xy, label, radius=0.18, color="#f5f5f7"):
    circle = patches.Circle(xy, radius, facecolor=color, edgecolor="#222",
                              linewidth=2.0, zorder=3)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=12, fontweight="bold", zorder=4)


def draw_arrow(ax, src, dst, label="", color="#3a7ebf",
                curve_rad=0.0, label_offset=(0.0, 0.0), fontsize=9,
                style="-|>"):
    arrowstyle = style
    ax.annotate("", xy=dst, xytext=src,
                 arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                  lw=1.8,
                                  connectionstyle=f"arc3,rad={curve_rad}",
                                  shrinkA=10, shrinkB=10))
    if label:
        mx = 0.5 * (src[0] + dst[0]) + label_offset[0]
        my = 0.5 * (src[1] + dst[1]) + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, color=color,
                ha="center", va="center", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                              gridspec_kw={"width_ratios": [1, 1]})

    # ----- Left: MDP -----
    ax = axes[0]
    ax.set_title("MDP — agent sees the true state", fontsize=12)
    positions = {"S0": (0.2, 0.85), "S1": (0.8, 0.85)}
    for name, pos in positions.items():
        draw_state(ax, pos, name)
    draw_arrow(ax, positions["S0"], positions["S1"],
                label="a, r", color="#3a7ebf", curve_rad=0.18,
                label_offset=(0, 0.08), fontsize=10)
    draw_arrow(ax, positions["S1"], positions["S0"],
                label="a, r", color="#3a7ebf", curve_rad=0.18,
                label_offset=(0, -0.08), fontsize=10)

    # "Agent sees s" arrows
    eye_y = 0.30
    ax.text(0.5, eye_y + 0.06, "agent", ha="center", fontsize=11,
            fontweight="bold")
    draw_arrow(ax, (0.2, 0.85 - 0.18), (0.4, eye_y), color="#222",
                style="->", curve_rad=-0.2)
    draw_arrow(ax, (0.8, 0.85 - 0.18), (0.6, eye_y), color="#222",
                style="->", curve_rad=0.2)
    ax.text(0.5, 0.03, "agent observes $s_t$ directly",
            fontsize=10, color="#666", style="italic", ha="center")

    ax.set_xlim(0, 1.0); ax.set_ylim(0.05, 1.15)
    ax.set_aspect("equal"); ax.axis("off")

    # ----- Right: POMDP -----
    ax = axes[1]
    ax.set_title("POMDP — agent sees a noisy observation", fontsize=12)
    positions = {"S0": (0.2, 0.85), "S1": (0.8, 0.85)}
    for name, pos in positions.items():
        draw_state(ax, pos, name, color="#fff2cc")
    draw_arrow(ax, positions["S0"], positions["S1"],
                label="a, r", color="#3a7ebf", curve_rad=0.18,
                label_offset=(0, 0.08), fontsize=10)
    draw_arrow(ax, positions["S1"], positions["S0"],
                label="a, r", color="#3a7ebf", curve_rad=0.18,
                label_offset=(0, -0.08), fontsize=10)

    # Observation layer.
    obs_positions = {"o0": (0.35, 0.55), "o1": (0.65, 0.55)}
    for name, pos in obs_positions.items():
        rect = patches.Rectangle((pos[0] - 0.07, pos[1] - 0.07), 0.14, 0.14,
                                  facecolor="#e8f0fe", edgecolor="#3a7ebf",
                                  linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], "$o_t$", ha="center", va="center",
                fontsize=11, fontweight="bold")

    # Stochastic emission: each state can produce either observation.
    draw_arrow(ax, positions["S0"], obs_positions["o0"], color="#888",
                style="->", curve_rad=0.0, label="Z", fontsize=8,
                label_offset=(-0.05, 0.05))
    draw_arrow(ax, positions["S0"], obs_positions["o1"], color="#bbb",
                style="->", curve_rad=0.0)
    draw_arrow(ax, positions["S1"], obs_positions["o0"], color="#bbb",
                style="->", curve_rad=0.0)
    draw_arrow(ax, positions["S1"], obs_positions["o1"], color="#888",
                style="->", curve_rad=0.0)

    # Agent sees observations.
    eye_y = 0.18
    ax.text(0.5, eye_y + 0.06, "agent", ha="center", fontsize=11,
            fontweight="bold")
    draw_arrow(ax, obs_positions["o0"], (0.45, eye_y),
                color="#222", style="->", curve_rad=-0.1)
    draw_arrow(ax, obs_positions["o1"], (0.55, eye_y),
                color="#222", style="->", curve_rad=0.1)

    ax.text(0.5, 0.03, "$s_t$ is hidden — agent must infer it",
            fontsize=10, color="#666", style="italic", ha="center")

    ax.set_xlim(0, 1.0); ax.set_ylim(0.0, 1.15)
    ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle(
        "From MDP to POMDP. In the POMDP, the agent must reason about "
        "what state it's in, not just what to do.",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    out = Path("figures/01d_pomdp_schematic.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
