"""Figure: actor-critic architecture diagram.

Shows the two interacting components — actor (policy) and critic (value) —
with arrows for the data flow between them and the environment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_box(ax, xy, w, h, text, color="#e9ecef", textcolor="#222",
              fontsize=11, fontweight="normal", border="#222"):
    rect = patches.FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02", facecolor=color, edgecolor=border,
        linewidth=1.8, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            color=textcolor, fontsize=fontsize, fontweight=fontweight,
            zorder=4)


def draw_arrow(ax, src, dst, label="", color="#222", fontsize=9,
                label_offset=(0.0, 0.0), lw=1.6, curve=0.0):
    ax.annotate("", xy=dst, xytext=src,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                  connectionstyle=f"arc3,rad={curve}",
                                  shrinkA=4, shrinkB=4))
    if label:
        mx = 0.5 * (src[0] + dst[0]) + label_offset[0]
        my = 0.5 * (src[1] + dst[1]) + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=fontsize, color=color, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Environment box (left)
    draw_box(ax, (0.13, 0.55), 0.18, 0.35,
              "Environment\n(MDP)", color="#fff2cc", fontsize=11,
              fontweight="bold")

    # Actor box (top right)
    draw_box(ax, (0.50, 0.82), 0.22, 0.20,
              r"Actor:  $\pi_\theta(a \mid s)$",
              color="#e8f0fe", fontsize=12, fontweight="bold")

    # Critic box (bottom right)
    draw_box(ax, (0.50, 0.28), 0.22, 0.20,
              r"Critic:  $V_\phi(s)$",
              color="#fde2e2", fontsize=12, fontweight="bold")

    # Trajectory data (middle)
    draw_box(ax, (0.86, 0.55), 0.20, 0.35,
              "Trajectory\n$(s_t, a_t, r_{t+1})$\n+ bootstrap",
              color="#e0f0e0", fontsize=10)

    # Arrows.
    # Environment → actor (state to act)
    draw_arrow(ax, (0.22, 0.62), (0.39, 0.82),
                label=r"$s_t$", color="#222", fontsize=10,
                label_offset=(-0.03, 0.04))
    # Actor → environment (sample action)
    draw_arrow(ax, (0.40, 0.78), (0.22, 0.58),
                label=r"$a_t \sim \pi_\theta$", color="#3a7ebf", fontsize=10,
                label_offset=(-0.04, -0.04))
    # Environment → trajectory storage
    draw_arrow(ax, (0.22, 0.55), (0.76, 0.55),
                label=r"$r_{t+1}, s_{t+1}$", color="#222", fontsize=10,
                label_offset=(0.0, 0.04))
    # Trajectory → critic (TD target)
    draw_arrow(ax, (0.86, 0.38), (0.61, 0.28),
                label=r"target = $r + \gamma V_\phi(s')$",
                color="#c44e52", fontsize=10,
                label_offset=(0.0, -0.05), curve=0.15)
    # Critic → trajectory storage (provides V values for GAE)
    draw_arrow(ax, (0.61, 0.32), (0.84, 0.50),
                label=r"$V_\phi(s_t), V_\phi(s_{t+1})$",
                color="#c44e52", fontsize=9,
                label_offset=(0.02, 0.0), curve=-0.1)
    # Trajectory → actor (advantage)
    draw_arrow(ax, (0.86, 0.72), (0.61, 0.82),
                label=r"$A_t = G_t - V_\phi(s_t)$",
                color="#3a7ebf", fontsize=10,
                label_offset=(0.0, 0.05), curve=-0.15)
    # Actor self-update
    draw_arrow(ax, (0.40, 0.86), (0.40, 0.94),
                color="#3a7ebf")
    draw_arrow(ax, (0.40, 0.94), (0.60, 0.94),
                color="#3a7ebf")
    draw_arrow(ax, (0.60, 0.94), (0.60, 0.86),
                color="#3a7ebf",
                label=r"$\theta \leftarrow \theta + \alpha A_t \nabla_\theta \log \pi_\theta$",
                fontsize=10, label_offset=(0.0, 0.04))

    # Critic self-update
    draw_arrow(ax, (0.40, 0.24), (0.40, 0.16),
                color="#c44e52")
    draw_arrow(ax, (0.40, 0.16), (0.60, 0.16),
                color="#c44e52")
    draw_arrow(ax, (0.60, 0.16), (0.60, 0.24),
                color="#c44e52",
                label=r"$\phi \leftarrow \phi - \alpha_\phi \nabla_\phi (V_\phi(s) - \text{target})^2$",
                fontsize=10, label_offset=(0.0, -0.04))

    ax.text(0.50, 0.55,
            "shared\ntrajectory",
            fontsize=10, color="#555", style="italic",
            ha="center", va="center")

    ax.set_xlim(0, 1.05); ax.set_ylim(0.05, 1.05)
    ax.axis("off")

    fig.suptitle(
        "Actor-Critic: two networks learn together. "
        "Actor improves on advantages; critic learns to predict returns.",
        fontsize=13, y=0.99,
    )
    fig.tight_layout()
    out = Path("figures/02b_actor_critic_schematic.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
