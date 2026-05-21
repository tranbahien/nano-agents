"""Figure: policy gradient schematic.

Compares the gradient structures of supervised learning and policy gradient:

  Supervised learning: gradient of log-likelihood with all weights = 1.
  Policy gradient:     gradient of log pi weighted by the return R(tau).

The visual makes the structural similarity (and the weighting difference) explicit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_box(ax, xy, w, h, text, color="#e9ecef", textcolor="#222",
              fontsize=10, fontweight="normal", border="#222"):
    rect = patches.FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2), w, h,
        boxstyle="round,pad=0.02", facecolor=color, edgecolor=border,
        linewidth=1.5, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            color=textcolor, fontsize=fontsize, fontweight=fontweight,
            zorder=4)


def draw_arrow(ax, src, dst, label="", color="#222", fontsize=10,
                label_offset=(0.0, 0.0), lw=1.5):
    ax.annotate("", xy=dst, xytext=src,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                  shrinkA=2, shrinkB=2))
    if label:
        mx = 0.5 * (src[0] + dst[0]) + label_offset[0]
        my = 0.5 * (src[1] + dst[1]) + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=fontsize, color=color, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8))

    # ----- Top: Supervised learning -----
    ax = axes[0]
    ax.set_title("Supervised learning (classification)", fontsize=12)

    draw_box(ax, (0.10, 0.55), 0.16, 0.30,
              "input  $x$", color="#e8f0fe")
    draw_box(ax, (0.32, 0.55), 0.16, 0.30,
              "model  $p_\\theta(y \\mid x)$",
              color="#fff2cc", fontweight="bold")
    draw_box(ax, (0.55, 0.55), 0.16, 0.30,
              "label  $y$", color="#e8f0fe")
    draw_box(ax, (0.84, 0.55), 0.25, 0.30,
              r"loss: $-\log p_\theta(y\mid x)$",
              color="#ffe5e5", fontweight="bold")

    draw_arrow(ax, (0.18, 0.55), (0.24, 0.55))
    draw_arrow(ax, (0.40, 0.55), (0.47, 0.55))
    draw_arrow(ax, (0.63, 0.55), (0.71, 0.55))

    ax.text(0.5, 0.18,
            r"$\nabla_\theta J = -\, \mathbb{E}_{(x, y)}\left[\nabla_\theta \log p_\theta(y \mid x)\right]$",
            ha="center", fontsize=14, color="#222")
    ax.text(0.5, 0.05, "Every example contributes equally.",
            ha="center", fontsize=10, style="italic", color="#666")

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 0.9)
    ax.axis("off")

    # ----- Bottom: Policy gradient -----
    ax = axes[1]
    ax.set_title("Policy gradient (REINFORCE)", fontsize=12)

    draw_box(ax, (0.08, 0.55), 0.14, 0.30,
              "state  $s_t$", color="#e8f0fe")
    draw_box(ax, (0.27, 0.55), 0.16, 0.30,
              "policy  $\\pi_\\theta(a \\mid s)$",
              color="#fff2cc", fontweight="bold")
    draw_box(ax, (0.48, 0.55), 0.14, 0.30,
              "action  $a_t$", color="#e8f0fe")
    draw_box(ax, (0.68, 0.55), 0.20, 0.30,
              r"return  $G_t = \sum \gamma^k r_{t+k+1}$",
              color="#e0f0e0", fontsize=10)
    draw_box(ax, (0.92, 0.55), 0.13, 0.30,
              "weight",
              color="#ffe5e5", fontweight="bold")

    draw_arrow(ax, (0.15, 0.55), (0.19, 0.55))
    draw_arrow(ax, (0.35, 0.55), (0.41, 0.55))
    draw_arrow(ax, (0.55, 0.55), (0.58, 0.55), label="sample\nrollout",
                fontsize=8, label_offset=(0.0, 0.10))
    draw_arrow(ax, (0.78, 0.55), (0.85, 0.55))

    ax.text(0.5, 0.18,
            r"$\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t G_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$",
            ha="center", fontsize=14, color="#222")
    ax.text(0.5, 0.05,
            "Same log-prob gradient — but weighted by the return that followed.",
            ha="center", fontsize=10, style="italic", color="#666")

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 0.9)
    ax.axis("off")

    fig.suptitle(
        "Supervised learning vs policy gradient — same gradient structure, "
        "different weights.",
        fontsize=13, y=0.99,
    )
    fig.tight_layout()

    out = Path("figures/02a_pg_schematic.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
