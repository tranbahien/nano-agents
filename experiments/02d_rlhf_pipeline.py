"""Figure: RLHF pipeline schematic.

A diagram showing the canonical RLHF flow:
  1. Pre-trained / SFT policy generates pairs of completions.
  2. Humans choose preferred completions.
  3. Reward model trained on preferences (Bradley-Terry).
  4. Policy trained via PPO using the reward model + a KL penalty
     against the reference (SFT) policy.

Also indicates where DPO and GRPO bypass parts of this pipeline.
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
        linewidth=1.8, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            color=textcolor, fontsize=fontsize, fontweight=fontweight,
            zorder=4)


def draw_arrow(ax, src, dst, label="", color="#222", fontsize=9,
                label_offset=(0.0, 0.0), lw=1.5, curve=0.0):
    ax.annotate("", xy=dst, xytext=src,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                  connectionstyle=f"arc3,rad={curve}",
                                  shrinkA=5, shrinkB=5))
    if label:
        mx = 0.5 * (src[0] + dst[0]) + label_offset[0]
        my = 0.5 * (src[1] + dst[1]) + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=fontsize, color=color, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.5))

    # Top-left: Pre-trained / SFT policy
    draw_box(ax, (0.10, 0.78), 0.18, 0.14,
              "Pre-trained / SFT\npolicy  $\\pi_{\\rm ref}$",
              color="#fff2cc", fontsize=11, fontweight="bold")

    # Completions / preference data
    draw_box(ax, (0.36, 0.78), 0.18, 0.14,
              "Preference\npairs  $(x, y_w, y_l)$",
              color="#e0f0e0", fontsize=11)

    # Reward model
    draw_box(ax, (0.62, 0.78), 0.18, 0.14,
              "Reward model\n$\\hat r_\\phi(x, y)$",
              color="#fde2e2", fontsize=11, fontweight="bold")

    # Final policy
    draw_box(ax, (0.88, 0.78), 0.18, 0.14,
              "Final policy\n$\\pi_\\theta$",
              color="#e8f0fe", fontsize=11, fontweight="bold")

    # Arrows along the top row.
    draw_arrow(ax, (0.19, 0.78), (0.27, 0.78),
                label="sample\ncompletions", fontsize=8,
                label_offset=(0, 0.06))
    draw_arrow(ax, (0.45, 0.78), (0.53, 0.78),
                label="train via\nBradley-Terry NLL", fontsize=8,
                label_offset=(0, 0.06))
    draw_arrow(ax, (0.71, 0.78), (0.79, 0.78),
                label="PPO  +  reference KL", fontsize=8,
                label_offset=(0, 0.06))

    # KL penalty arrow from SFT to final policy (curved over the top).
    draw_arrow(ax, (0.10, 0.85), (0.88, 0.85),
                color="#888", curve=-0.35, fontsize=9,
                label=r"KL($\pi_\theta \,\|\, \pi_{\rm ref}$)  penalty",
                label_offset=(0, 0.07))

    # Bottom: DPO bypass.
    draw_arrow(ax, (0.36, 0.71), (0.88, 0.71),
                color="#3a7ebf", curve=0.4, fontsize=10, lw=2.2,
                label="DPO: skip reward model entirely",
                label_offset=(0, -0.05))

    # Bottom: GRPO bypass.
    # GRPO uses reward (model OR oracle) but no separate critic.
    # Show as text annotation.
    ax.text(0.50, 0.20,
            "GRPO:  PPO + group-relative baseline\n"
            "         (no learned critic, no explicit value function)",
            fontsize=11, color="#c44e52", ha="center",
            bbox=dict(facecolor="#fff5f5", edgecolor="#c44e52", boxstyle="round,pad=0.5"))

    # DPO box
    ax.text(0.50, 0.08,
            "DPO:  closed-form preference loss on $\\pi_\\theta$\n"
            "         (no reward model, no PPO loop)",
            fontsize=11, color="#3a7ebf", ha="center",
            bbox=dict(facecolor="#f0f5ff", edgecolor="#3a7ebf", boxstyle="round,pad=0.5"))

    # Section labels.
    ax.text(0.50, 0.36, "Variations on the recipe:",
            fontsize=11, color="#222", ha="center", style="italic")

    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.axis("off")

    fig.suptitle(
        "The RLHF stack and its variants. "
        "Standard pipeline: sample → preferences → reward model → PPO. "
        "DPO and GRPO simplify different pieces.",
        fontsize=12, y=0.99,
    )
    fig.tight_layout()
    out = Path("figures/02d_rlhf_pipeline.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
