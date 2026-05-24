"""Figure: Bandit problem schematic.

Shows K slot machines with hidden true means and visible observed samples.
Used as the hero intuition figure in Post 1a.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# Update matplotlib configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"], 
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True 
})



def main() -> None:
    np.random.seed(7)

    K = 5
    true_mus = [0.3, 0.5, 0.7, 0.75, 0.8]  # hidden from the agent
    sample_counts = [12, 9, 6, 14, 8]
    samples = [np.random.binomial(1, mu, n) for mu, n in zip(true_mus, sample_counts)]

    fig, ax = plt.subplots(figsize=(9, 3.5))

    box_w, box_h, gap = 1.6, 3.2, 0.4
    for i, (s, n) in enumerate(zip(samples, sample_counts)):
        x0 = i * (box_w + gap)
        # Slot-machine body
        body = patches.FancyBboxPatch(
            (x0, 0), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#f5f5f7", edgecolor="#222", linewidth=1.5,
        )
        ax.add_patch(body)

        # Arm label
        ax.text(x0 + box_w / 2, box_h - 0.25, f"Arm {i + 1}",
                ha="center", va="center", fontsize=13, fontweight="bold")

        # True mean (hidden)
        ax.text(x0 + box_w / 2, box_h - 0.7, r"$\mu_k = \;?$",
                ha="center", va="center", fontsize=12, color="#999")

        # Empirical mean
        mean = s.mean()
        ax.text(x0 + box_w / 2, box_h - 1.15, rf"$\hat\mu_k = {mean:.2f}$",
                ha="center", va="center", fontsize=11, color="#222")
        ax.text(x0 + box_w / 2, box_h - 1.5, f"n = {n} pulls",
                ha="center", va="center", fontsize=9, color="#888")

        # Sample reward dots (green = 1, red = 0)
        per_row = 6
        dot_x0 = x0 + 0.18
        dot_y0 = 0.35
        for j, r in enumerate(s):
            col, row = j % per_row, j // per_row
            color = "#2a9d4a" if r == 1 else "#c83737"
            ax.plot(dot_x0 + col * 0.22, dot_y0 + row * 0.35,
                    "o", color=color, markersize=8, markeredgecolor="white")

    # Legend
    ax.plot([], [], "o", color="#2a9d4a", markersize=10, label="reward = 1")
    ax.plot([], [], "o", color="#c83737", markersize=10, label="reward = 0")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, frameon=False, fontsize=11)

    ax.set_xlim(-0.3, K * (box_w + gap))
    ax.set_ylim(-0.4, box_h + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "The multi-armed bandit: K arms with unknown reward distributions.\n"
        "You observe rewards $r_t \\sim P_{a_t}$ but never the true means $\\mu_k$.",
        fontsize=12, pad=10,
    )

    fig.tight_layout()
    out = Path("figures/01a_bandit_schematic.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
