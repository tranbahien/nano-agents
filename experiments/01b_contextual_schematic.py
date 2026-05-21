"""Figure: Contextual bandit schematic.

Shows that with context-dependent rewards, the optimal arm changes per
context — the central thing that distinguishes contextual bandits from
plain multi-armed bandits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Hand-picked arms for clarity:
    # arm 1 prefers feature 1, arm 2 prefers feature 2, arm 3 likes both,
    # arm 4 likes feature 2 but dislikes feature 1.
    thetas = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.7],
        [-0.7, 0.7],
    ])

    contexts = {
        "Context A: $x = (0.8, 0.6)$": np.array([0.8, 0.6]),
        "Context B: $x = (-0.7, 0.7)$": np.array([-0.7, 0.7]),
    }
    arm_colors = ["#4c72b0", "#dd8452", "#55a467", "#c44e52"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5),
                              gridspec_kw={"width_ratios": [1, 2.5],
                                           "wspace": 0.3, "hspace": 0.5})

    for row, (label, x) in enumerate(contexts.items()):
        # Left column: the context vector visualized as a small bar plot.
        ax_x = axes[row, 0]
        ax_x.bar([0, 1], x, color="#666", edgecolor="white", width=0.6)
        ax_x.axhline(0, color="#222", linewidth=0.8)
        ax_x.set_xticks([0, 1])
        ax_x.set_xticklabels(["$x_1$", "$x_2$"], fontsize=11)
        ax_x.set_ylim(-1.0, 1.0)
        ax_x.set_title(label, fontsize=12)
        ax_x.set_ylabel("feature value")
        ax_x.grid(alpha=0.3, axis="y")
        ax_x.set_axisbelow(True)

        # Right column: predicted reward x^T theta_k for each arm.
        ax_r = axes[row, 1]
        scores = thetas @ x
        best = int(np.argmax(scores))
        colors = [arm_colors[k] if k == best else "#cccccc" for k in range(4)]
        ax_r.bar(range(4), scores, color=colors, edgecolor="white", width=0.7)
        ax_r.axhline(0, color="#222", linewidth=0.8)
        ax_r.set_xticks(range(4))
        ax_r.set_xticklabels([f"Arm {k+1}" for k in range(4)], fontsize=11)
        ax_r.set_ylim(-1.0, 1.2)
        ax_r.set_ylabel(r"predicted reward $x^\top \theta_k$")
        ax_r.grid(alpha=0.3, axis="y")
        ax_r.set_axisbelow(True)
        for k, s in enumerate(scores):
            ax_r.text(k, s + (0.05 if s >= 0 else -0.12),
                      f"{s:.2f}", ha="center", fontsize=10,
                      fontweight="bold" if k == best else "normal",
                      color="#222" if k == best else "#777")
        ax_r.set_title(f"Best arm: Arm {best+1}", fontsize=12, color=arm_colors[best])

        # Annotate which arm wins
        ax_r.annotate("", xy=(best, scores[best] + 0.5),
                       xytext=(best, scores[best] + 0.9),
                       arrowprops=dict(arrowstyle="->", color=arm_colors[best], lw=2))

    fig.suptitle(
        "Contextual bandits: the optimal arm depends on context.\n"
        "Same four arms, two different contexts, two different winners.",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()

    out = Path("figures/01b_contextual_schematic.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
