"""Figure: schematic of autoregressive LM generation as a tree.

Visualizes how each token choice branches the space of possible sequences.
For a vocabulary of V and length L, there are V^L sequences; we draw a
small tree showing the first 2-3 levels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from nano_agents.decoding import TinyMarkovLM


def main() -> None:
    lm = TinyMarkovLM(vocab_size=5, seq_length=3, seed=7)

    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.set_xlim(-0.5, lm.L + 0.5); ax.set_ylim(-0.5, lm.V + 0.5)
    ax.axis("off")

    # Root: BOS at (0, V/2)
    root_y = lm.V / 2
    ax.scatter([0], [root_y], s=400, color="#3a7ebf", zorder=5,
                edgecolor="white", linewidth=2)
    ax.text(0, root_y, "BOS", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold", zorder=6)

    # Greedy path
    cur = lm.BOS
    greedy = [cur]
    for _ in range(lm.L):
        p = lm.conditional_probs(cur, T=1.0)
        t = int(p.argmax())
        greedy.append(t)
        cur = t

    # Layout: at each step, draw all non-BOS tokens, with size proportional
    # to conditional probability from the parent (for top branches).
    parent_positions = {(0, lm.BOS): root_y}
    for step in range(lm.L):
        new_positions = {}
        # Get all branch positions for this step.
        # For visualization clarity, only show branches from the *top 2*
        # parents at each step.
        parent_keys = list(parent_positions.keys())
        if step > 0:
            parent_keys = parent_keys[:2]
        for (parent_step, parent_token) in parent_keys:
            parent_y = parent_positions[(parent_step, parent_token)]
            probs = lm.conditional_probs(parent_token, T=1.0)
            # Show top-3 branches.
            topk = np.argsort(probs)[::-1][:3]
            for j, t in enumerate(topk):
                child_y = parent_y + (j - 1) * 1.0  # spread vertically
                child_y = max(0.5, min(lm.V - 0.5, child_y))
                x = step + 1
                # Edge from parent to child.
                is_greedy = (parent_token == greedy[step]
                             and int(t) == greedy[step + 1])
                color = "#c44e52" if is_greedy else "#aaaaaa"
                lw = 2.5 if is_greedy else 1.2
                ax.plot([parent_step, x], [parent_y, child_y],
                         color=color, linewidth=lw, alpha=0.85)
                # Edge label: probability.
                mx, my = 0.5 * (parent_step + x), 0.5 * (parent_y + child_y)
                ax.text(mx, my + 0.08, f"{probs[t]:.2f}",
                         ha="center", fontsize=8, color="#222",
                         bbox=dict(facecolor="white", edgecolor="none", pad=1))
                # Node circle.
                ax.scatter([x], [child_y], s=270, color="#dd8452", zorder=5,
                            edgecolor="white", linewidth=1.5)
                ax.text(x, child_y, f"t{int(t)}", ha="center",
                         va="center", color="white", fontsize=9,
                         fontweight="bold", zorder=6)
                new_positions[(x, int(t))] = child_y
        parent_positions = new_positions

    # Annotate greedy path label.
    ax.text(lm.L / 2 + 0.5, lm.V + 0.0, "greedy / MAP path",
            color="#c44e52", fontsize=11, fontweight="bold",
            ha="center")

    # Step labels
    for s in range(lm.L + 1):
        label = "BOS" if s == 0 else f"step {s}"
        ax.text(s, -0.3, label, ha="center", fontsize=10, color="#222")

    ax.set_title(
        "Autoregressive generation: each token chooses a branch. "
        f"With V = {lm.V} and L = {lm.L} there are {(lm.V-1) ** lm.L} total sequences.",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path("figures/03a_autoregressive_tree.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
