"""Solution 3c.1: the MCTS exploration constant.

UCT balances exploitation (node value) and exploration (visit-count bonus)
via the constant c_uct:

  UCT(child) = value(child) + c_uct * sqrt(log N(parent) / N(child)).

Too small -> greedy, gets stuck on the first promising branch.
Too large  -> explores too uniformly, wastes budget.
We sweep c_uct and find the sweet spot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.search import NoisyValueHeuristic, ReasoningTree, mcts


def main() -> None:
    c_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    n_trees = 200
    noise = 0.3
    budgets = [32, 128]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#3a7ebf", "#c44e52"]

    for budget, color in zip(budgets, colors):
        success = np.zeros(len(c_values))
        for t in range(n_trees):
            tree = ReasoningTree(depth=5, branching=3, n_correct=4,
                                  n_clusters=2, cluster_depth=2, seed=t)
            for ci, c in enumerate(c_values):
                h = NoisyValueHeuristic(tree, noise=noise, seed=t)
                success[ci] += int(
                    mcts(tree, h, budget=budget, c_uct=c,
                         rng=np.random.default_rng(t))[0])
        success /= n_trees
        ax.plot(c_values, success, "o-", color=color, linewidth=2,
                markersize=9, label=f"budget = {budget}")
        best_ci = int(np.argmax(success))
        ax.scatter([c_values[best_ci]], [success[best_ci]], s=160,
                   color=color, zorder=5, edgecolor="white", linewidth=1.5)
        print(f"  budget {budget}: best c_uct = {c_values[best_ci]} "
              f"(success {success[best_ci]:.2f})")

    ax.set_xlabel("exploration constant $c_{\\rm UCT}$")
    ax.set_ylabel("fraction of trees solved")
    ax.set_title(
        f"MCTS exploration constant (noise={noise}, {n_trees} trees).\n"
        "c=0 is pure exploitation; large c over-explores. Sweet spot in between.",
        fontsize=11)
    ax.legend(loc="best"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out = Path("figures/03c_sol1_cuct_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
