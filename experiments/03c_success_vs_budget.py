"""Figure: success rate vs compute budget for four search strategies.

Greedy, ToT-beam, ToT-DFS, and MCTS, each given a noisy value heuristic and
a budget of node expansions. We average over many random trees and report
the fraction that find a correct leaf within budget.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.search import (
    NoisyValueHeuristic,
    ReasoningTree,
    beam_search_tot,
    dfs_tot,
    greedy_search,
    mcts,
)


def main() -> None:
    budgets = [1, 2, 4, 8, 16, 32, 64, 128]
    n_trees = 200
    noise = 0.25

    results = {"greedy": np.zeros(len(budgets)),
               "ToT-beam (b=3)": np.zeros(len(budgets)),
               "ToT-DFS": np.zeros(len(budgets)),
               "MCTS": np.zeros(len(budgets))}

    for t in range(n_trees):
        tree = ReasoningTree(depth=5, branching=3, n_correct=4, n_clusters=2,
                              cluster_depth=2, seed=t)
        for bi, budget in enumerate(budgets):
            h = NoisyValueHeuristic(tree, noise=noise, seed=t)
            results["greedy"][bi] += int(greedy_search(tree, h, budget)[0])
            h.reset()
            results["ToT-beam (b=3)"][bi] += int(
                beam_search_tot(tree, h, beam_width=3, budget=budget)[0])
            h.reset()
            results["ToT-DFS"][bi] += int(dfs_tot(tree, h, budget=budget)[0])
            h.reset()
            results["MCTS"][bi] += int(
                mcts(tree, h, budget=budget, c_uct=1.0,
                     rng=np.random.default_rng(t))[0])

    for k in results:
        results[k] /= n_trees

    fig, ax = plt.subplots(figsize=(10, 5.8))
    styles = {"greedy": ("o-", "#888888"),
              "ToT-beam (b=3)": ("s-", "#3a7ebf"),
              "ToT-DFS": ("^-", "#dd8452"),
              "MCTS": ("D-", "#c44e52")}
    for k, (style, color) in styles.items():
        ax.plot(budgets, results[k], style, color=color, linewidth=2,
                markersize=8, label=k)

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("compute budget (node expansions)")
    ax.set_ylabel("fraction of trees solved")
    ax.set_title(
        f"Search success vs budget (noise={noise}, {n_trees} random trees, "
        f"depth 5, branching 3)",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out = Path("figures/03c_success_vs_budget.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    for k in results:
        print(f"  {k}: {results[k][-1]:.2f} at budget {budgets[-1]}")


if __name__ == "__main__":
    main()
