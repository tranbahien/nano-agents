"""Figure: robustness to value-heuristic noise.

The heuristic is what guides search. As it gets noisier (a weaker value
model / less reliable self-evaluation), how does each algorithm hold up?
We fix a generous budget and sweep the heuristic noise.
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
    noises = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5]
    n_trees = 200
    budget = 128

    results = {"greedy": np.zeros(len(noises)),
               "ToT-beam (b=3)": np.zeros(len(noises)),
               "ToT-DFS": np.zeros(len(noises)),
               "MCTS": np.zeros(len(noises))}

    for t in range(n_trees):
        tree = ReasoningTree(depth=5, branching=3, n_correct=4, n_clusters=2,
                              cluster_depth=2, seed=t)
        for ni, noise in enumerate(noises):
            h = NoisyValueHeuristic(tree, noise=noise, seed=t)
            results["greedy"][ni] += int(greedy_search(tree, h, budget)[0])
            h.reset()
            results["ToT-beam (b=3)"][ni] += int(
                beam_search_tot(tree, h, beam_width=3, budget=budget)[0])
            h.reset()
            results["ToT-DFS"][ni] += int(dfs_tot(tree, h, budget=budget)[0])
            h.reset()
            results["MCTS"][ni] += int(
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
        ax.plot(noises, results[k], style, color=color, linewidth=2,
                markersize=8, label=k)

    ax.set_xlabel("value-heuristic noise (std of error)")
    ax.set_ylabel("fraction of trees solved")
    ax.set_title(
        f"Robustness to heuristic noise (budget={budget}, {n_trees} trees).\n"
        "DFS with backtracking is most noise-robust; greedy collapses first.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out = Path("figures/03c_noise_robustness.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    for k in results:
        print(f"  {k}: noise=0 -> {results[k][0]:.2f}, noise=1.5 -> {results[k][-1]:.2f}")


if __name__ == "__main__":
    main()
