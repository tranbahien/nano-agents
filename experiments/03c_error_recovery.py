"""Figure: search defeats error compounding.

Post 3b showed that a single ReAct path's success decays as (1-p)^L with
chain length L. The promise of search is recovery: if a step goes wrong,
backtracking or exploration can fix it. We show success vs depth for a
single greedy path versus DFS/MCTS search, holding the per-node heuristic
noise fixed. Search flattens the error-compounding curve.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.search import (
    NoisyValueHeuristic,
    ReasoningTree,
    dfs_tot,
    greedy_search,
    mcts,
)


def main() -> None:
    depths = [2, 3, 4, 5, 6, 7]
    n_trees = 150
    noise = 0.3
    budget = 256

    methods = {
        "greedy (single path)": np.zeros(len(depths)),
        "ToT-DFS (search)": np.zeros(len(depths)),
        "MCTS (search)": np.zeros(len(depths)),
    }

    for di, depth in enumerate(depths):
        for t in range(n_trees):
            tree = ReasoningTree(depth=depth, branching=3, n_correct=4,
                                  n_clusters=2, cluster_depth=min(2, depth - 1),
                                  seed=t)
            h = NoisyValueHeuristic(tree, noise=noise, seed=t)
            methods["greedy (single path)"][di] += int(
                greedy_search(tree, h, budget)[0])
            h.reset()
            methods["ToT-DFS (search)"][di] += int(dfs_tot(tree, h, budget)[0])
            h.reset()
            methods["MCTS (search)"][di] += int(
                mcts(tree, h, budget=budget, c_uct=1.0,
                     rng=np.random.default_rng(t))[0])
    for k in methods:
        methods[k] /= n_trees

    fig, ax = plt.subplots(figsize=(10, 5.8))
    styles = {
        "greedy (single path)": ("o-", "#c44e52"),
        "ToT-DFS (search)": ("^-", "#dd8452"),
        "MCTS (search)": ("D-", "#3a7ebf"),
    }
    for k, (style, color) in styles.items():
        ax.plot(depths, methods[k], style, color=color, linewidth=2,
                markersize=9, label=k)

    ax.set_xlabel("reasoning depth (chain length)")
    ax.set_ylabel("fraction of problems solved")
    ax.set_title(
        f"Search mitigates error compounding (heuristic noise={noise}, "
        f"budget={budget}).\n"
        "DFS keeps far higher success than a single path, but a fixed budget "
        "can't fully outrun exponential growth.",
        fontsize=11)
    ax.legend(loc="lower left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(depths)

    fig.tight_layout()
    out = Path("figures/03c_error_recovery.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    for k in methods:
        print(f"  {k}: depth 2 -> {methods[k][0]:.2f}, depth 7 -> {methods[k][-1]:.2f}")


if __name__ == "__main__":
    main()
