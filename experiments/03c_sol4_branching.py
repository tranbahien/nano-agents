"""Solution 3c.4: branching factor and the cost of search.

The branching factor K is the number of distinct next-thoughts considered at
each reasoning step — the effective action space. As K grows, the tree
widens exponentially (K^depth leaves) and a fixed budget covers less of it.
We sweep K at fixed depth and budget, comparing the four strategies.

This is directly relevant to LLM reasoning: a "wider" thought-generator
(more candidate next steps per node) gives search more options but makes the
space harder to cover. There is a sweet spot.
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
    branchings = [2, 3, 4, 5, 6]
    n_trees = 200
    noise = 0.3
    budget = 128

    results = {"greedy": np.zeros(len(branchings)),
               "ToT-beam (b=3)": np.zeros(len(branchings)),
               "ToT-DFS": np.zeros(len(branchings)),
               "MCTS": np.zeros(len(branchings))}

    for t in range(n_trees):
        for ki, K in enumerate(branchings):
            tree = ReasoningTree(depth=4, branching=K, n_correct=4,
                                  n_clusters=2, cluster_depth=2, seed=t)
            h = NoisyValueHeuristic(tree, noise=noise, seed=t)
            results["greedy"][ki] += int(greedy_search(tree, h, budget)[0])
            h.reset()
            results["ToT-beam (b=3)"][ki] += int(
                beam_search_tot(tree, h, beam_width=3, budget=budget)[0])
            h.reset()
            results["ToT-DFS"][ki] += int(dfs_tot(tree, h, budget=budget)[0])
            h.reset()
            results["MCTS"][ki] += int(
                mcts(tree, h, budget=budget, c_uct=1.0,
                     rng=np.random.default_rng(t))[0])
    for k in results:
        results[k] /= n_trees

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    styles = {"greedy": ("o-", "#888888"),
              "ToT-beam (b=3)": ("s-", "#3a7ebf"),
              "ToT-DFS": ("^-", "#dd8452"),
              "MCTS": ("D-", "#c44e52")}
    for k, (style, color) in styles.items():
        ax.plot(branchings, results[k], style, color=color, linewidth=2,
                markersize=9, label=k)

    # Annotate leaf count growth.
    for K in branchings:
        ax.annotate(f"{K**4}", (K, -0.02), ha="center", fontsize=8,
                     color="#999", annotation_clip=False)
    ax.text(branchings[-1] + 0.1, -0.02, "leaves", fontsize=8, color="#999",
            va="center")

    ax.set_xlabel("branching factor K (candidate thoughts per step)")
    ax.set_ylabel("fraction of trees solved")
    ax.set_title(
        f"Branching factor vs search success (depth 4, budget={budget}, "
        f"noise={noise}).\n"
        "Wider action spaces give more options but are harder to cover.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(branchings)

    fig.tight_layout()
    out = Path("figures/03c_sol4_branching.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    for k in results:
        print(f"  {k}: K=2 -> {results[k][0]:.2f}, K=6 -> {results[k][-1]:.2f}")


if __name__ == "__main__":
    main()
