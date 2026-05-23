"""Figure: tree search vs flat sampling at equal compute budget.

Connects to Post 3a's best-of-N / self-consistency. Flat sampling draws N
independent root-to-leaf paths and succeeds if any is correct (best-of-N).
Heuristic-guided sampling biases each step toward higher-value children.
Tree search (MCTS, DFS) reuses structure across the budget.

The question: for the same compute, does structured search beat flat
sampling? It depends on how informative the heuristic is.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.search import (
    NoisyValueHeuristic,
    ReasoningTree,
    dfs_tot,
    mcts,
)


def flat_sampling(tree, n_paths, rng, heuristic=None, guide_strength=0.0):
    """Draw n_paths root-to-leaf paths; success if any reaches a correct leaf.

    If heuristic is provided and guide_strength > 0, bias each step toward
    higher-value children via a softmax over heuristic values.
    """
    for _ in range(n_paths):
        node = ()
        while not tree.is_leaf(node):
            children = tree.children(node)
            if heuristic is not None and guide_strength > 0:
                vals = np.array([heuristic(c) for c in children])
                logits = guide_strength * vals
                logits -= logits.max()
                p = np.exp(logits); p /= p.sum()
                node = children[int(rng.choice(len(children), p=p))]
            else:
                node = children[int(rng.integers(len(children)))]
        if tree.reward(node) > 0:
            return True
    return False


def main() -> None:
    budgets = [1, 2, 4, 8, 16, 32, 64, 128]
    n_trees = 200
    noise = 0.25

    methods = {
        "flat sampling (uniform)": np.zeros(len(budgets)),
        "guided sampling (heuristic)": np.zeros(len(budgets)),
        "MCTS": np.zeros(len(budgets)),
        "ToT-DFS": np.zeros(len(budgets)),
    }

    for t in range(n_trees):
        tree = ReasoningTree(depth=5, branching=3, n_correct=4, n_clusters=2,
                              cluster_depth=2, seed=t)
        for bi, budget in enumerate(budgets):
            rng = np.random.default_rng(t * 100 + bi)
            # Flat sampling: budget = number of paths.
            methods["flat sampling (uniform)"][bi] += int(
                flat_sampling(tree, budget, rng))
            h = NoisyValueHeuristic(tree, noise=noise, seed=t)
            methods["guided sampling (heuristic)"][bi] += int(
                flat_sampling(tree, budget, rng, heuristic=h, guide_strength=8.0))
            h.reset()
            methods["MCTS"][bi] += int(
                mcts(tree, h, budget=budget, c_uct=1.0,
                     rng=np.random.default_rng(t))[0])
            h.reset()
            methods["ToT-DFS"][bi] += int(dfs_tot(tree, h, budget=budget)[0])
    for k in methods:
        methods[k] /= n_trees

    fig, ax = plt.subplots(figsize=(10, 5.8))
    styles = {
        "flat sampling (uniform)": ("o--", "#888888"),
        "guided sampling (heuristic)": ("s--", "#55a467"),
        "MCTS": ("D-", "#c44e52"),
        "ToT-DFS": ("^-", "#dd8452"),
    }
    for k, (style, color) in styles.items():
        ax.plot(budgets, methods[k], style, color=color, linewidth=2,
                markersize=8, label=k)

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("compute budget (paths sampled or nodes expanded)")
    ax.set_ylabel("fraction of trees solved")
    ax.set_title(
        f"Search vs sampling at equal budget (noise={noise}, {n_trees} trees).\n"
        "A good heuristic lets structured search and guided sampling beat "
        "blind sampling.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out = Path("figures/03c_search_vs_sampling.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    for k in methods:
        print(f"  {k}: {methods[k][-1]:.2f} at budget {budgets[-1]}")


if __name__ == "__main__":
    main()
