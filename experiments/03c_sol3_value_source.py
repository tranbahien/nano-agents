"""Solution 3c.3: where does the value heuristic come from?

Two ways to estimate a node's value:
  - A learned value model (here: noisy oracle) — one cheap query per node,
    but its quality is fixed by the model.
  - Monte Carlo rollouts — sample k random completions, average the reward.
    No learned model needed, and accuracy improves with k, but each rollout
    costs compute.

We compare beam search guided by a noisy-oracle heuristic against beam
search guided by rollout-based values, as a function of rollouts-per-node.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.search import (
    NoisyValueHeuristic,
    ReasoningTree,
    beam_search_tot,
    rollout_value,
)


def main() -> None:
    n_trees = 200
    beam_width = 3
    budget = 256
    rollout_counts = [1, 2, 4, 8, 16, 32]

    # Rollout-guided beam search.
    rollout_success = np.zeros(len(rollout_counts))
    for t in range(n_trees):
        tree = ReasoningTree(depth=5, branching=3, n_correct=4,
                              n_clusters=2, cluster_depth=2, seed=t)
        for ri, k in enumerate(rollout_counts):
            rng = np.random.default_rng(t * 50 + ri)
            cache = {}
            def h(node, _k=k, _rng=rng, _cache=cache, _tree=tree):
                if node not in _cache:
                    _cache[node] = rollout_value(_tree, node, _k, _rng)
                return _cache[node]
            rollout_success[ri] += int(
                beam_search_tot(tree, h, beam_width=beam_width, budget=budget)[0])
    rollout_success /= n_trees

    # Noisy-oracle baselines at a few noise levels.
    oracle_noises = [0.1, 0.3, 0.6]
    oracle_success = {}
    for noise in oracle_noises:
        s = 0
        for t in range(n_trees):
            tree = ReasoningTree(depth=5, branching=3, n_correct=4,
                                  n_clusters=2, cluster_depth=2, seed=t)
            hh = NoisyValueHeuristic(tree, noise=noise, seed=t)
            s += int(beam_search_tot(tree, hh, beam_width=beam_width,
                                      budget=budget)[0])
        oracle_success[noise] = s / n_trees

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(rollout_counts, rollout_success, "o-", color="#3a7ebf",
            linewidth=2, markersize=9, label="rollout-based value")
    oracle_colors = ["#55a467", "#dd8452", "#c44e52"]
    for noise, c in zip(oracle_noises, oracle_colors):
        ax.axhline(oracle_success[noise], color=c, linestyle="--",
                    linewidth=2, alpha=0.7,
                    label=f"noisy-oracle value (noise={noise})")

    ax.set_xscale("log", base=2)
    ax.set_xticks(rollout_counts); ax.set_xticklabels([str(k) for k in rollout_counts])
    ax.set_xlabel("rollouts per node")
    ax.set_ylabel("fraction of trees solved")
    ax.set_title(
        f"Value source: Monte Carlo rollouts vs a learned value model.\n"
        "More rollouts -> better value -> better search (beam width "
        f"{beam_width}, {n_trees} trees).",
        fontsize=11)
    ax.legend(loc="lower right", fontsize=9); ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    out = Path("figures/03c_sol3_value_source.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"  rollout success: {rollout_success}")
    print(f"  oracle success: {oracle_success}")


if __name__ == "__main__":
    main()
