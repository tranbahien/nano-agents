"""Value heuristics for guiding tree search.

A search algorithm needs to estimate how promising a node is without
expanding its whole subtree. We provide two kinds, mirroring practice:

  - NoisyValueHeuristic: the true node value plus Gaussian noise. Models a
    learned value head or an LLM self-evaluation ("rate how promising this
    partial solution is, 0-10") that is informative but imperfect.

  - rollout_value: estimate value by doing random completions (Monte Carlo
    rollouts) and averaging their rewards — the classic MCTS simulation step,
    requiring no learned model.
"""

from __future__ import annotations

import numpy as np

from nano_agents.search.tree import ReasoningTree


class NoisyValueHeuristic:
    """Estimates node value as clip(true_value + N(0, noise), 0, 1)."""

    def __init__(self, tree: ReasoningTree, noise: float = 0.2, seed: int = 0):
        self.tree = tree
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def __call__(self, node: tuple) -> float:
        # Cache so repeated queries to the same node are consistent within a run.
        if node not in self._cache:
            v = self.tree.true_value(node)
            est = v + self.rng.normal(0, self.noise)
            self._cache[node] = float(np.clip(est, 0.0, 1.0))
        return self._cache[node]

    def reset(self):
        self._cache.clear()


def rollout_value(tree: ReasoningTree, node: tuple, n_rollouts: int,
                   rng) -> float:
    """Estimate value by random completions from `node`.

    Returns the fraction of rollouts that reach a correct leaf. Each rollout
    counts as one "expansion" of compute in the budget accounting handled by
    the caller.
    """
    successes = 0
    for _ in range(n_rollouts):
        cur = node
        while not tree.is_leaf(cur):
            cur = cur + (int(rng.integers(tree.branching)),)
        successes += int(tree.reward(cur) > 0)
    return successes / max(n_rollouts, 1)
