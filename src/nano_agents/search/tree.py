"""A reasoning search tree for studying Tree-of-Thoughts and MCTS.

We model reasoning as search over a tree:
  - The root is the problem.
  - Each edge is one "thought" / reasoning step (a choice among `branching`
    options).
  - A path of length `depth` reaches a leaf — a complete reasoning chain
    ending in an answer.
  - A small set of leaves are "correct" (reward 1); the rest are wrong.

The node value we care about is the probability that a uniformly-random
completion from that node reaches a correct leaf — i.e. the fraction of
correct leaves in the node's subtree. A good search algorithm climbs this
value gradient; a noisy value heuristic makes that hard.

The tree is small enough (branching^depth leaves) to enumerate exactly,
so we can compute true node values and measure any algorithm's success
precisely.
"""

from __future__ import annotations

import numpy as np


class ReasoningTree:
    """Depth-D, branching-K tree with planted (optionally clustered) solutions.

    Nodes are represented as tuples of choices: () is the root, (0,) its
    first child, (0, 2) a grandchild, and length-D tuples are leaves.
    """

    def __init__(self, depth: int = 4, branching: int = 3,
                 n_correct: int = 6, n_clusters: int = 2,
                 cluster_depth: int = 2, seed: int = 0):
        self.depth = depth
        self.branching = branching
        self.n_leaves = branching ** depth
        rng = np.random.default_rng(seed)

        # Plant correct leaves, clustered under a few subtrees to create a
        # learnable value gradient.
        cluster_depth = min(cluster_depth, depth - 1)
        cluster_nodes = set()
        while len(cluster_nodes) < n_clusters:
            node = tuple(int(rng.integers(branching)) for _ in range(cluster_depth))
            cluster_nodes.add(node)
        self.cluster_nodes = list(cluster_nodes)

        correct = set()
        attempts = 0
        while len(correct) < n_correct and attempts < 10000:
            attempts += 1
            cnode = self.cluster_nodes[int(rng.integers(len(self.cluster_nodes)))]
            # Complete the path to a leaf with random choices.
            rest = tuple(int(rng.integers(branching))
                         for _ in range(depth - len(cnode)))
            leaf = cnode + rest
            correct.add(self._leaf_index(leaf))
        self.correct_leaves = correct

    # ---- structure ----

    def is_leaf(self, node: tuple) -> bool:
        return len(node) == self.depth

    def children(self, node: tuple) -> list:
        if self.is_leaf(node):
            return []
        return [node + (k,) for k in range(self.branching)]

    def _leaf_index(self, leaf: tuple) -> int:
        idx = 0
        for c in leaf:
            idx = idx * self.branching + c
        return idx

    def leaf_range(self, node: tuple) -> tuple:
        """[lo, hi) range of leaf indices under this node."""
        remaining = self.depth - len(node)
        span = self.branching ** remaining
        lo = 0
        for c in node:
            lo = lo * self.branching + c
        lo = lo * (self.branching ** remaining)
        return (lo, lo + span)

    # ---- values / rewards ----

    def reward(self, leaf: tuple) -> float:
        assert self.is_leaf(leaf)
        return 1.0 if self._leaf_index(leaf) in self.correct_leaves else 0.0

    def true_value(self, node: tuple) -> float:
        """Fraction of correct leaves in this node's subtree.

        Equals P(a uniformly-random completion reaches a correct leaf).
        """
        lo, hi = self.leaf_range(node)
        n_correct = sum(1 for c in self.correct_leaves if lo <= c < hi)
        return n_correct / (hi - lo)

    def subtree_has_solution(self, node: tuple) -> bool:
        lo, hi = self.leaf_range(node)
        return any(lo <= c < hi for c in self.correct_leaves)
