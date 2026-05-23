"""Tests for the search subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.search import (
    NoisyValueHeuristic,
    ReasoningTree,
    beam_search_tot,
    dfs_tot,
    greedy_search,
    mcts,
    rollout_value,
)


def test_tree_structure():
    tree = ReasoningTree(depth=3, branching=2, n_correct=2, seed=0)
    assert tree.n_leaves == 8
    assert tree.is_leaf((0, 1, 0))
    assert not tree.is_leaf((0, 1))
    assert tree.children((0,)) == [(0, 0), (0, 1)]


def test_leaf_range():
    tree = ReasoningTree(depth=3, branching=2, n_correct=2, seed=0)
    # Root covers all leaves.
    assert tree.leaf_range(()) == (0, 8)
    # First child covers first half.
    assert tree.leaf_range((0,)) == (0, 4)
    assert tree.leaf_range((1,)) == (4, 8)
    # A leaf covers exactly itself.
    lo, hi = tree.leaf_range((1, 0, 1))
    assert hi - lo == 1


def test_correct_leaves_have_reward():
    tree = ReasoningTree(depth=4, branching=3, n_correct=5, seed=1)
    n_rewarded = 0
    # Enumerate all leaves; count rewarded ones.
    def enumerate_leaves(node):
        if tree.is_leaf(node):
            return [node]
        out = []
        for c in tree.children(node):
            out.extend(enumerate_leaves(c))
        return out
    leaves = enumerate_leaves(())
    n_rewarded = sum(1 for l in leaves if tree.reward(l) > 0)
    assert n_rewarded == len(tree.correct_leaves)
    assert n_rewarded >= 1


def test_true_value_root_is_solution_fraction():
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=2)
    v = tree.true_value(())
    assert abs(v - len(tree.correct_leaves) / tree.n_leaves) < 1e-9


def test_true_value_under_solution_node_positive():
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=2)
    # Some node must contain a solution.
    a_correct_leaf_idx = next(iter(tree.correct_leaves))
    # Reconstruct the leaf path from index.
    path = []
    idx = a_correct_leaf_idx
    for _ in range(tree.depth):
        path.append(idx // (tree.branching ** (tree.depth - len(path) - 1)))
        idx = idx % (tree.branching ** (tree.depth - len(path)))
    # Its parent has positive value.
    parent = tuple(path[:-1])
    assert tree.true_value(parent) > 0


def test_perfect_heuristic_greedy_succeeds():
    """With a noiseless heuristic, greedy should find a solution when one is
    reachable by following max-value children."""
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=0)
    perfect = lambda node: tree.true_value(node)
    found, n_exp, leaf = greedy_search(tree, perfect, budget=100)
    assert found


def test_beam_search_finds_solution():
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=0)
    perfect = lambda node: tree.true_value(node)
    found, n_exp, leaf = beam_search_tot(tree, perfect, beam_width=3, budget=200)
    assert found


def test_dfs_finds_solution():
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=0)
    perfect = lambda node: tree.true_value(node)
    found, n_exp, leaf = dfs_tot(tree, perfect, budget=200)
    assert found


def test_mcts_finds_solution_with_enough_budget():
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=0)
    perfect = lambda node: tree.true_value(node)
    found, n_exp, leaf = mcts(tree, perfect, budget=300, c_uct=1.0,
                               rng=np.random.default_rng(0))
    assert found


def test_rollout_value_in_range():
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=0)
    v = rollout_value(tree, (), n_rollouts=200, rng=np.random.default_rng(0))
    assert 0.0 <= v <= 1.0


def test_noisy_heuristic_cache_consistent():
    tree = ReasoningTree(depth=4, branching=3, n_correct=6, seed=0)
    h = NoisyValueHeuristic(tree, noise=0.3, seed=0)
    v1 = h((0, 1))
    v2 = h((0, 1))
    assert v1 == v2  # cached, consistent within a run
