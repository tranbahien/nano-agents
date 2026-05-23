"""Search algorithms over a ReasoningTree.

All four take a per-node value heuristic and a budget (max node expansions),
and return (found_solution, n_expansions, best_leaf). They differ in how
they spend the budget:

  - greedy_search: descend to the highest-value child each step. Cheap,
    fast, easily misled by heuristic noise. No backtracking.

  - beam_search_tot: Tree-of-Thoughts BFS. Keep the top `beam_width` nodes
    at each depth, ranked by heuristic value. Robust to noise via breadth.

  - dfs_tot: Tree-of-Thoughts DFS with value-ordered children and
    backtracking. Explores depth-first but tries the most promising child
    first; backtracks on dead ends.

  - mcts: Monte Carlo Tree Search with UCT. Balances exploitation (node
    value) and exploration (visit counts) to allocate expansions, then
    commits to the most-visited path.

"Expansion" = generating a node's children (one unit of compute). This is
the currency we budget, mirroring the cost of an LLM call per thought.
"""

from __future__ import annotations

import numpy as np

from nano_agents.search.tree import ReasoningTree


def greedy_search(tree: ReasoningTree, heuristic, budget: int):
    """Descend to the highest-heuristic child until a leaf. No backtracking."""
    node = ()
    n_exp = 0
    while not tree.is_leaf(node):
        children = tree.children(node)
        n_exp += 1
        if n_exp > budget:
            break
        node = max(children, key=heuristic)
    found = tree.is_leaf(node) and tree.reward(node) > 0
    return found, n_exp, node


def beam_search_tot(tree: ReasoningTree, heuristic, beam_width: int,
                      budget: int):
    """Tree-of-Thoughts BFS: keep top-`beam_width` nodes per depth."""
    beam = [()]
    n_exp = 0
    best_leaf = None
    for _ in range(tree.depth):
        candidates = []
        for node in beam:
            n_exp += 1
            if n_exp > budget:
                break
            candidates.extend(tree.children(node))
        if not candidates:
            break
        # Keep the top beam_width candidates by heuristic value.
        candidates.sort(key=heuristic, reverse=True)
        beam = candidates[:beam_width]
        if n_exp > budget:
            break
    # Best leaf among the final beam (by actual reward, since these are leaves).
    leaves = [n for n in beam if tree.is_leaf(n)]
    if leaves:
        best_leaf = max(leaves, key=lambda l: tree.reward(l))
        found = tree.reward(best_leaf) > 0
    else:
        found = False
    return found, n_exp, best_leaf


def dfs_tot(tree: ReasoningTree, heuristic, budget: int,
             prune_threshold: float = 0.0):
    """Tree-of-Thoughts DFS with value-ordered children and backtracking.

    Visits the most promising child first; backtracks on leaves. Optionally
    prunes children whose heuristic value is below prune_threshold.
    """
    n_exp = [0]
    found = [False]
    best_leaf = [None]

    def visit(node):
        if found[0] or n_exp[0] > budget:
            return
        if tree.is_leaf(node):
            if best_leaf[0] is None:
                best_leaf[0] = node
            if tree.reward(node) > 0:
                found[0] = True
                best_leaf[0] = node
            return
        n_exp[0] += 1
        if n_exp[0] > budget:
            return
        children = tree.children(node)
        children = [c for c in children if heuristic(c) >= prune_threshold]
        children.sort(key=heuristic, reverse=True)
        for c in children:
            visit(c)
            if found[0]:
                return

    visit(())
    return found[0], n_exp[0], best_leaf[0]


def mcts(tree: ReasoningTree, heuristic, budget: int, c_uct: float = 1.0,
          rng=None):
    """Monte Carlo Tree Search with UCT and heuristic value estimates.

    Each iteration: select a path by UCT down to an unexpanded node, expand
    it, evaluate with the heuristic, and backpropagate. Budget counts node
    expansions. Returns the most-visited root-to-leaf path's outcome.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    # Statistics per node.
    N: dict = {(): 0}      # visit counts
    W: dict = {(): 0.0}    # total value
    children_of: dict = {}

    def value(node):
        return W[node] / N[node] if N[node] > 0 else 0.0

    def uct_child(node):
        kids = children_of[node]
        logN = np.log(N[node] + 1)
        best, best_score = None, -np.inf
        for c in kids:
            if N.get(c, 0) == 0:
                return c  # prioritize unexplored children
            score = value(c) + c_uct * np.sqrt(logN / N[c])
            if score > best_score:
                best, best_score = c, score
        return best

    n_exp = 0
    found_leaf = None
    iterations = 0
    while iterations < budget:
        iterations += 1
        # --- selection ---
        node = ()
        path = [node]
        while node in children_of and not tree.is_leaf(node):
            node = uct_child(node)
            path.append(node)
        # --- expansion ---
        if not tree.is_leaf(node):
            n_exp += 1
            kids = tree.children(node)
            children_of[node] = kids
            for c in kids:
                N.setdefault(c, 0)
                W.setdefault(c, 0.0)
            # Pick one child to evaluate this iteration.
            node = kids[int(rng.integers(len(kids)))]
            path.append(node)
        # --- evaluation ---
        if tree.is_leaf(node):
            v = tree.reward(node)
            if v > 0:
                found_leaf = node
        else:
            v = heuristic(node)
        # --- backprop ---
        for nd in path:
            N[nd] = N.get(nd, 0) + 1
            W[nd] = W.get(nd, 0.0) + v

    # Commit: follow most-visited children from the root to a leaf.
    node = ()
    while not tree.is_leaf(node) and node in children_of:
        node = max(children_of[node], key=lambda c: N.get(c, 0))
    committed_found = tree.is_leaf(node) and tree.reward(node) > 0
    # Report success if MCTS *found* a solution during search OR commits to one.
    found = committed_found or (found_leaf is not None)
    return found, iterations, (node if committed_found else found_leaf)
