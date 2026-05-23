"""Figure: reasoning as a search tree.

Left: a single ReAct path (Post 3b) — one trajectory, committed at each step.
Right: tree search (ToT / MCTS) — explore multiple branches, expand the
promising ones, recover from bad steps by trying alternatives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.search import ReasoningTree


def layout_tree(tree, max_depth):
    """Compute (x, y) positions for nodes up to max_depth, breadth-first."""
    positions = {(): (0.5, 1.0)}
    levels = {0: [()]}
    for d in range(1, max_depth + 1):
        parents = levels[d - 1]
        nodes = []
        for p in parents:
            nodes.extend(tree.children(p))
        levels[d] = nodes
        n = len(nodes)
        for i, nd in enumerate(nodes):
            x = (i + 0.5) / n
            y = 1.0 - d / max_depth
            positions[nd] = (x, y)
    return positions, levels


def draw_tree(ax, tree, positions, levels, max_depth, highlight_path=None,
               expanded=None, title=""):
    highlight_path = highlight_path or []
    expanded = expanded if expanded is not None else set(positions.keys())
    # Edges.
    for d in range(1, max_depth + 1):
        for nd in levels[d]:
            parent = nd[:-1]
            if parent in positions and nd in positions:
                x0, y0 = positions[parent]
                x1, y1 = positions[nd]
                on_path = (parent in highlight_path and nd in highlight_path)
                in_exp = (nd in expanded)
                color = "#c44e52" if on_path else ("#888" if in_exp else "#ddd")
                lw = 2.5 if on_path else (1.0 if in_exp else 0.4)
                alpha = 1.0 if (on_path or in_exp) else 0.3
                ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha,
                        zorder=1)
    # Nodes.
    for nd, (x, y) in positions.items():
        if len(nd) > max_depth:
            continue
        is_sol_leaf = tree.is_leaf(nd) and tree.reward(nd) > 0
        on_path = nd in highlight_path
        in_exp = nd in expanded
        if is_sol_leaf:
            color = "#55a467"
        elif on_path:
            color = "#c44e52"
        elif in_exp:
            color = "#dd8452"
        else:
            color = "#e8e8e8"
        size = 90 if (on_path or is_sol_leaf) else (45 if in_exp else 18)
        ax.scatter([x], [y], s=size, color=color, zorder=3,
                   edgecolor="white", linewidth=1.0)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def main() -> None:
    tree = ReasoningTree(depth=4, branching=3, n_correct=4, n_clusters=2,
                          cluster_depth=2, seed=3)
    max_depth = 4
    positions, levels = layout_tree(tree, max_depth)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: single ReAct path. Pick a path that ends at a solution if possible.
    sol_leaf = None
    for nd in levels[max_depth]:
        if tree.reward(nd) > 0:
            sol_leaf = nd
            break
    if sol_leaf is None:
        sol_leaf = levels[max_depth][0]
    react_path = [sol_leaf[:i] for i in range(len(sol_leaf) + 1)]
    # For the ReAct panel, only show the single path's nodes as "expanded".
    react_expanded = set(react_path)
    draw_tree(axes[0], tree, positions, levels, max_depth,
              highlight_path=react_path, expanded=react_expanded,
              title="ReAct: a single committed path\n(one bad step derails the whole chain)")

    # Right: tree search explores many branches.
    # Mark a subset of nodes as "expanded" (e.g., the top-2 branches at each level).
    expanded = set()
    frontier = [()]
    expanded.add(())
    for d in range(max_depth):
        scored = sorted(frontier, key=lambda n: tree.true_value(n), reverse=True)
        keep = scored[:3]  # beam-like expansion
        new_frontier = []
        for nd in keep:
            for c in tree.children(nd):
                expanded.add(c)
                new_frontier.append(c)
        frontier = new_frontier
    draw_tree(axes[1], tree, positions, levels, max_depth,
              highlight_path=[], expanded=expanded,
              title="Tree search (ToT / MCTS): explore branches,\nexpand the promising, recover from dead ends")

    # Manual legend.
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#55a467",
               markersize=10, label="correct leaf"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44e52",
               markersize=10, label="committed path"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#dd8452",
               markersize=10, label="expanded node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e8e8e8",
               markersize=8, label="unexplored"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "From path to tree. A ReAct trace commits to one trajectory; tree "
        "search explores many and recovers from individual bad steps.",
        fontsize=13, y=1.02)
    fig.tight_layout()
    out = Path("figures/03c_path_vs_tree.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
