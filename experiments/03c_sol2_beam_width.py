"""Solution 3c.2: Tree-of-Thoughts beam width.

Wider beams explore more branches per level (more noise-robust) but cost
more expansions per level. We sweep beam width at a few noise levels and
fixed budget, finding where breadth stops paying off.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.search import NoisyValueHeuristic, ReasoningTree, beam_search_tot


def main() -> None:
    beam_widths = [1, 2, 3, 5, 8, 12, 20]
    noises = [0.1, 0.3, 0.6]
    n_trees = 200
    budget = 256

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#55a467", "#3a7ebf", "#c44e52"]

    for noise, color in zip(noises, colors):
        success = np.zeros(len(beam_widths))
        for t in range(n_trees):
            tree = ReasoningTree(depth=5, branching=3, n_correct=4,
                                  n_clusters=2, cluster_depth=2, seed=t)
            for bi, bw in enumerate(beam_widths):
                h = NoisyValueHeuristic(tree, noise=noise, seed=t)
                success[bi] += int(
                    beam_search_tot(tree, h, beam_width=bw, budget=budget)[0])
        success /= n_trees
        ax.plot(beam_widths, success, "o-", color=color, linewidth=2,
                markersize=8, label=f"heuristic noise = {noise}")
        print(f"  noise {noise}: " +
              ", ".join(f"b{bw}={success[i]:.2f}" for i, bw in enumerate(beam_widths)))

    ax.set_xlabel("beam width")
    ax.set_ylabel("fraction of trees solved")
    ax.set_title(
        f"ToT beam width vs noise (budget={budget}, {n_trees} trees).\n"
        "Wider beams help more when the heuristic is noisier.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(beam_widths)

    fig.tight_layout()
    out = Path("figures/03c_sol2_beam_width.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
