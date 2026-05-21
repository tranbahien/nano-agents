"""Solution experiment 1d.1: Tiger problem sensitivity to listen accuracy.

How do the open-left / listen / open-right decision thresholds shift as
the listen action becomes more or less reliable? Solve the Tiger POMDP at
several listen accuracies and plot the optimal policy as a function of
belief, side by side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.pomdp import TigerPOMDP, discretized_pomdp_value_iteration


def main() -> None:
    accuracies = [0.55, 0.70, 0.85, 0.95]
    gamma = 0.95

    fig, ax = plt.subplots(figsize=(10.5, 5))
    action_names = ["Open Left", "Open Right", "Listen"]
    action_colors = {0: "#c44e52", 1: "#dd8452", 2: "#3a7ebf"}

    # We'll plot policy as a strip per accuracy, stacked vertically.
    n_strip = len(accuracies)
    for i, acc in enumerate(accuracies):
        pomdp = TigerPOMDP(listen_accuracy=acc)
        V, pi, b_grid, _ = discretized_pomdp_value_iteration(
            pomdp, n_belief=401, gamma=gamma, tol=1e-8, max_iters=10000)
        N = len(b_grid)
        # Find region boundaries.
        cuts = np.where(np.diff(pi) != 0)[0]
        edges = np.concatenate([[0], cuts + 1, [N]])
        y_lo = i - 0.4
        y_hi = i + 0.4
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            a = int(pi[lo])
            ax.fill_betweenx([y_lo, y_hi],
                              b_grid[lo], b_grid[min(hi, N - 1)],
                              color=action_colors[a], alpha=0.55)

        # Print actual thresholds for the post.
        thr_lo = b_grid[cuts[0] + 1] if len(cuts) > 0 else None
        thr_hi = b_grid[cuts[-1] + 1] if len(cuts) > 0 else None
        print(f"accuracy={acc}: open-right when b(TL) < {thr_lo:.3f}, "
              f"open-left when b(TL) > {thr_hi:.3f}")

    ax.set_yticks(range(n_strip))
    ax.set_yticklabels([f"accuracy = {a}" for a in accuracies])
    ax.set_xlabel(r"$b(\mathrm{TL}) = P(\mathrm{tiger\ is\ left})$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n_strip - 0.4)
    ax.set_title(
        "Tiger POMDP: decision thresholds shift with listen accuracy.\n"
        "Better listening means smaller listen region — the agent commits sooner.",
        fontsize=12,
    )

    # Legend.
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=action_colors[k], alpha=0.55,
                              label=action_names[k]) for k in range(3)]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = Path("figures/01d_sol1_tiger_accuracy.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
