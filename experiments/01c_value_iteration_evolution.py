"""Figure: Value iteration evolution.

Shows V_k(s) as heatmaps over the gridworld at k = 0, 3, 10, 30, and
converged. Value information propagates outward from the goals as the
Bellman operator is repeatedly applied.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.mdp.visualization import plot_value


def main() -> None:
    # A small slip makes VI converge more slowly and produces a richer
    # propagation sequence than the deterministic case.
    env = TwoGoalGridWorld(slip=0.1, step_reward=-0.04)
    gamma = 0.95
    V_final, pi_final, hist = value_iteration(env, gamma=gamma)

    snapshots = hist["V_snapshots"]
    n_iter = len(snapshots) - 1
    # Pick k = 0, 2, 5, 15, converged — handle the case where n_iter is small.
    candidates = [0, 2, 5, 15, n_iter]
    targets = sorted({min(t, n_iter) for t in candidates})

    # Common scale across panels: use the converged range.
    vmin = float(min(V_final.min(), -0.1))
    vmax = float(V_final.max())

    fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 3.6))
    if len(targets) == 1:
        axes = [axes]
    for ax, k in zip(axes, targets):
        V = snapshots[k]
        plot_value(env, V, ax=ax, vmin=vmin, vmax=vmax, fontsize=8)
        label = f"k = {k}" if k < n_iter else f"k = {n_iter} (converged)"
        ax.set_title(label, fontsize=11)

    fig.suptitle(
        rf"Value iteration on TwoGoalGridWorld ($\gamma = {gamma}$). "
        "Value 'propagates' outward from the +1 and +10 goals as $k$ grows.",
        fontsize=12, y=1.05,
    )
    fig.tight_layout()
    out = Path("figures/01c_value_iteration_evolution.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
