"""Figure: Optimal value function and policy.

After running value iteration, show V*(s) as a heatmap with the optimal
policy as arrows. Two panels: deterministic (slip=0) and slippery (slip=0.2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.mdp.visualization import plot_policy


def main() -> None:
    gamma = 0.95

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, slip in zip(axes, [0.0, 0.2]):
        env = TwoGoalGridWorld(slip=slip, step_reward=-0.04)
        V, pi, _ = value_iteration(env, gamma=gamma)
        plot_policy(env, pi, V=V, ax=ax)
        ax.set_title(
            f"slip = {slip}    "
            rf"$V^\star$(start) = {V[env.state_to_idx[env.start]]:.2f}",
            fontsize=12,
        )

    fig.suptitle(
        rf"Optimal value and policy on TwoGoalGridWorld ($\gamma = {gamma}$). "
        "Arrows show the optimal action in each non-terminal state.",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/01c_optimal_value_policy.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
