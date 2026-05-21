"""Solution experiment 1c.1: discount factor sensitivity.

Run value iteration on TwoGoalGridWorld with gamma in {0.5, 0.8, 0.95, 0.99}
and visualize the resulting optimal policies. Low gamma → 'myopic', goes
for the closer +1 goal. High gamma → 'patient', goes for the +10 goal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.mdp.visualization import plot_policy


def main() -> None:
    gammas = [0.5, 0.8, 0.95, 0.99]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for ax, gamma in zip(axes, gammas):
        env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
        V, pi, _ = value_iteration(env, gamma=gamma)
        plot_policy(env, pi, V=V, ax=ax)
        v_start = V[env.state_to_idx[env.start]]
        ax.set_title(rf"$\gamma = {gamma}$    $V^\star$(start) = {v_start:.2f}",
                      fontsize=11)

    fig.suptitle(
        "Discount factor controls how patient the agent is. "
        "Low γ → grab the nearby +1; high γ → walk to +10.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out = Path("figures/01c_sol1_discount_sensitivity.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
