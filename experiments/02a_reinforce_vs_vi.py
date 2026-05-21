"""Figure: REINFORCE's learned policy vs value iteration's optimal policy.

Train REINFORCE on TwoGoalGridWorld for enough episodes, then show both
policies as arrows on a value heatmap. Like the 1d Q-learning vs VI figure,
this shows that REINFORCE recovers (essentially) the same optimal policy
without ever knowing P or R or computing Q-values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.mdp.visualization import plot_policy
from nano_agents.policy_gradient import SoftmaxPolicy, train_reinforce


def main() -> None:
    env = TwoGoalGridWorld(slip=0.1, step_reward=-0.04)
    gamma = 0.95

    V_vi, pi_vi, _ = value_iteration(env, gamma=gamma)

    policy = SoftmaxPolicy(env.nS, env.nA)
    train_reinforce(env, policy, n_episodes=8000, lr=0.05, gamma=gamma,
                     baseline="mean", rng=np.random.default_rng(0))
    pi_pg = policy.greedy()

    # For the value heatmap, use the (greedy) value of the trained policy.
    # Easiest is to evaluate by re-rolling — but more cleanly, just use V_vi as the colormap.
    # We display V_vi twice as the background (a fair reference frame).

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_policy(env, pi_vi, V=V_vi, ax=axes[0])
    axes[0].set_title("Value iteration  (model-based)", fontsize=12)
    plot_policy(env, pi_pg, V=V_vi, ax=axes[1])
    axes[1].set_title("REINFORCE  (sampled trajectories only)", fontsize=12)

    matches = int(np.sum(pi_vi == pi_pg))
    nonterm = sum(1 for s in env.states if not env.is_terminal(s))
    matches_nonterm = sum(1 for s in env.states if not env.is_terminal(s)
                           and pi_vi[env.state_to_idx[s]]
                           == pi_pg[env.state_to_idx[s]])

    fig.suptitle(
        f"REINFORCE recovers the same optimal policy from samples alone. "
        f"Policy match: {matches_nonterm}/{nonterm} non-terminal states.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02a_reinforce_vs_vi.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
