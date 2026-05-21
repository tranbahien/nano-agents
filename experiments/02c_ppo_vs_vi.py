"""Figure: PPO-learned policy vs value iteration on TwoGoalGridWorld.

Continues the comparison sequence:
  1d: Q-learning vs VI
  2a: REINFORCE vs VI
  2b: A2C+GAE vs VI
  2c: PPO vs VI

Demonstrates that PPO converges to the optimal policy and value function
on the same problem.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import GridWorld, value_iteration
from nano_agents.mdp.visualization import plot_policy
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_ppo,
)


def main() -> None:
    env = GridWorld(rows=5, cols=5,
                     terminals={(4, 4): 1.0, (0, 4): -1.0},
                     step_reward=-0.04, slip=0.15)
    gamma = 0.95

    V_vi, pi_vi, _ = value_iteration(env, gamma=gamma)

    policy = SoftmaxPolicy(env.nS, env.nA)
    baseline = TabularBaseline(env.nS)
    train_ppo(env, policy, baseline,
              n_iterations=400,
              n_trajectories_per_iter=16,
              n_epochs=4,
              lr_actor=0.05, lr_critic=0.2,
              gamma=gamma, lam=0.95, eps=0.2,
              rng=np.random.default_rng(0))
    pi_ppo = policy.greedy()
    V_ppo = baseline.V

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_policy(env, pi_vi, V=V_vi, ax=axes[0])
    axes[0].set_title("Value iteration  (model-based)", fontsize=12)
    plot_policy(env, pi_ppo, V=V_ppo, ax=axes[1])
    axes[1].set_title("PPO  (clipped surrogate, 4 epochs/batch)", fontsize=12)

    nonterm = sum(1 for s in env.states if not env.is_terminal(s))
    matches_nonterm = sum(1 for s in env.states if not env.is_terminal(s)
                           and pi_vi[env.state_to_idx[s]]
                           == pi_ppo[env.state_to_idx[s]])
    diff = float(np.max(np.abs(V_vi - V_ppo)))

    fig.suptitle(
        f"PPO recovers the optimal policy from sampled trajectories. "
        f"Policy match: {matches_nonterm}/{nonterm} non-terminal states. "
        rf"$\|V_\phi - V^\star\|_\infty = {diff:.3f}$.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02c_ppo_vs_vi.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
