"""Figure: A2C-learned policy vs value iteration's policy.

Same comparison style as 1d (Q-learning vs VI) and 2a (REINFORCE vs VI).
Demonstrates that A2C recovers the optimal policy.
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
    train_a2c,
)


def main() -> None:
    env = GridWorld(rows=5, cols=5,
                     terminals={(4, 4): 1.0, (0, 4): -1.0},
                     step_reward=-0.04, slip=0.15)
    gamma = 0.95

    V_vi, pi_vi, _ = value_iteration(env, gamma=gamma)

    policy = SoftmaxPolicy(env.nS, env.nA)
    baseline = TabularBaseline(env.nS)
    train_a2c(env, policy, baseline,
              n_episodes=6000,
              lr_actor=0.05, lr_critic=0.2,
              gamma=gamma, lam=0.95,
              rng=np.random.default_rng(0))
    pi_ac = policy.greedy()
    V_ac = baseline.V

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_policy(env, pi_vi, V=V_vi, ax=axes[0])
    axes[0].set_title("Value iteration  (model-based)", fontsize=12)
    plot_policy(env, pi_ac, V=V_ac, ax=axes[1])
    axes[1].set_title("A2C+GAE  (learned actor + critic)", fontsize=12)

    matches_nonterm = sum(1 for s in env.states if not env.is_terminal(s)
                           and pi_vi[env.state_to_idx[s]]
                           == pi_ac[env.state_to_idx[s]])
    nonterm = sum(1 for s in env.states if not env.is_terminal(s))
    diff = float(np.max(np.abs(V_vi - V_ac)))

    fig.suptitle(
        f"A2C+GAE recovers the optimal policy from sampled trajectories. "
        f"Policy match: {matches_nonterm}/{nonterm} non-terminal states. "
        rf"$\|V_\phi - V^\star\|_\infty = {diff:.3f}$.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02b_a2c_vs_vi.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
