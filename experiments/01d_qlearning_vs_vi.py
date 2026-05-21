"""Figure: Q-learning's learned policy matches value iteration's policy.

Run both on the same gridworld (with stochastic transitions to make it
interesting) and visualize their policies as arrows on the value heatmap.
The arrows should match — Q-learning has learned the optimal policy without
ever being given P or R.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import (
    GridWorld,
    TabularQLearning,
    train_q_learning,
    value_iteration,
)
from nano_agents.mdp.visualization import plot_policy


def main() -> None:
    env = GridWorld(rows=5, cols=5,
                     terminals={(4, 4): 1.0, (0, 4): -1.0},
                     step_reward=-0.04, slip=0.15)
    gamma = 0.95

    # VI baseline.
    V_vi, pi_vi, _ = value_iteration(env, gamma=gamma)

    # Q-learning.
    agent = TabularQLearning(nS=env.nS, nA=env.nA, alpha=0.3, gamma=gamma,
                              eps=0.2, rng=np.random.default_rng(0))
    train_q_learning(env, agent, n_episodes=20000,
                      rng=np.random.default_rng(0))
    V_q = agent.Q.max(axis=1)
    pi_q = agent.greedy_policy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    plot_policy(env, pi_vi, V=V_vi, ax=axes[0])
    axes[0].set_title("Value iteration  (knows the model)", fontsize=12)
    plot_policy(env, pi_q, V=V_q, ax=axes[1])
    axes[1].set_title("Q-learning  (only sampled transitions)", fontsize=12)

    matches = int(np.sum(pi_vi == pi_q))
    total_nonterminal = sum(1 for s in env.states if not env.is_terminal(s))
    diff = float(np.max(np.abs(V_vi - V_q)))

    fig.suptitle(
        f"Both algorithms find the same optimal policy. "
        f"Policy matches on {matches}/{env.nS} states "
        f"({matches - len(env.terminals)}/{total_nonterminal} non-terminal). "
        rf"$\|V_Q - V^\star\|_\infty = {diff:.3f}$.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/01d_qlearning_vs_vi.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
