"""Figure: critic accuracy over A2C training.

Train A2C+GAE and track ||V_phi - V*||_inf over episodes. Also visualize
the final V_phi as a heatmap next to the true V*.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.mdp.visualization import plot_value
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_a2c,
)


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    V_star, _, _ = value_iteration(env, gamma=gamma)

    policy = SoftmaxPolicy(env.nS, env.nA)
    baseline = TabularBaseline(env.nS)
    n_episodes = 2000
    chunk = 50
    n_chunks = n_episodes // chunk

    errors = np.zeros(n_chunks)
    rng = np.random.default_rng(0)

    for c in range(n_chunks):
        train_a2c(env, policy, baseline,
                   n_episodes=chunk,
                   lr_actor=0.05, lr_critic=0.2,
                   gamma=gamma, lam=0.95, rng=rng)
        errors[c] = float(np.max(np.abs(baseline.V - V_star)))

    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])

    # Left: ||V_phi - V*|| over training.
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(np.arange(n_chunks) * chunk, errors, color="#c44e52",
                 linewidth=2)
    ax.set_xlabel("episode")
    ax.set_ylabel(r"$\|V_\phi - V^\star\|_\infty$  (log scale)")
    ax.set_title("Critic error vs true V* over training")
    ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)

    # Middle: true V*.
    ax = fig.add_subplot(gs[0, 1])
    plot_value(env, V_star, ax=ax, fontsize=8)
    ax.set_title(r"true $V^\star$ (value iteration)", fontsize=11)

    # Right: learned V_phi.
    ax = fig.add_subplot(gs[0, 2])
    plot_value(env, baseline.V, ax=ax, fontsize=8)
    ax.set_title(rf"$V_\phi$ after {n_episodes} episodes", fontsize=11)

    fig.suptitle(
        f"A2C+GAE critic accuracy on TwoGoalGridWorld. "
        rf"Final $\|V_\phi - V^\star\|_\infty = {errors[-1]:.3f}$.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path("figures/02b_critic_accuracy.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
