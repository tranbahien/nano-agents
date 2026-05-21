"""Solution 2b.4: entropy regularization.

Adding β·H[π(·|s)] to the policy gradient objective biases the actor
toward higher-entropy policies. This is meant to maintain exploration
and prevent premature collapse.

We sweep β and visualize:
  - Final return (do high-entropy bonuses hurt convergence?)
  - Average policy entropy over training (do we actually stay more uncertain?)
  - The final greedy-policy quality
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_a2c,
)


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    n_episodes = 2500
    n_seeds = 4

    betas = [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(betas)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for beta, color in zip(betas, colors):
        all_returns = np.zeros((n_seeds, n_episodes))
        all_entropies = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            baseline = TabularBaseline(env.nS)
            hist = train_a2c(
                env, policy, baseline,
                n_episodes=n_episodes,
                lr_actor=0.05, lr_critic=0.2,
                gamma=gamma, lam=0.95,
                ent_coef=beta,
                rng=np.random.default_rng(seed),
            )
            all_returns[seed] = hist["returns"]
            all_entropies[seed] = hist["entropy"]

        w = 100
        # Returns plot.
        ma = np.array([np.convolve(r, np.ones(w) / w, mode="valid")
                        for r in all_returns])
        x = np.arange(ma.shape[1]) + w // 2
        axes[0].plot(x, ma.mean(axis=0), color=color, linewidth=2,
                      label=rf"$\beta = {beta}$")

        # Entropy plot.
        ma_e = np.array([np.convolve(e, np.ones(w) / w, mode="valid")
                          for e in all_entropies])
        axes[1].plot(x, ma_e.mean(axis=0), color=color, linewidth=2,
                      label=rf"$\beta = {beta}$")

    axes[0].set_xlabel("episode")
    axes[0].set_ylabel(f"return ({w}-ep moving average)")
    axes[0].set_title("Returns under entropy regularization")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].set_xlabel("episode")
    axes[1].set_ylabel(f"avg policy entropy along trajectory")
    axes[1].set_title("Policy entropy throughout training")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)
    # Reference: entropy of uniform 4-action policy.
    H_uniform = np.log(env.nA)
    axes[1].axhline(H_uniform, color="#222", linestyle=":",
                     linewidth=1, alpha=0.5,
                     label=f"uniform policy entropy ({H_uniform:.2f})")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Entropy regularization slows policy collapse and trades final return "
        "for exploration.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02b_sol4_entropy_regularization.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
