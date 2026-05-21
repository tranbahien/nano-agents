"""Figure: why naive policy gradient fails with large updates.

We revisit the lr-sensitivity story from 2a §8.4, but now visualize WHY
it fails. Show:
  - Episode returns for vanilla REINFORCE at increasing learning rates.
  - The policy "distance" to the optimal policy over training.
  - The variance of returns across seeds.

The lesson: with large lr, the policy jumps so far in a single update
that the next batch is from a very different distribution than the one
that produced the gradient. The on-policy assumption breaks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.policy_gradient import SoftmaxPolicy, train_reinforce


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    V_star, pi_star, _ = value_iteration(env, gamma=gamma)
    n_episodes = 600
    n_seeds = 8

    lrs = [0.05, 0.5, 2.0]
    colors = ["#3a7ebf", "#dd8452", "#c44e52"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for lr, color in zip(lrs, colors):
        all_returns = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            hist = train_reinforce(
                env, policy, n_episodes=n_episodes, lr=lr, gamma=gamma,
                baseline="mean", rng=np.random.default_rng(seed))
            all_returns[seed] = hist["returns"]

        w = 30
        ma = np.array([np.convolve(r, np.ones(w) / w, mode="valid")
                        for r in all_returns])
        mean_ma = ma.mean(axis=0); std_ma = ma.std(axis=0)
        x = np.arange(len(mean_ma)) + w // 2
        # Left: mean return
        axes[0].plot(x, mean_ma, color=color, linewidth=2,
                      label=rf"$\alpha = {lr}$")
        axes[0].fill_between(x, mean_ma - std_ma, mean_ma + std_ma,
                              color=color, alpha=0.18)
        # Right: variance (std) across seeds
        axes[1].plot(x, std_ma, color=color, linewidth=2,
                      label=rf"$\alpha = {lr}$")

    axes[0].set_xlabel("episode"); axes[0].set_ylabel(f"return ({w}-ep MA)")
    axes[0].set_title("Mean return")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].set_xlabel("episode")
    axes[1].set_ylabel(f"std of return across {n_seeds} seeds")
    axes[1].set_title("Seed-to-seed variance — big lr ⇒ unpredictable outcomes")
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)

    fig.suptitle(
        "Vanilla REINFORCE under increasing learning rates. "
        "α = 0.05 is stable; α = 2.0 is catastrophic AND noisy across seeds.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02c_naive_pg_failure.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
