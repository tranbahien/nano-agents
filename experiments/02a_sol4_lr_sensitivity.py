"""Solution experiment 2a.4: REINFORCE learning-rate sensitivity.

REINFORCE's behavior depends critically on the learning rate. Too small
and convergence is glacial; too large and the gradient noise causes
catastrophic policy collapse. We sweep lr over two orders of magnitude
and plot the resulting average return after a fixed budget of episodes.

The expected shape is a classic U-curve, with the optimum around lr ≈ 0.05.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld
from nano_agents.policy_gradient import SoftmaxPolicy, train_reinforce


def main() -> None:
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    gamma = 0.95
    n_episodes = 1500
    n_seeds = 5

    lrs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    final_returns = np.zeros((len(lrs), n_seeds))
    for i, lr in enumerate(lrs):
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            hist = train_reinforce(
                env, policy, n_episodes=n_episodes, lr=lr, gamma=gamma,
                baseline="mean", rng=np.random.default_rng(seed))
            # Use the final 100 episodes' mean return as the score.
            final_returns[i, seed] = float(np.mean(hist["returns"][-100:]))
        print(f"lr={lr}: mean final return = {final_returns[i].mean():.2f}")

    means = final_returns.mean(axis=1)
    sems = final_returns.std(axis=1) / np.sqrt(n_seeds)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.errorbar(lrs, means, yerr=sems, fmt="o-", color="#3a7ebf",
                 linewidth=2, markersize=9, capsize=4,
                 markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(
        f"average return over final 100 of {n_episodes} episodes")
    ax.set_title(
        f"REINFORCE sensitivity to learning rate\n"
        f"shaded = ±1 SEM over {n_seeds} seeds, running-mean baseline",
        fontsize=11,
    )
    ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)
    best_idx = int(np.argmax(means))
    ax.annotate(rf"best: lr = {lrs[best_idx]}",
                xy=(lrs[best_idx], means[best_idx]),
                xytext=(lrs[best_idx] * 0.5, means[best_idx] - 2),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#222"))

    fig.tight_layout()
    out = Path("figures/02a_sol4_lr_sensitivity.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
