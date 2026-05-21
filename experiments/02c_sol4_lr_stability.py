"""Solution 2c.4: PPO stability under aggressive learning rates.

A2C (no clipping) breaks at large lr (as we saw in 2a §8.4 for REINFORCE).
PPO with the same large lr stays stable thanks to the clip.

We sweep lr from 0.01 to 2.0 and plot final returns for both algorithms.
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
    train_ppo,
)


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    n_episodes = 1500  # for A2C
    n_iter_ppo = n_episodes // 16
    n_trajs = 16
    n_seeds = 5

    lrs = [0.01, 0.03, 0.05, 0.1, 0.3, 1.0, 2.0]

    final_a2c = np.zeros((len(lrs), n_seeds))
    final_ppo = np.zeros((len(lrs), n_seeds))

    for i, lr in enumerate(lrs):
        for seed in range(n_seeds):
            # A2C
            policy = SoftmaxPolicy(env.nS, env.nA)
            baseline = TabularBaseline(env.nS)
            hist = train_a2c(
                env, policy, baseline,
                n_episodes=n_episodes,
                lr_actor=lr, lr_critic=0.2,
                gamma=gamma, lam=0.95,
                rng=np.random.default_rng(seed))
            final_a2c[i, seed] = float(np.mean(hist["returns"][-100:]))

            # PPO
            policy = SoftmaxPolicy(env.nS, env.nA)
            baseline = TabularBaseline(env.nS)
            hist = train_ppo(
                env, policy, baseline,
                n_iterations=n_iter_ppo,
                n_trajectories_per_iter=n_trajs,
                n_epochs=4,
                lr_actor=lr, lr_critic=0.2,
                gamma=gamma, lam=0.95, eps=0.2,
                rng=np.random.default_rng(seed))
            final_ppo[i, seed] = float(np.mean(hist["returns"][-20:]))
        print(f"  lr = {lr}: A2C = {final_a2c[i].mean():.2f} ± "
              f"{final_a2c[i].std():.2f}, "
              f"PPO = {final_ppo[i].mean():.2f} ± {final_ppo[i].std():.2f}")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.errorbar(lrs, final_a2c.mean(axis=1),
                 yerr=final_a2c.std(axis=1) / np.sqrt(n_seeds),
                 fmt="o-", color="#c44e52", linewidth=2,
                 markersize=9, capsize=4,
                 label="A2C (no clip)")
    ax.errorbar(lrs, final_ppo.mean(axis=1),
                 yerr=final_ppo.std(axis=1) / np.sqrt(n_seeds),
                 fmt="o-", color="#3a7ebf", linewidth=2,
                 markersize=9, capsize=4,
                 label=r"PPO  (ε=0.2, 4 epochs)")
    ax.set_xscale("log")
    ax.set_xlabel("actor learning rate")
    ax.set_ylabel(f"final return")
    ax.set_title(
        "PPO is stable across two orders of magnitude of lr; A2C collapses past lr ≈ 0.3.\n"
        f"mean ± SEM over {n_seeds} seeds",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02c_sol4_lr_stability.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
