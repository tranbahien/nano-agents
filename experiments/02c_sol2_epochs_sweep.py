"""Solution 2c.2: PPO epochs-per-batch sweep.

PPO's multiple-epochs trick is its main efficiency gain over A2C. How
many epochs is too many? We sweep and look at:
  - Sample efficiency (transitions to reach a threshold)
  - Final return
  - Mean clip fraction (more epochs ⇒ more clipping)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_ppo,
)


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    n_iterations = 80
    n_seeds = 5

    epoch_values = [1, 2, 4, 8, 16, 32]

    sample_eff = np.zeros((len(epoch_values), n_seeds))
    final_returns = np.zeros((len(epoch_values), n_seeds))
    clip_fracs = np.zeros((len(epoch_values), n_seeds))
    threshold = 9.5
    n_trajs_per_iter = 16
    avg_traj_len = 12  # approximate

    for i, n_epochs in enumerate(epoch_values):
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            baseline = TabularBaseline(env.nS)
            hist = train_ppo(
                env, policy, baseline,
                n_iterations=n_iterations,
                n_trajectories_per_iter=n_trajs_per_iter,
                n_epochs=n_epochs,
                lr_actor=0.05, lr_critic=0.2,
                gamma=gamma, lam=0.95, eps=0.2,
                rng=np.random.default_rng(seed),
            )
            returns = hist["returns"]
            final_returns[i, seed] = float(np.mean(returns[-20:]))
            above = np.where(returns >= threshold)[0]
            iters_to_threshold = above[0] if len(above) > 0 else n_iterations
            # Sample efficiency = transitions needed.
            sample_eff[i, seed] = iters_to_threshold * n_trajs_per_iter * avg_traj_len
            clip_fracs[i, seed] = float(np.mean(hist["clip_fraction"]))
        print(f"  n_epochs = {n_epochs}: "
              f"final = {final_returns[i].mean():.2f}, "
              f"transitions to threshold = {sample_eff[i].mean():.0f}, "
              f"clip frac = {clip_fracs[i].mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    axes[0].errorbar(epoch_values, sample_eff.mean(axis=1),
                       yerr=sample_eff.std(axis=1) / np.sqrt(n_seeds),
                       fmt="o-", color="#3a7ebf", linewidth=2,
                       markersize=9, capsize=4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n_epochs per batch")
    axes[0].set_ylabel(f"transitions to return ≥ {threshold}")
    axes[0].set_title("Sample efficiency")
    axes[0].grid(which="both", alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].errorbar(epoch_values, final_returns.mean(axis=1),
                       yerr=final_returns.std(axis=1) / np.sqrt(n_seeds),
                       fmt="o-", color="#55a467", linewidth=2,
                       markersize=9, capsize=4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("n_epochs per batch")
    axes[1].set_ylabel(f"final return (last 20 of {n_iterations} iters)")
    axes[1].set_title("Final return")
    axes[1].grid(which="both", alpha=0.3); axes[1].set_axisbelow(True)

    axes[2].errorbar(epoch_values, clip_fracs.mean(axis=1),
                       yerr=clip_fracs.std(axis=1) / np.sqrt(n_seeds),
                       fmt="o-", color="#c44e52", linewidth=2,
                       markersize=9, capsize=4)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("n_epochs per batch")
    axes[2].set_ylabel("mean clip fraction")
    axes[2].set_title("Clipping engagement")
    axes[2].grid(which="both", alpha=0.3); axes[2].set_axisbelow(True)

    fig.suptitle(
        f"PPO epochs-per-batch sweep — more epochs trade compute for sample efficiency, "
        f"until clipping caps the benefit.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02c_sol2_epochs_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
