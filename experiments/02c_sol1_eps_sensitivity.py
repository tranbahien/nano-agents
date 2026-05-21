"""Solution 2c.1: PPO clip parameter ε sensitivity.

Sweep ε from very small (almost no updates allowed) to very large (no
clipping at all), measuring both convergence speed and final return.
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
    n_iterations = 100
    n_seeds = 5

    eps_values = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 2.0]

    final_returns = np.zeros((len(eps_values), n_seeds))
    converge_iter = np.zeros((len(eps_values), n_seeds))
    threshold = 9.5

    for i, eps in enumerate(eps_values):
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            baseline = TabularBaseline(env.nS)
            hist = train_ppo(
                env, policy, baseline,
                n_iterations=n_iterations,
                n_trajectories_per_iter=16,
                n_epochs=4,
                lr_actor=0.05, lr_critic=0.2,
                gamma=gamma, lam=0.95, eps=eps,
                rng=np.random.default_rng(seed),
            )
            returns = hist["returns"]
            final_returns[i, seed] = float(np.mean(returns[-20:]))
            # Iteration when first reaching the threshold (or n_iterations if never).
            above = np.where(returns >= threshold)[0]
            converge_iter[i, seed] = above[0] if len(above) > 0 else n_iterations
        print(f"  ε = {eps}: final return = {final_returns[i].mean():.2f}, "
              f"converged at iter ≈ {converge_iter[i].mean():.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

    axes[0].errorbar(eps_values, final_returns.mean(axis=1),
                       yerr=final_returns.std(axis=1) / np.sqrt(n_seeds),
                       fmt="o-", color="#3a7ebf", linewidth=2,
                       markersize=8, capsize=4,
                       markeredgecolor="white", markeredgewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"clip parameter $\epsilon$")
    axes[0].set_ylabel(f"final return (last 20 of {n_iterations} iters)")
    axes[0].set_title("Final return vs ε")
    axes[0].grid(which="both", alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].errorbar(eps_values, converge_iter.mean(axis=1),
                       yerr=converge_iter.std(axis=1) / np.sqrt(n_seeds),
                       fmt="o-", color="#c44e52", linewidth=2,
                       markersize=8, capsize=4,
                       markeredgecolor="white", markeredgewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"clip parameter $\epsilon$")
    axes[1].set_ylabel(f"iterations to reach return = {threshold}")
    axes[1].set_title("Convergence speed vs ε")
    axes[1].grid(which="both", alpha=0.3); axes[1].set_axisbelow(True)

    fig.suptitle(
        f"PPO clip-parameter sweep on TwoGoalGridWorld\n"
        f"mean ± SEM over {n_seeds} seeds",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02c_sol1_eps_sensitivity.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
