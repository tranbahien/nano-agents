"""Figure: PPO learning curves — many epochs per batch.

Compare three configurations on the same gridworld:
  - A2C+GAE (one update per batch)
  - PPO with n_epochs = 1
  - PPO with n_epochs = 4 (the standard choice)
  - PPO with n_epochs = 10 (aggressive)

Plotted against environment interactions (transitions), not iterations,
so we measure sample efficiency. PPO with multiple epochs should reach
the optimum with far fewer transitions than A2C.
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
    n_seeds = 5
    n_trajs_per_iter = 16
    avg_traj_len = 12  # approximate, used to convert iters → transitions
    target_transitions = 30_000

    configs = [
        ("A2C+GAE  (1 update / traj)", "a2c", 1, "#888888"),
        ("PPO,  n_epochs = 1", "ppo", 1, "#dd8452"),
        ("PPO,  n_epochs = 4  (standard)", "ppo", 4, "#3a7ebf"),
        ("PPO,  n_epochs = 10  (aggressive)", "ppo", 10, "#c44e52"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for label, algo, n_epochs, color in configs:
        n_iterations = target_transitions // (n_trajs_per_iter * avg_traj_len)
        all_returns = []
        all_transitions = []

        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            baseline = TabularBaseline(env.nS)
            if algo == "a2c":
                # A2C: one trajectory per "iteration", but match the batched setup.
                # We collect n_trajs_per_iter trajs and apply 1 update per traj.
                # Equivalent: train_a2c with n_episodes = n_iterations * n_trajs_per_iter
                hist = train_a2c(
                    env, policy, baseline,
                    n_episodes=n_iterations * n_trajs_per_iter,
                    lr_actor=0.05, lr_critic=0.2,
                    gamma=gamma, lam=0.95,
                    rng=np.random.default_rng(seed))
                # Group returns into "iterations" of n_trajs_per_iter episodes for fair plotting.
                returns_by_iter = hist["returns"].reshape(-1, n_trajs_per_iter).mean(axis=1)
                trans_by_iter = np.arange(1, len(returns_by_iter) + 1) * (n_trajs_per_iter * avg_traj_len)
            else:
                hist = train_ppo(
                    env, policy, baseline,
                    n_iterations=n_iterations,
                    n_trajectories_per_iter=n_trajs_per_iter,
                    n_epochs=n_epochs,
                    lr_actor=0.05, lr_critic=0.2,
                    gamma=gamma, lam=0.95, eps=0.2,
                    rng=np.random.default_rng(seed))
                returns_by_iter = hist["returns"]
                trans_by_iter = np.arange(1, len(returns_by_iter) + 1) * (n_trajs_per_iter * avg_traj_len)
            all_returns.append(returns_by_iter)
            all_transitions.append(trans_by_iter)

        # Interpolate to common transition grid.
        grid = np.linspace(n_trajs_per_iter * avg_traj_len, target_transitions, 80)
        avgs = np.zeros((n_seeds, len(grid)))
        for i in range(n_seeds):
            avgs[i] = np.interp(grid, all_transitions[i], all_returns[i])
        mean_r = avgs.mean(axis=0); std_r = avgs.std(axis=0)
        ax.plot(grid, mean_r, color=color, linewidth=2, label=label)
        ax.fill_between(grid, mean_r - std_r, mean_r + std_r,
                         color=color, alpha=0.18)

    ax.set_xlabel("transitions seen (approx.)")
    ax.set_ylabel("mean return per iteration")
    ax.set_title(
        f"PPO's multiple-epochs trick — more updates per batch, same data.\n"
        f"shaded = ±1 std over {n_seeds} seeds",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02c_ppo_learning_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
