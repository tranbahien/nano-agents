"""Figure: REINFORCE learning curves on TwoGoalGridWorld.

Shows returns over episodes for:
  - Vanilla REINFORCE (no baseline)
  - REINFORCE with running-mean baseline
  - REINFORCE with V^pi-optimal baseline (computed by VI)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.policy_gradient import SoftmaxPolicy, train_reinforce


def main() -> None:
    gamma = 0.95
    n_episodes = 3000
    lr = 0.05
    n_seeds = 5

    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    # Pre-compute V* as the optimal baseline.
    V_star, _, _ = value_iteration(env, gamma=gamma)

    configs = [
        ("no baseline", None, "#c44e52"),
        ("running-mean baseline", "mean", "#dd8452"),
        (r"$V^\pi$ baseline (oracle)", V_star, "#3a7ebf"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5))

    for label, baseline, color in configs:
        all_returns = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            hist = train_reinforce(
                env, policy, n_episodes=n_episodes, lr=lr, gamma=gamma,
                baseline=baseline, rng=np.random.default_rng(seed))
            all_returns[seed] = hist["returns"]
        # 100-episode moving average per seed, then mean across seeds.
        w = 50
        ma = np.array([np.convolve(r, np.ones(w) / w, mode="valid")
                        for r in all_returns])
        mean_ma = ma.mean(axis=0)
        std_ma = ma.std(axis=0)
        x = np.arange(len(mean_ma)) + w // 2
        ax.plot(x, mean_ma, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean_ma - std_ma, mean_ma + std_ma,
                         color=color, alpha=0.18)

    ax.set_xlabel("episode")
    ax.set_ylabel(f"return ({w}-episode moving average)")
    ax.set_title(
        f"REINFORCE on TwoGoalGridWorld with three baselines\n"
        f"shaded = ±1 std across {n_seeds} seeds, lr = {lr}, γ = {gamma}",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02a_reinforce_learning_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
