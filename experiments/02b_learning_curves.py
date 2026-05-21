"""Figure: REINFORCE vs A2C vs A2C+GAE.

Compares three policy-gradient variants on the same gridworld:
  - Vanilla REINFORCE (Monte Carlo, no learned baseline).
  - A2C with λ = 1.0 (MC returns minus learned V_φ).
  - A2C with λ = 0.95 (GAE — standard modern choice).
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
    train_reinforce,
)


def main() -> None:
    gamma = 0.95
    n_episodes = 2500
    n_seeds = 5
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)

    configs = [
        ("REINFORCE (running-mean baseline)", "reinforce", None, "#c44e52"),
        ("A2C (learned V, λ = 1.0)", "a2c", 1.0, "#dd8452"),
        ("A2C + GAE (λ = 0.95)", "a2c", 0.95, "#3a7ebf"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for label, algo, lam, color in configs:
        all_returns = np.zeros((n_seeds, n_episodes))
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(env.nS, env.nA)
            rng = np.random.default_rng(seed)
            if algo == "reinforce":
                hist = train_reinforce(
                    env, policy, n_episodes=n_episodes, lr=0.05, gamma=gamma,
                    baseline="mean", rng=rng)
            else:
                baseline = TabularBaseline(env.nS)
                hist = train_a2c(
                    env, policy, baseline,
                    n_episodes=n_episodes,
                    lr_actor=0.05, lr_critic=0.2,
                    gamma=gamma, lam=lam, rng=rng)
            all_returns[seed] = hist["returns"]

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
        f"REINFORCE vs A2C vs A2C+GAE on TwoGoalGridWorld\n"
        f"shaded = ±1 std across {n_seeds} seeds",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02b_learning_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
