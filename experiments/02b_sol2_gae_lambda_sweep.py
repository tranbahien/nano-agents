"""Solution 2b.2: GAE λ sweep.

Run A2C+GAE over a range of λ values and plot final returns. On a small
deterministic problem the curve is flat; on a stochastic problem the
intermediate λ values are clearly better.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import GridWorld
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    train_a2c,
)


def evaluate(env, gamma, lam, n_episodes, n_seeds):
    finals = np.zeros(n_seeds)
    for seed in range(n_seeds):
        policy = SoftmaxPolicy(env.nS, env.nA)
        baseline = TabularBaseline(env.nS)
        hist = train_a2c(
            env, policy, baseline,
            n_episodes=n_episodes,
            lr_actor=0.05, lr_critic=0.2,
            gamma=gamma, lam=lam,
            rng=np.random.default_rng(seed),
        )
        finals[seed] = float(np.mean(hist["returns"][-200:]))
    return finals


def main() -> None:
    gamma = 0.95
    lams = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    n_episodes = 2000
    n_seeds = 5

    # Use a stochastic environment so the bias-variance tradeoff matters more.
    env = GridWorld(rows=5, cols=5,
                     terminals={(4, 4): 1.0, (0, 4): -1.0},
                     step_reward=-0.04, slip=0.2)

    means, sems = [], []
    for lam in lams:
        print(f"  λ = {lam}...")
        finals = evaluate(env, gamma, lam, n_episodes, n_seeds)
        means.append(float(np.mean(finals)))
        sems.append(float(np.std(finals) / np.sqrt(n_seeds)))

    means = np.array(means); sems = np.array(sems)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.errorbar(lams, means, yerr=sems, fmt="o-",
                 color="#3a7ebf", linewidth=2, markersize=10,
                 capsize=5, markeredgecolor="white", markeredgewidth=1.2)
    best = int(np.argmax(means))
    ax.annotate(rf"best: λ = {lams[best]}",
                xy=(lams[best], means[best]),
                xytext=(lams[best] - 0.25, means[best] - 0.015),
                fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#222"))
    ax.set_xlabel(r"GAE parameter $\lambda$", fontsize=11)
    ax.set_ylabel(f"avg return, final 200 of {n_episodes} episodes",
                   fontsize=11)
    ax.set_title(
        f"A2C+GAE: final return vs $\\lambda$\n"
        f"5×5 slippery gridworld (slip=0.2), mean ± SEM across {n_seeds} seeds",
        fontsize=12,
    )
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(lams)
    ax.set_xticklabels([f"{l}" for l in lams], rotation=0)
    # Stretch y-axis to highlight the curve.
    pad = (means.max() - means.min()) * 0.5
    ax.set_ylim(means.min() - pad, means.max() + pad)

    fig.tight_layout()
    out = Path("figures/02b_sol2_gae_lambda_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
