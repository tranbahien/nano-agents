"""Solution experiment 1d.2: Q-learning sample efficiency.

How fast does Q-learning converge? Try several learning rates alpha and
plot ||V_Q - V*||_inf over episodes. The Robbins-Monro conditions require
sum(alpha) = infinity and sum(alpha^2) < infinity for convergence guarantees,
but with a constant alpha we can still converge in practice — just with
a noise floor proportional to alpha.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import (
    GridWorld,
    TabularQLearning,
    train_q_learning,
    value_iteration,
)


def main() -> None:
    env = GridWorld(rows=5, cols=5,
                     terminals={(4, 4): 1.0, (0, 4): -1.0},
                     step_reward=-0.04, slip=0.1)
    gamma = 0.95
    V_star, _, _ = value_iteration(env, gamma=gamma)

    alphas = [0.05, 0.2, 0.5, 0.9]
    colors = ["#888888", "#dd8452", "#3a7ebf", "#c44e52"]
    n_episodes = 5000
    chunk = 50

    fig, ax = plt.subplots(figsize=(9, 5))

    for alpha, color in zip(alphas, colors):
        agent = TabularQLearning(nS=env.nS, nA=env.nA, alpha=alpha,
                                  gamma=gamma, eps=0.2,
                                  rng=np.random.default_rng(0))
        chunks = n_episodes // chunk
        errors = np.zeros(chunks)
        for c in range(chunks):
            train_q_learning(env, agent, n_episodes=chunk,
                              rng=np.random.default_rng(c * 7 + 1))
            V_now = agent.Q.max(axis=1)
            errors[c] = np.max(np.abs(V_now - V_star))
        ax.semilogy(np.arange(chunks) * chunk, errors, color=color,
                     linewidth=2, label=rf"$\alpha = {alpha}$")

    ax.set_xlabel("episode")
    ax.set_ylabel(r"$\|V_Q - V^\star\|_\infty$")
    ax.set_title(
        "Q-learning convergence under different (constant) learning rates.\n"
        "Small α → low noise floor but slow. Large α → fast but noisy.",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    ax.grid(which="both", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/01d_sol2_qlearning_alpha.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
