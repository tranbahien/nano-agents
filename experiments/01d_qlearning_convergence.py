"""Figure: Q-learning convergence on a gridworld.

Compares Q-learning's learned value function against the true V* (computed
by value iteration with the model). Shows:
  - Returns per episode (learning curve).
  - ||V_Q - V*||_inf over episodes.
  - The final learned Q-induced value function as a heatmap, side by side
    with the true V*.
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
from nano_agents.mdp.visualization import plot_value


def main() -> None:
    rng = np.random.default_rng(0)
    env = GridWorld(rows=5, cols=5, terminals={(4, 4): 1.0, (0, 4): -1.0},
                     step_reward=-0.04, slip=0.0)
    gamma = 0.95

    V_star, pi_star, _ = value_iteration(env, gamma=gamma)

    n_episodes = 6000
    agent = TabularQLearning(nS=env.nS, nA=env.nA, alpha=0.5, gamma=gamma,
                              eps=0.2, rng=rng)

    # Train in chunks so we can probe ||V_Q - V*|| over time.
    chunk = 100
    chunks = n_episodes // chunk
    errors = np.zeros(chunks)
    avg_returns = np.zeros(chunks)
    all_returns = []
    for c in range(chunks):
        hist = train_q_learning(env, agent, n_episodes=chunk,
                                  rng=np.random.default_rng(c * 7 + 1))
        V_now = agent.Q.max(axis=1)
        errors[c] = np.max(np.abs(V_now - V_star))
        avg_returns[c] = float(np.mean(hist["returns"]))
        all_returns.append(hist["returns"])
    all_returns = np.concatenate(all_returns)
    V_Q = agent.Q.max(axis=1)

    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1],
                          width_ratios=[1.4, 1.4, 1, 1])

    # Top-left: returns per episode.
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(all_returns, color="#888", alpha=0.4, linewidth=0.5)
    # 100-episode moving average.
    w = 100
    if len(all_returns) >= w:
        ma = np.convolve(all_returns, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(ma)) + w // 2, ma, color="#3a7ebf",
                linewidth=2, label="100-episode moving average")
    ax.set_xlabel("episode"); ax.set_ylabel("return")
    ax.set_title("Q-learning returns per episode", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # Bottom-left: error to V*.
    ax = fig.add_subplot(gs[1, 0:2])
    ax.semilogy(np.arange(chunks) * chunk, errors, color="#c44e52",
                linewidth=2)
    ax.set_xlabel("episode"); ax.set_ylabel(r"$\|V_Q - V^\star\|_\infty$")
    ax.set_title("Sup-norm error vs the true optimal value", fontsize=11)
    ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)

    # Right two columns: V* and V_Q side by side.
    ax = fig.add_subplot(gs[:, 2])
    plot_value(env, V_star, ax=ax, fontsize=8)
    ax.set_title(r"true $V^\star$ (value iteration)", fontsize=11)

    ax = fig.add_subplot(gs[:, 3])
    plot_value(env, V_Q, ax=ax, fontsize=8)
    ax.set_title(rf"$V_Q$ after {n_episodes} episodes", fontsize=11)

    fig.suptitle(
        f"Q-learning on a 5×5 gridworld. No model — only sampled transitions. "
        f"α = {agent.alpha}, ε = {agent.eps}, γ = {gamma}.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/01d_qlearning_convergence.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Final ||V_Q - V*||_inf = {errors[-1]:.4f}")


if __name__ == "__main__":
    main()
