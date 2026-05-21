"""Solution experiment 1d.4: Q-learning under non-stationarity.

The MDP is stationary for the first half of training, then we swap the
positive and negative terminal positions at episode N/2 (the world has
changed). How does Q-learning respond? We compare two learning rates:
  α = 0.05  (small, slow to adapt)
  α = 0.5   (large, fast to adapt)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import (
    GridWorld,
    TabularQLearning,
    simulate_step,
    value_iteration,
)


def train_with_switch(env_phase1, env_phase2, agent, n_episodes, switch_at, rng):
    """Train with a hard environment switch midway through training."""
    returns = np.zeros(n_episodes)
    errors = np.zeros(n_episodes)
    V_star_p1, _, _ = value_iteration(env_phase1, gamma=agent.gamma)
    V_star_p2, _, _ = value_iteration(env_phase2, gamma=agent.gamma)
    non_terminal_p1 = [s for s in env_phase1.states if not env_phase1.is_terminal(s)]
    non_terminal_p2 = [s for s in env_phase2.states if not env_phase2.is_terminal(s)]

    for ep in range(n_episodes):
        if ep < switch_at:
            env = env_phase1
            V_ref = V_star_p1
            non_terminal = non_terminal_p1
        else:
            env = env_phase2
            V_ref = V_star_p2
            non_terminal = non_terminal_p2

        s = non_terminal[rng.integers(len(non_terminal))]
        cum_r = 0.0
        for _ in range(200):
            si = env.state_to_idx[s]
            a = agent.select(si)
            s_next, r, done = simulate_step(env, s, a, rng)
            si_next = env.state_to_idx[s_next]
            agent.update(si, a, r, si_next, done)
            cum_r += r
            s = s_next
            if done:
                break
        returns[ep] = cum_r
        errors[ep] = np.max(np.abs(agent.Q.max(axis=1) - V_ref))
    return returns, errors


def main() -> None:
    gamma = 0.95
    # Phase 1: +1 at bottom-right, -1 at top-right.
    env_p1 = GridWorld(rows=5, cols=5,
                        terminals={(4, 4): 1.0, (0, 4): -1.0},
                        step_reward=-0.04, slip=0.1)
    # Phase 2: swap them.
    env_p2 = GridWorld(rows=5, cols=5,
                        terminals={(0, 4): 1.0, (4, 4): -1.0},
                        step_reward=-0.04, slip=0.1)

    n_episodes = 6000
    switch_at = n_episodes // 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    alphas = [(0.05, "#3a7ebf"), (0.5, "#c44e52")]
    for alpha, color in alphas:
        agent = TabularQLearning(nS=env_p1.nS, nA=env_p1.nA, alpha=alpha,
                                  gamma=gamma, eps=0.15,
                                  rng=np.random.default_rng(0))
        returns, errors = train_with_switch(
            env_p1, env_p2, agent, n_episodes, switch_at,
            rng=np.random.default_rng(0))
        w = 100
        if len(returns) >= w:
            ma = np.convolve(returns, np.ones(w) / w, mode="valid")
            axes[0].plot(np.arange(len(ma)) + w // 2, ma,
                          color=color, linewidth=2,
                          label=rf"$\alpha = {alpha}$")
        # Plot smoothed error too.
        err_ma = np.convolve(errors, np.ones(w) / w, mode="valid")
        axes[1].plot(np.arange(len(err_ma)) + w // 2, err_ma,
                      color=color, linewidth=2,
                      label=rf"$\alpha = {alpha}$")

    for ax in axes:
        ax.axvline(switch_at, color="#222", linestyle="--", linewidth=1.5,
                    alpha=0.6)
        ax.text(switch_at, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0
                 else ax.get_ylim()[0] * 0.95,
                " ← world flips", fontsize=10, color="#222",
                ha="left", va="top")

    axes[0].set_xlabel("episode"); axes[0].set_ylabel("100-ep avg return")
    axes[0].set_title("Returns through the regime change")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].set_xlabel("episode")
    axes[1].set_ylabel(r"$\|V_Q - V^\star_{\mathrm{current}}\|_\infty$")
    axes[1].set_title("Tracking error against current $V^\\star$")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)

    fig.suptitle(
        "Q-learning under non-stationarity. World flips at episode 3000. "
        "Small α tracks slowly; large α adapts but with higher noise floor.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/01d_sol4_nonstationary.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
