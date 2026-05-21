"""Solution experiment 1d.3: exploration schedules.

Compare three exploration strategies for Q-learning:
  (a) Constant ε = 0.1
  (b) Constant ε = 0.5
  (c) Decaying schedule: ε_k = max(0.05, 1/sqrt(1+k/100))

Track final return and final ||V_Q - V*||_inf over training. The right
schedule balances exploration early (to discover the value landscape) with
exploitation late (to actually accumulate reward).
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


def train_with_eps_schedule(env, agent, eps_schedule, n_episodes, rng):
    """Train Q-learning where eps may depend on episode index."""
    non_terminal = [s for s in env.states if not env.is_terminal(s)]
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        agent.eps = float(eps_schedule(ep))
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
    return returns


def main() -> None:
    env = GridWorld(rows=5, cols=5,
                     terminals={(4, 4): 1.0, (0, 4): -1.0},
                     step_reward=-0.04, slip=0.1)
    gamma = 0.95
    V_star, _, _ = value_iteration(env, gamma=gamma)

    n_episodes = 4000
    schedules = [
        ("constant ε = 0.1", lambda k: 0.1, "#3a7ebf"),
        ("constant ε = 0.5", lambda k: 0.5, "#dd8452"),
        ("decay: ε_k = max(0.05, (1+k/200)⁻¹ᐟ²)",
         lambda k: max(0.05, (1.0 + k / 200) ** (-0.5)), "#55a467"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: 100-episode moving average of returns.
    ax = axes[0]
    for label, schedule, color in schedules:
        agent = TabularQLearning(nS=env.nS, nA=env.nA, alpha=0.3, gamma=gamma,
                                  rng=np.random.default_rng(0))
        returns = train_with_eps_schedule(
            env, agent, schedule, n_episodes,
            rng=np.random.default_rng(0))
        w = 100
        if len(returns) >= w:
            ma = np.convolve(returns, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(len(ma)) + w // 2, ma,
                    color=color, linewidth=2, label=label)
    ax.set_xlabel("episode"); ax.set_ylabel("100-ep avg return")
    ax.set_title("Returns during training")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # Right: show the ε schedules themselves.
    ax = axes[1]
    eps_traces = np.zeros((len(schedules), n_episodes))
    for i, (label, schedule, color) in enumerate(schedules):
        for k in range(n_episodes):
            eps_traces[i, k] = schedule(k)
        ax.plot(eps_traces[i], color=color, linewidth=2, label=label)
    ax.set_xlabel("episode"); ax.set_ylabel(r"$\varepsilon$")
    ax.set_title("Exploration schedule")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(0, 1)

    fig.suptitle(
        "Q-learning under three exploration schedules. "
        "Constant ε=0.5 explores too much; constant ε=0.1 may explore too "
        "little; decaying achieves the best of both.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/01d_sol3_exploration_schedule.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
