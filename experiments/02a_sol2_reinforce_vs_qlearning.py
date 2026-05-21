"""Solution experiment 2a.2: REINFORCE vs Q-learning sample efficiency.

Both are model-free. Both should converge to the optimal policy. But they
use samples very differently:
  - Q-learning: each (s, a, r, s') transition updates one Q-value entry.
  - REINFORCE: each *full trajectory* updates all theta entries along the path.

We compare them at equal "transitions used" budgets, plotting average
return as a function of transitions seen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import (
    TwoGoalGridWorld,
    TabularQLearning,
    simulate_step,
    train_q_learning,
    value_iteration,
)
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    collect_trajectory,
    compute_returns,
)


def run_qlearning(env, gamma, n_steps_budget, alpha, eps, rng):
    """Train Q-learning, tracking returns per 100-transition chunk."""
    agent = TabularQLearning(env.nS, env.nA, alpha=alpha, gamma=gamma,
                              eps=eps, rng=rng)
    non_terminal = [s for s in env.states if not env.is_terminal(s)]
    transitions_used = 0
    chunks = []
    cum_returns = []
    running_returns = []
    cum_R = 0
    n_ep_in_chunk = 0
    chunk_size = 200  # transitions per measurement

    while transitions_used < n_steps_budget:
        s = non_terminal[rng.integers(len(non_terminal))]
        ep_R = 0.0
        for _ in range(200):
            si = env.state_to_idx[s]
            a = agent.select(si)
            s_next, r, done = simulate_step(env, s, a, rng)
            si_next = env.state_to_idx[s_next]
            agent.update(si, a, r, si_next, done)
            ep_R += r
            transitions_used += 1
            s = s_next
            if done or transitions_used >= n_steps_budget:
                break
        running_returns.append(ep_R)
        n_ep_in_chunk += 1
        if transitions_used // chunk_size > len(chunks):
            chunks.append(transitions_used)
            cum_returns.append(float(np.mean(running_returns[-50:])))
    return np.array(chunks), np.array(cum_returns)


def run_reinforce(env, gamma, n_steps_budget, lr, rng):
    policy = SoftmaxPolicy(env.nS, env.nA)
    non_terminal = [s for s in env.states if not env.is_terminal(s)]
    transitions_used = 0
    chunks = []
    cum_returns = []
    running_returns = []
    chunk_size = 200
    running_mean = 0.0
    n_seen = 0

    while transitions_used < n_steps_budget:
        start = non_terminal[rng.integers(len(non_terminal))]
        traj = collect_trajectory(env, policy, start, rng, max_steps=200)
        rewards = [r for (_, _, r) in traj]
        G = compute_returns(rewards, gamma)
        ep_R = float(np.sum(rewards))
        # Update running mean baseline.
        n_seen += 1
        running_mean += (ep_R - running_mean) / n_seen
        # Apply update.
        for idx, (s, a, _) in enumerate(traj):
            adv = G[idx] - running_mean
            p = policy.probs(s)
            policy.theta[s] += lr * (-adv * p)
            policy.theta[s, a] += lr * adv
        transitions_used += len(traj)
        running_returns.append(ep_R)
        if transitions_used // chunk_size > len(chunks):
            chunks.append(transitions_used)
            cum_returns.append(float(np.mean(running_returns[-50:])))
    return np.array(chunks), np.array(cum_returns)


def main() -> None:
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    gamma = 0.95
    n_steps_budget = 40_000
    n_seeds = 5

    fig, ax = plt.subplots(figsize=(9.5, 5))

    # Q-learning
    all_traces = []
    for seed in range(n_seeds):
        x, y = run_qlearning(env, gamma, n_steps_budget, alpha=0.3,
                              eps=0.15, rng=np.random.default_rng(seed))
        all_traces.append((x, y))
    # Average across seeds via interpolation on a common x grid.
    grid = np.linspace(200, n_steps_budget, 100)
    avgs = np.zeros((len(all_traces), len(grid)))
    for i, (x, y) in enumerate(all_traces):
        avgs[i] = np.interp(grid, x, y)
    mean_q = avgs.mean(axis=0); std_q = avgs.std(axis=0)
    ax.plot(grid, mean_q, color="#3a7ebf", linewidth=2, label="Q-learning")
    ax.fill_between(grid, mean_q - std_q, mean_q + std_q,
                     color="#3a7ebf", alpha=0.2)

    # REINFORCE
    all_traces = []
    for seed in range(n_seeds):
        x, y = run_reinforce(env, gamma, n_steps_budget, lr=0.05,
                               rng=np.random.default_rng(seed))
        all_traces.append((x, y))
    avgs = np.zeros((len(all_traces), len(grid)))
    for i, (x, y) in enumerate(all_traces):
        avgs[i] = np.interp(grid, x, y)
    mean_r = avgs.mean(axis=0); std_r = avgs.std(axis=0)
    ax.plot(grid, mean_r, color="#c44e52", linewidth=2, label="REINFORCE")
    ax.fill_between(grid, mean_r - std_r, mean_r + std_r,
                     color="#c44e52", alpha=0.2)

    # Reference V*(uniform start)
    V_star, _, _ = value_iteration(env, gamma=gamma)
    nonterm = [env.state_to_idx[s] for s in env.states if not env.is_terminal(s)]
    v_uniform = float(np.mean(V_star[nonterm]))
    ax.axhline(v_uniform, color="#222", linestyle="--", linewidth=1,
                alpha=0.6,
                label=rf"$V^\star$ avg over starts = {v_uniform:.2f}")

    ax.set_xlabel("transitions seen")
    ax.set_ylabel("average episode return (last 50 episodes)")
    ax.set_title(
        f"REINFORCE vs Q-learning on TwoGoalGridWorld\n"
        f"shaded = ±1 std over {n_seeds} seeds",
        fontsize=12,
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02a_sol2_reinforce_vs_qlearning.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
