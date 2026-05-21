"""REINFORCE (Williams 1992) with optional baseline.

Trajectory:
  Sample (s_0, a_0, r_0, s_1, a_1, ...) under the current policy.
Returns:
  G_t = sum_{k=0..} gamma^k r_{t+k+1}
Estimator:
  grad J(theta) ≈ (1/N) * sum_n sum_t (G_t^n - b(s_t)) * grad_theta log pi(a_t|s_t)

The baseline b(s) is subtracted to reduce variance. Choices include
constant (e.g. mean return), state-dependent (V^pi(s) is optimal in
the variance-reduction sense), or learned (next post: actor-critic).
"""

from __future__ import annotations

import numpy as np

from nano_agents.mdp import GridWorld, simulate_step


def collect_trajectory(env: GridWorld, policy, start_state, rng,
                        max_steps: int = 200):
    """Roll out a single episode under the given policy.

    Returns a list of (state_idx, action, reward) tuples. The episode
    terminates when env says it's terminal or max_steps is hit.
    """
    s = start_state
    traj = []
    for _ in range(max_steps):
        si = env.state_to_idx[s]
        a = policy.sample(si, rng)
        s_next, r, done = simulate_step(env, s, a, rng)
        traj.append((si, a, r))
        s = s_next
        if done:
            break
    return traj


def compute_returns(rewards, gamma: float) -> np.ndarray:
    """Discounted returns from each timestep onward: G_t = sum_{k=0..} gamma^k r_{t+k+1}."""
    n = len(rewards)
    G = np.zeros(n)
    running = 0.0
    for t in reversed(range(n)):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def reinforce_step(policy, traj, returns: np.ndarray, lr: float,
                    baseline: np.ndarray | None = None):
    """Apply one REINFORCE gradient update from a single trajectory.

    Returns the L2 norm of the gradient applied (for diagnostics).
    """
    if baseline is None:
        advantages = returns
    else:
        # baseline[s] is the value baseline at each state.
        states = np.array([s for (s, _, _) in traj])
        advantages = returns - baseline[states]

    grad = np.zeros_like(policy.theta)
    for (s, a, _), A in zip(traj, advantages):
        # SoftmaxPolicy.grad_log_prob is sparse — only row s is nonzero.
        # Accumulate directly into that row for efficiency.
        p = policy.probs(s)
        # gradient is A * (e_a - p)
        grad[s] += A * (-p)
        grad[s, a] += A
    policy.theta += lr * grad
    return float(np.linalg.norm(grad))


def train_reinforce(env: GridWorld, policy, n_episodes: int, lr: float,
                     gamma: float = 0.95, baseline=None, rng=None,
                     start_state=None, max_steps: int = 200):
    """Train REINFORCE on the given environment.

    baseline can be:
      None       — no baseline (vanilla REINFORCE).
      'mean'     — running mean of episode returns (constant baseline).
      np.ndarray — fixed state-dependent baseline of length env.nS.
      callable   — a function that takes the current policy and returns
                   a baseline array.

    Returns a history dict with returns, gradient norms, and (when applicable)
    the per-episode baseline values used.
    """
    rng = rng if rng is not None else np.random.default_rng()
    non_terminal = [s for s in env.states if not env.is_terminal(s)]
    returns_history = np.zeros(n_episodes)
    grad_norms = np.zeros(n_episodes)

    running_mean = 0.0
    n_seen = 0

    for ep in range(n_episodes):
        start = start_state if start_state is not None else non_terminal[
            rng.integers(len(non_terminal))]
        traj = collect_trajectory(env, policy, start, rng, max_steps=max_steps)
        rewards = [r for (_, _, r) in traj]
        Gs = compute_returns(rewards, gamma)

        # Resolve baseline for this episode.
        if baseline is None:
            b = None
        elif isinstance(baseline, str) and baseline == "mean":
            # State-dependent constant baseline = running mean.
            b = np.full(env.nS, running_mean)
        elif callable(baseline):
            b = baseline(policy)
        else:
            b = np.asarray(baseline)

        grad_norms[ep] = reinforce_step(policy, traj, Gs, lr, baseline=b)
        ep_return = float(np.sum(rewards))
        returns_history[ep] = ep_return

        # Update running mean.
        n_seen += 1
        running_mean += (ep_return - running_mean) / n_seen

    return {"returns": returns_history, "grad_norms": grad_norms}
