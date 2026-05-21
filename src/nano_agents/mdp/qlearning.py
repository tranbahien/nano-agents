"""Tabular Q-learning for finite MDPs.

Watkins (1989); proof of convergence Watkins & Dayan (1992).

Given an MDP with finite |S| and |A| (here we use the GridWorld environment),
Q-learning updates a Q-table from experienced transitions:

    Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

It is *off-policy*: it learns about the greedy policy w.r.t. its current Q
while behaving with an exploratory policy (usually ε-greedy).
"""

from __future__ import annotations

import numpy as np

from .environments import ACTIONS, GridWorld


class TabularQLearning:
    """ε-greedy tabular Q-learning."""

    def __init__(self, nS: int, nA: int, alpha: float = 0.1,
                 gamma: float = 0.95, eps: float = 0.1, rng=None):
        self.nS, self.nA = nS, nA
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.Q = np.zeros((nS, nA))
        self.rng = rng if rng is not None else np.random.default_rng()

    def select(self, s: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.nA))
        return int(np.argmax(self.Q[s]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        """Apply one Q-learning update; return the TD error."""
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return float(td_error)

    def greedy_policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)


def simulate_step(env: GridWorld, s: tuple[int, int], a: int, rng) -> tuple:
    """Sample one transition (s', r, done) from the GridWorld dynamics."""
    outcomes = env.step_outcomes(s, a)
    # outcomes is a list of (s', prob, reward).
    probs = np.array([p for _, p, _ in outcomes])
    idx = rng.choice(len(outcomes), p=probs)
    s_next, _, r = outcomes[idx]
    done = env.is_terminal(s_next)
    return s_next, r, done


def train_q_learning(
    env: GridWorld,
    agent: TabularQLearning,
    n_episodes: int = 5000,
    max_steps: int = 200,
    start_state=None,
    rng=None,
):
    """Train Q-learning by episodic interaction with the GridWorld.

    Returns a history dict with returns per episode and a snapshot of Q at the end.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if start_state is None:
        # Pick a non-terminal, non-wall state to start from each episode.
        non_terminal = [s for s in env.states if not env.is_terminal(s)]
    else:
        non_terminal = [start_state]

    history = {"returns": np.zeros(n_episodes),
                "td_errors": np.zeros(n_episodes)}

    for ep in range(n_episodes):
        s = non_terminal[rng.integers(len(non_terminal))]
        cum_r = 0.0
        td_sum = 0.0
        for _ in range(max_steps):
            si = env.state_to_idx[s]
            a = agent.select(si)
            s_next, r, done = simulate_step(env, s, a, rng)
            si_next = env.state_to_idx[s_next]
            td = agent.update(si, a, r, si_next, done)
            td_sum += abs(td)
            cum_r += r
            s = s_next
            if done:
                break
        history["returns"][ep] = cum_r
        history["td_errors"][ep] = td_sum
    return history
