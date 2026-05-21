"""Experiment harness for contextual bandits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def run_contextual(agent, bandit, T: int) -> np.ndarray:
    """Run one episode and return cumulative regret over time."""
    regret = np.zeros(T)
    cum = 0.0
    for t in range(T):
        x = bandit.sample_context()
        k = agent.select(x)
        r = bandit.pull(x, k)
        agent.update(x, k, r)
        cum += bandit.regret_of(x, k)
        regret[t] = cum
    return regret


def run_many_contextual(
    agent_factories: dict[str, Callable],
    bandit_factory: Callable[[int], object],
    T: int,
    n_runs: int = 20,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run multiple contextual episodes per agent and average cumulative regret.

    Each run uses a fresh bandit with a different seed so the parameters
    theta_k differ across runs; the average isolates algorithmic effects.
    """
    np.random.seed(seed)
    results = {name: np.zeros(T) for name in agent_factories}
    for i in range(n_runs):
        bandit = bandit_factory(seed=seed + i)
        for name, ctor in agent_factories.items():
            results[name] += run_contextual(ctor(), bandit, T) / n_runs
    return results
