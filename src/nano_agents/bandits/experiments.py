"""Experiment harness for bandits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def run(agent, bandit, T: int) -> np.ndarray:
    """Run one episode and return cumulative regret over time.

    Returns
    -------
    np.ndarray of shape (T,)
        Cumulative regret at each step.
    """
    regret = np.zeros(T)
    cum = 0.0
    for t in range(T):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        cum += bandit.regret_of(k)
        regret[t] = cum
    return regret


def run_many(
    agent_factories: dict[str, Callable],
    bandit_factory: Callable,
    T: int,
    n_runs: int = 100,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run multiple episodes per agent and return mean cumulative regret curves.

    Parameters
    ----------
    agent_factories : dict
        Mapping from agent name to a zero-arg constructor.
    bandit_factory : callable
        Zero-arg constructor for a fresh bandit per run.
    T : int
        Horizon per episode.
    n_runs : int
        Number of independent episodes to average over.
    seed : int
        RNG seed.
    """
    np.random.seed(seed)
    results = {name: np.zeros(T) for name in agent_factories}
    for _ in range(n_runs):
        bandit = bandit_factory()
        for name, ctor in agent_factories.items():
            results[name] += run(ctor(), bandit, T) / n_runs
    return results
