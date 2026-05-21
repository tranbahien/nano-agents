"""Tests for the contextual bandits module."""

from __future__ import annotations

import numpy as np
import pytest

from nano_agents.contextual import (
    ContextFreeUCB,
    ContextualEpsilonGreedy,
    LinearContextualBandit,
    LinearThompsonSampling,
    LinUCB,
)


def test_environment_shapes():
    bandit = LinearContextualBandit(K=4, d=5, seed=0)
    x = bandit.sample_context()
    assert x.shape == (5,)
    assert np.isclose(np.linalg.norm(x), 1.0)
    assert bandit.thetas.shape == (4, 5)
    r = bandit.pull(x, 0)
    assert isinstance(r, float)
    assert bandit.best_action(x) in range(4)
    assert bandit.regret_of(x, bandit.best_action(x)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "make_agent",
    [
        lambda: LinUCB(K=3, d=4, alpha=1.0),
        lambda: LinearThompsonSampling(K=3, d=4, v=1.0, rng=np.random.default_rng(0)),
        lambda: ContextualEpsilonGreedy(K=3, d=4, eps=0.1, rng=np.random.default_rng(0)),
        lambda: ContextFreeUCB(K=3, d=4),
    ],
)
def test_agent_runs(make_agent):
    """Each agent should run a short episode without errors and produce
    bounded cumulative regret (since per-step regret is bounded)."""
    bandit = LinearContextualBandit(K=3, d=4, sigma=0.05, seed=0)
    agent = make_agent()
    cum_regret = 0.0
    for _ in range(200):
        x = bandit.sample_context()
        k = agent.select(x)
        r = bandit.pull(x, k)
        agent.update(x, k, r)
        cum_regret += bandit.regret_of(x, k)
    assert cum_regret >= 0.0
    # 200 steps with reward gaps at most ~2 (since |x| = 1, |theta| = 1).
    assert cum_regret < 200 * 2.0


def test_linucb_beats_random():
    """LinUCB should beat uniform random sampling on a simple problem."""
    rng = np.random.default_rng(0)

    def run_random():
        bandit = LinearContextualBandit(K=4, d=5, sigma=0.05, seed=1)
        cum = 0.0
        for _ in range(500):
            x = bandit.sample_context()
            k = int(rng.integers(4))
            bandit.pull(x, k)
            cum += bandit.regret_of(x, k)
        return cum

    def run_linucb():
        bandit = LinearContextualBandit(K=4, d=5, sigma=0.05, seed=1)
        agent = LinUCB(K=4, d=5, alpha=1.0)
        cum = 0.0
        for _ in range(500):
            x = bandit.sample_context()
            k = agent.select(x)
            r = bandit.pull(x, k)
            agent.update(x, k, r)
            cum += bandit.regret_of(x, k)
        return cum

    assert run_linucb() < 0.5 * run_random()


def test_nonlinear_bandit():
    from nano_agents.contextual import NonlinearContextualBandit
    bandit = NonlinearContextualBandit(K=3, d=4, sigma=0.0, seed=0)
    x = bandit.sample_context()
    assert x.shape == (4,)
    # Without noise the regret of the best action is 0.
    best = bandit.best_action(x)
    assert bandit.regret_of(x, best) == pytest.approx(0.0)
    # Different actions usually have nonzero regret.
    assert bandit.regret_of(x, (best + 1) % 3) > 0
