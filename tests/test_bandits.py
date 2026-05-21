"""Tests for the bandits module."""

from __future__ import annotations

import numpy as np
import pytest

from nano_agents.bandits import (
    BernoulliBandit,
    EpsilonGreedy,
    GaussianBandit,
    ThompsonSampling,
    UCB1,
)


def test_bernoulli_bandit_deterministic_arms():
    b = BernoulliBandit([0.0, 1.0])
    np.random.seed(0)
    assert b.pull(0) == 0
    assert b.pull(1) == 1
    assert b.best == 1.0
    assert b.regret_of(0) == 1.0
    assert b.regret_of(1) == 0.0


def test_gaussian_bandit_shape():
    b = GaussianBandit([0.0, 1.0, 2.0], sigma=0.1)
    assert b.K == 3
    assert b.best == 2.0


@pytest.mark.parametrize(
    "make_agent",
    [
        lambda: EpsilonGreedy(3, eps=0.1),
        lambda: UCB1(3),
        lambda: ThompsonSampling(3),
    ],
)
def test_agent_converges_to_best_arm(make_agent):
    """Each agent should mostly play the best arm after enough pulls."""
    np.random.seed(0)
    bandit = BernoulliBandit([0.1, 0.5, 0.9])
    agent = make_agent()
    pulls: list[int] = []
    for _ in range(2000):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        pulls.append(k)
    # The last 500 pulls should overwhelmingly favor arm 2.
    last = pulls[-500:]
    assert sum(p == 2 for p in last) > 350, f"Best-arm pulls: {sum(p==2 for p in last)}/500"


def test_gaussian_ucb_runs():
    from nano_agents.bandits import GaussianBandit, GaussianUCB
    np.random.seed(0)
    bandit = GaussianBandit([0.0, 0.5, 1.0], sigma=0.5)
    agent = GaussianUCB(3, sigma=0.5)
    pulls = []
    for _ in range(1000):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        pulls.append(k)
    # Best arm is 2.
    assert sum(p == 2 for p in pulls[-200:]) > 120


def test_gaussian_thompson_sampling_runs():
    from nano_agents.bandits import GaussianBandit, GaussianThompsonSampling
    np.random.seed(0)
    bandit = GaussianBandit([0.0, 0.5, 1.0], sigma=0.5)
    agent = GaussianThompsonSampling(3, rng=np.random.default_rng(0))
    pulls = []
    for _ in range(1000):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        pulls.append(k)
    assert sum(p == 2 for p in pulls[-200:]) > 120


def test_drifting_bandit():
    from nano_agents.bandits import DriftingBernoulliBandit
    np.random.seed(0)
    bandit = DriftingBernoulliBandit([0.3, 0.5, 0.7], amplitude=0.2,
                                      period=500, seed=0)
    # Initial means should be near base + sin-perturbation.
    m0 = bandit.probs.copy()
    bandit.pull(0)
    bandit.pull(0)
    m_later = bandit.probs
    # Probs should have changed after a few pulls.
    assert not np.allclose(m0, m_later)
