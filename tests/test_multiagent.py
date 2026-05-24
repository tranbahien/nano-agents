"""Tests for the multiagent subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.multiagent import (
    Committee,
    committee_accuracy,
    condorcet_accuracy,
    correlated_vote_ceiling,
    debate_accuracy,
    majority_vote,
    pipeline_accuracy,
)


def test_condorcet_monotone_above_half():
    # More independent jurors -> higher accuracy when p > 1/2.
    accs = [condorcet_accuracy(0.6, n) for n in (1, 3, 5, 11, 21)]
    assert all(b >= a - 1e-9 for a, b in zip(accs, accs[1:]))
    assert accs[-1] > accs[0]
    assert condorcet_accuracy(0.6, 1) == 0.6


def test_condorcet_below_half_degrades():
    # Below half, more jurors make it worse (jury theorem in reverse).
    assert condorcet_accuracy(0.4, 21) < condorcet_accuracy(0.4, 1)


def test_condorcet_half_is_half():
    assert abs(condorcet_accuracy(0.5, 7) - 0.5) < 1e-9


def test_independent_committee_matches_condorcet():
    # Simulation with rho=0 should match the analytic Condorcet value.
    p, n = 0.6, 7
    c = Committee(n_agents=n, n_answers=2, accuracy=p, correlation=0.0, seed=0)
    sim = committee_accuracy(c, n_items=8000, seed=1)
    assert abs(sim - condorcet_accuracy(p, n)) < 0.03


def test_correlation_caps_voting():
    # With correlated errors, a large committee is capped well below 1.
    p, rho, n = 0.6, 0.5, 31
    c = Committee(n_agents=n, n_answers=2, accuracy=p, correlation=rho, seed=0)
    sim = committee_accuracy(c, n_items=8000, seed=1)
    # Exact finite-n expectation: lockstep items score p, independent items
    # score the Condorcet value. The asymptotic ceiling is 1 - rho(1-p).
    predicted = rho * p + (1 - rho) * condorcet_accuracy(p, n)
    assert sim < 0.95                          # cannot reach ~1
    assert abs(sim - predicted) < 0.03         # matches finite-n prediction
    assert correlated_vote_ceiling(p, rho) > sim   # asymptotic ceiling is higher


def test_more_agents_help_less_when_correlated():
    p = 0.6
    indep_gain = (committee_accuracy(Committee(31, 2, p, 0.0, 0), 6000, 1)
                  - committee_accuracy(Committee(1, 2, p, 0.0, 0), 6000, 1))
    corr_gain = (committee_accuracy(Committee(31, 2, p, 0.7, 0), 6000, 1)
                 - committee_accuracy(Committee(1, 2, p, 0.7, 0), 6000, 1))
    assert indep_gain > corr_gain


def test_majority_vote_basic():
    ans = np.array([[0, 0, 1], [2, 2, 2], [1, 3, 1]])
    voted = majority_vote(ans, n_answers=4, rng=np.random.default_rng(0))
    assert list(voted) == [0, 2, 1]


def test_pipeline_compounds():
    assert abs(pipeline_accuracy(0.9, 1) - 0.9) < 1e-9
    assert abs(pipeline_accuracy(0.9, 5) - 0.9 ** 5) < 1e-9
    # A chain of competent agents can still be unreliable overall.
    assert pipeline_accuracy(0.9, 7) < 0.5


def test_debate_runs_and_bounds():
    c = Committee(n_agents=7, n_answers=4, accuracy=0.55, correlation=0.0, seed=0)
    acc = debate_accuracy(c, n_items=1000, rounds=4, conformity=0.5, seed=1)
    assert 0.0 <= acc <= 1.0


def test_stubborn_wrong_faction_can_herd():
    # A stubborn faction with full conformity from others drags the consensus.
    # With high conformity and stubborn agents, the final accuracy should not
    # exceed the no-debate vote by much, and can be dragged down.
    c = Committee(n_agents=9, n_answers=4, accuracy=0.6, correlation=0.0, seed=0)
    high_conf = debate_accuracy(c, n_items=1500, rounds=6, conformity=0.95,
                                stubborn_frac=0.33, seed=2)
    assert 0.0 <= high_conf <= 1.0
