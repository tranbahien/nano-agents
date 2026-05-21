"""Tests for POMDP and Q-learning code."""

from __future__ import annotations

import numpy as np
import pytest

from nano_agents.mdp import (
    GridWorld,
    TabularQLearning,
    TwoGoalGridWorld,
    train_q_learning,
    value_iteration,
)
from nano_agents.pomdp import (
    TigerPOMDP,
    belief_update,
    discretized_pomdp_value_iteration,
)


def test_tiger_pomdp_shape():
    p = TigerPOMDP()
    assert p.P.shape == (2, 3, 2)
    assert p.Z.shape == (2, 3, 2)
    assert p.R.shape == (2, 3)
    # Transition probabilities sum to 1.
    assert np.allclose(p.P.sum(axis=-1), 1.0)
    # Observation probabilities sum to 1.
    assert np.allclose(p.Z.sum(axis=-1), 1.0)


def test_belief_update_listen():
    p = TigerPOMDP(listen_accuracy=0.85)
    b = np.array([0.5, 0.5])
    # Listen, hear GL → belief should shift toward TL.
    b_next = belief_update(b, a=2, o=0, P=p.P, Z=p.Z)
    assert b_next[0] > 0.5
    assert np.isclose(b_next.sum(), 1.0)
    # Listen, hear GR → belief should shift toward TR.
    b_next = belief_update(b, a=2, o=1, P=p.P, Z=p.Z)
    assert b_next[1] > 0.5


def test_belief_reset_on_open():
    p = TigerPOMDP()
    # Start with a strong belief.
    b = np.array([0.9, 0.1])
    # After opening either door (any observation), belief should be near 0.5.
    b_next = belief_update(b, a=0, o=0, P=p.P, Z=p.Z)
    assert np.allclose(b_next, [0.5, 0.5])


def test_pomdp_value_iteration_converges():
    p = TigerPOMDP()
    V, pi, b_grid, hist = discretized_pomdp_value_iteration(
        p, n_belief=51, gamma=0.95, tol=1e-5)
    assert hist["max_diff"][-1] < 1e-5
    # At belief b(TL) = 0.5 the agent should listen (action 2), not open.
    mid_idx = len(b_grid) // 2
    assert pi[mid_idx] == 2


def test_q_learning_converges_to_vi():
    """On a small deterministic gridworld, Q-learning should learn V close to VI."""
    env = GridWorld(rows=4, cols=4, terminals={(3, 3): 1.0}, step_reward=-0.04,
                     slip=0.0)
    # Reference: VI gives the true V*.
    V_vi, pi_vi, _ = value_iteration(env, gamma=0.95)
    # Q-learning.
    agent = TabularQLearning(nS=env.nS, nA=env.nA, alpha=0.5, gamma=0.95,
                              eps=0.3, rng=np.random.default_rng(0))
    train_q_learning(env, agent, n_episodes=4000, rng=np.random.default_rng(0))
    V_q = agent.Q.max(axis=1)
    # Q-values should be close to V*.
    diff = np.max(np.abs(V_q - V_vi))
    assert diff < 0.2, f"Q-learning V differs from V* by {diff:.3f}"
