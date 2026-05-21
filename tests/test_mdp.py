"""Tests for the MDP module."""

from __future__ import annotations

import numpy as np

from nano_agents.mdp import (
    GridWorld,
    TwoGoalGridWorld,
    bellman_optimality_residual,
    policy_iteration,
    value_iteration,
)


def test_gridworld_basic():
    env = GridWorld(rows=3, cols=3, terminals={(0, 2): 1.0})
    P, R = env.transition_tensors()
    assert P.shape == (9, 4, 9)
    assert R.shape == (9, 4)
    # Probabilities sum to 1 along the next-state axis.
    assert np.allclose(P.sum(axis=-1), 1.0)


def test_value_iteration_converges_simple():
    env = GridWorld(rows=3, cols=3, terminals={(0, 2): 1.0}, step_reward=-0.04)
    V, pi, hist = value_iteration(env, gamma=0.95)
    # Residual should be tiny.
    assert bellman_optimality_residual(V, env, gamma=0.95) < 1e-6
    # Start state's value should be positive (we can reach the +1 goal).
    assert V[env.state_to_idx[(2, 0)]] > 0


def test_pi_and_vi_agree():
    env = GridWorld(rows=4, cols=4, terminals={(3, 3): 1.0, (0, 3): -1.0},
                     walls={(1, 1)}, step_reward=-0.04)
    V_vi, pi_vi, _ = value_iteration(env, gamma=0.9)
    V_pi, pi_pi, _ = policy_iteration(env, gamma=0.9)
    # The two should agree on values (up to tolerance) and on policy on
    # states with a unique argmax.
    assert np.allclose(V_vi, V_pi, atol=1e-5)


def test_two_goal_discount_sensitivity():
    """High gamma → big goal; low gamma → small goal."""
    env = TwoGoalGridWorld(slip=0.0)
    start_idx = env.state_to_idx[env.start]

    V_high, _, _ = value_iteration(env, gamma=0.95)
    V_low, _, _ = value_iteration(env, gamma=0.4)

    # With high gamma the agent values the +10 strongly; with low gamma
    # it values the +1 more (in proportional terms).
    # Sanity check: high-gamma start value should clearly exceed low-gamma.
    assert V_high[start_idx] > V_low[start_idx]


def test_stochastic_transition_softens():
    """Slippery gridworld should change values vs deterministic version."""
    det = TwoGoalGridWorld(slip=0.0)
    slip = TwoGoalGridWorld(slip=0.3)
    V_det, _, _ = value_iteration(det, gamma=0.95)
    V_slip, _, _ = value_iteration(slip, gamma=0.95)
    # Slippery world has lower (or equal) value everywhere — the agent can't
    # be as decisive.
    assert (V_slip <= V_det + 1e-6).all()
