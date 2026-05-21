"""Tests for the policy_gradient subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.mdp import GridWorld, value_iteration
from nano_agents.policy_gradient import (
    GaussianPolicy,
    SoftmaxPolicy,
    compute_returns,
    train_reinforce,
)


def test_softmax_policy_basic():
    p = SoftmaxPolicy(nS=4, nA=3)
    # Uniform initialization.
    probs = p.probs(0)
    assert np.allclose(probs, 1.0 / 3.0)
    assert np.isclose(probs.sum(), 1.0)


def test_softmax_grad_log_prob_sanity():
    """Check grad log pi by finite differences."""
    rng = np.random.default_rng(0)
    p = SoftmaxPolicy(nS=3, nA=4)
    p.theta = rng.standard_normal((3, 4)) * 0.5

    s = 1
    a = 2
    g = p.grad_log_prob(s, a)

    # Finite-difference check on a few coordinates.
    eps = 1e-6
    for sp in range(3):
        for ap in range(4):
            p.theta[sp, ap] += eps
            lp_plus = p.log_prob(s, a)
            p.theta[sp, ap] -= 2 * eps
            lp_minus = p.log_prob(s, a)
            p.theta[sp, ap] += eps
            numerical = (lp_plus - lp_minus) / (2 * eps)
            assert abs(g[sp, ap] - numerical) < 1e-4, (
                f"grad mismatch at ({sp},{ap}): analytical={g[sp,ap]:.6f}, "
                f"numerical={numerical:.6f}"
            )


def test_compute_returns():
    rewards = [1.0, 1.0, 1.0]
    gamma = 0.5
    G = compute_returns(rewards, gamma)
    # G[2] = 1.0, G[1] = 1.0 + 0.5*1.0 = 1.5, G[0] = 1.0 + 0.5*1.5 = 1.75.
    assert np.allclose(G, [1.75, 1.5, 1.0])


def test_gaussian_policy_grad():
    """Finite-difference check for Gaussian policy grad."""
    rng = np.random.default_rng(0)
    p = GaussianPolicy(mu_init=0.5, sigma=1.5)
    a = float(rng.normal())

    g = p.grad_log_prob_mu(a)
    eps = 1e-6
    p.mu += eps
    lp_plus = p.log_prob(a)
    p.mu -= 2 * eps
    lp_minus = p.log_prob(a)
    p.mu += eps
    numerical = (lp_plus - lp_minus) / (2 * eps)
    assert abs(g - numerical) < 1e-4


def test_reinforce_learns_simple_gridworld():
    """REINFORCE should learn a near-optimal policy on a tiny gridworld."""
    env = GridWorld(rows=3, cols=3, terminals={(2, 2): 1.0}, step_reward=-0.04,
                     slip=0.0)
    V_star, pi_star, _ = value_iteration(env, gamma=0.95)

    policy = SoftmaxPolicy(nS=env.nS, nA=env.nA)
    rng = np.random.default_rng(0)
    train_reinforce(env, policy, n_episodes=2000, lr=0.1, gamma=0.95,
                     baseline="mean", rng=rng)

    # Compare REINFORCE's greedy policy against VI's at the start state.
    start_idx = env.state_to_idx[(0, 0)]
    # At (0, 0) the optimal action is Right or Down (both lead to the goal).
    # We check that REINFORCE's argmax matches at least one optimal action.
    pg_action = int(policy.theta[start_idx].argmax())
    # Compute one-step lookahead values from VI to check pg_action is optimal.
    P, R = env.transition_tensors()
    Q_star = R + 0.95 * (P @ V_star)
    optimal_actions = np.where(Q_star[start_idx] >= Q_star[start_idx].max() - 1e-6)[0]
    assert pg_action in optimal_actions, (
        f"REINFORCE picked action {pg_action} at start; "
        f"optimal actions are {optimal_actions.tolist()}"
    )
