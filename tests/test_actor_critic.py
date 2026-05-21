"""Tests for the actor-critic / GAE / baseline code in policy_gradient."""

from __future__ import annotations

import numpy as np

from nano_agents.mdp import GridWorld, value_iteration
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    entropy_grad,
    gae,
    n_step_returns,
    policy_entropy,
    train_a2c,
)


def test_tabular_baseline_update():
    b = TabularBaseline(nS=3, init_value=0.0)
    td = b.update(s_idx=1, target=2.0, lr=0.5)
    assert np.isclose(b.V[1], 1.0)
    assert np.isclose(td, 2.0)


def test_n_step_returns():
    # Trajectory of 3 rewards, gamma = 0.5, values all zero except boot.
    rewards = [1.0, 1.0, 1.0]
    values = np.array([0.0, 0.0, 0.0])
    G = n_step_returns(rewards, values, next_value=0.0, gamma=0.5, n=10)
    # G[0] = 1 + 0.5*1 + 0.25*1 = 1.75. G[1] = 1.5. G[2] = 1.0.
    assert np.allclose(G, [1.75, 1.5, 1.0])


def test_gae_collapses_to_td_at_lam_zero():
    """GAE(λ=0) should equal TD(0) advantages: r + γ V(s') - V(s)."""
    rewards = [1.0, 0.5, -0.2]
    values = np.array([0.3, 0.2, 0.1])
    next_value = 0.0
    gamma = 0.9
    A_gae = gae(rewards, values, next_value, gamma=gamma, lam=0.0)
    A_td = np.zeros(3)
    for t in range(3):
        v_next = next_value if t == 2 else values[t + 1]
        A_td[t] = rewards[t] + gamma * v_next - values[t]
    assert np.allclose(A_gae, A_td, atol=1e-10)


def test_gae_at_lam_one_collapses_to_monte_carlo():
    """GAE(λ=1) should equal G_t - V(s_t) (Monte Carlo advantage)."""
    rewards = [1.0, 0.5, -0.2]
    values = np.array([0.3, 0.2, 0.1])
    next_value = 0.0
    gamma = 0.9
    A_gae = gae(rewards, values, next_value, gamma=gamma, lam=1.0)
    # Compute MC advantage.
    T = len(rewards)
    G = np.zeros(T)
    running = next_value
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t] = running
    A_mc = G - values
    assert np.allclose(A_gae, A_mc, atol=1e-10)


def test_entropy_grad_finite_difference():
    rng = np.random.default_rng(0)
    policy = SoftmaxPolicy(nS=3, nA=4)
    policy.theta = rng.standard_normal((3, 4)) * 0.3

    s = 1
    g_analytic = entropy_grad(policy, s)

    eps = 1e-6
    for ap in range(policy.nA):
        policy.theta[s, ap] += eps
        H_plus = policy_entropy(policy, s)
        policy.theta[s, ap] -= 2 * eps
        H_minus = policy_entropy(policy, s)
        policy.theta[s, ap] += eps
        numerical = (H_plus - H_minus) / (2 * eps)
        assert abs(g_analytic[s, ap] - numerical) < 1e-4, (
            f"entropy grad mismatch at action {ap}: "
            f"analytical={g_analytic[s, ap]:.6f}, numerical={numerical:.6f}"
        )


def test_a2c_learns_simple_gridworld():
    env = GridWorld(rows=3, cols=3, terminals={(2, 2): 1.0}, step_reward=-0.04,
                     slip=0.0)
    V_star, _, _ = value_iteration(env, gamma=0.95)

    policy = SoftmaxPolicy(nS=env.nS, nA=env.nA)
    baseline = TabularBaseline(env.nS, init_value=0.0)
    train_a2c(env, policy, baseline,
              n_episodes=1500, lr_actor=0.1, lr_critic=0.2,
              gamma=0.95, lam=0.95, rng=np.random.default_rng(0))

    # At the start, the policy should prefer an optimal action.
    start_idx = env.state_to_idx[(0, 0)]
    pg_action = int(policy.theta[start_idx].argmax())
    P, R = env.transition_tensors()
    Q_star = R + 0.95 * (P @ V_star)
    optimal_actions = np.where(Q_star[start_idx] >= Q_star[start_idx].max() - 1e-6)[0]
    assert pg_action in optimal_actions

    # The critic should have a reasonable approximation of V*.
    V_phi = baseline.V
    err = np.max(np.abs(V_phi - V_star))
    assert err < 0.2, f"Critic error too large: {err}"
