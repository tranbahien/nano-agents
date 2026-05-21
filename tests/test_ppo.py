"""Tests for the PPO trainer."""

from __future__ import annotations

import numpy as np

from nano_agents.mdp import GridWorld, value_iteration
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    collect_batch,
    compute_batch_advantages,
    ppo_update_step,
    train_ppo,
)


def test_ppo_collect_batch():
    env = GridWorld(rows=3, cols=3, terminals={(2, 2): 1.0},
                     step_reward=-0.04, slip=0.0)
    policy = SoftmaxPolicy(env.nS, env.nA)
    rng = np.random.default_rng(0)
    batch = collect_batch(env, policy, n_trajectories=5, rng=rng,
                            start_state=(0, 0))
    assert len(batch) == 5
    for traj in batch:
        assert len(traj) >= 1
        # Each step is (s_idx, a, r)
        assert all(len(t) == 3 for t in traj)


def test_ppo_compute_advantages():
    env = GridWorld(rows=3, cols=3, terminals={(2, 2): 1.0},
                     step_reward=-0.04, slip=0.0)
    policy = SoftmaxPolicy(env.nS, env.nA)
    baseline = TabularBaseline(env.nS)
    rng = np.random.default_rng(0)
    batch = collect_batch(env, policy, n_trajectories=3, rng=rng,
                            start_state=(0, 0))
    samples = compute_batch_advantages(batch, baseline, gamma=0.95, lam=0.95)
    assert len(samples) > 0
    # Each sample is (s, a, adv, target)
    for sample in samples:
        assert len(sample) == 4


def test_ppo_clipping_no_op_at_init():
    """At iteration start, ratio = 1.0 exactly, so clipping is never triggered."""
    env = GridWorld(rows=3, cols=3, terminals={(2, 2): 1.0},
                     step_reward=-0.04, slip=0.0)
    policy = SoftmaxPolicy(env.nS, env.nA)
    baseline = TabularBaseline(env.nS)
    rng = np.random.default_rng(0)
    batch = collect_batch(env, policy, n_trajectories=3, rng=rng,
                            start_state=(0, 0))
    samples = compute_batch_advantages(batch, baseline, gamma=0.95, lam=0.95)
    old_log_probs = np.array(
        [float(np.log(policy.probs(s)[a] + 1e-30)) for (s, a, _, _) in samples])
    advantages = np.array([s[2] for s in samples])
    indices = np.arange(len(samples))
    # First step: ratios should be exactly 1.0, no clipping.
    ratio_mean, kl, clip_fraction = ppo_update_step(
        policy, old_log_probs, samples, indices, advantages,
        lr=0.0, eps=0.2,  # lr=0 so policy doesn't move
    )
    assert abs(ratio_mean - 1.0) < 1e-9
    assert clip_fraction == 0.0
    assert abs(kl) < 1e-9


def test_ppo_learns_simple_gridworld():
    """PPO should learn a near-optimal policy on a tiny gridworld."""
    env = GridWorld(rows=3, cols=3, terminals={(2, 2): 1.0},
                     step_reward=-0.04, slip=0.0)
    V_star, _, _ = value_iteration(env, gamma=0.95)

    policy = SoftmaxPolicy(env.nS, env.nA)
    baseline = TabularBaseline(env.nS)
    train_ppo(env, policy, baseline,
               n_iterations=200,
               n_trajectories_per_iter=8,
               n_epochs=4,
               lr_actor=0.1, lr_critic=0.2,
               gamma=0.95, lam=0.95, eps=0.2,
               rng=np.random.default_rng(0))

    start_idx = env.state_to_idx[(0, 0)]
    pg_action = int(policy.theta[start_idx].argmax())
    P, R = env.transition_tensors()
    Q_star = R + 0.95 * (P @ V_star)
    optimal_actions = np.where(Q_star[start_idx] >= Q_star[start_idx].max() - 1e-6)[0]
    assert pg_action in optimal_actions, (
        f"PPO picked action {pg_action} at start; "
        f"optimal actions are {optimal_actions.tolist()}"
    )
