"""Tests for the rlhf subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy
from nano_agents.rlhf import (
    PreferenceTask,
    TabularRewardModel,
    dpo_loss,
    train_dpo,
    train_grpo,
)


def test_preference_task_setup():
    task = PreferenceTask(n_contexts=3, n_completions=4, seed=0)
    assert task.true_rewards.shape == (3, 4)
    # Greedy policy returns the argmax completion per context.
    greedy = task.greedy_policy()
    for c in range(3):
        assert task.true_rewards[c, greedy[c]] == task.true_rewards[c].max()


def test_preference_pair_consistent_with_rewards():
    """The winner should be the higher-reward completion more often than not."""
    task = PreferenceTask(n_contexts=2, n_completions=2, reward_scale=3.0, seed=0)
    rng = np.random.default_rng(0)
    wins_for_higher = 0
    n_samples = 5000
    for _ in range(n_samples):
        c = int(rng.integers(task.n_contexts))
        w, l = task.sample_preference_pair(c, rng)
        if task.reward(c, w) > task.reward(c, l):
            wins_for_higher += 1
    # Should be well above 50% (true rewards have meaningful differences).
    assert wins_for_higher / n_samples > 0.70


def test_reward_model_recovers_ordering():
    """After fitting on many preferences, the reward model's argmax
    should match the true argmax per context."""
    task = PreferenceTask(n_contexts=4, n_completions=4, seed=0)
    rng = np.random.default_rng(0)
    prefs = task.collect_preferences(n_samples=2000, rng=rng)
    rm = TabularRewardModel(task.n_contexts, task.n_completions)
    rm.fit(prefs, n_steps=600, lr=0.5)
    for c in range(task.n_contexts):
        rm_best = int(rm.r[c].argmax())
        true_best = int(task.true_rewards[c].argmax())
        assert rm_best == true_best, (
            f"Mismatch at context {c}: rm_best={rm_best}, true_best={true_best}")


def test_dpo_loss_at_reference_is_log2():
    """When policy == ref_policy, the DPO margin is zero, so loss = -log sigmoid(0) = log 2."""
    policy = SoftmaxPolicy(nS=2, nA=3)
    ref = SoftmaxPolicy(nS=2, nA=3)
    prefs = [(0, 0, 1), (1, 2, 0)]
    L = dpo_loss(policy, ref, prefs, beta=0.1)
    assert abs(L - np.log(2)) < 1e-6


def test_dpo_pushes_winners_up():
    """After training, the policy should prefer winners over losers."""
    task = PreferenceTask(n_contexts=2, n_completions=3, seed=0)
    rng = np.random.default_rng(0)
    prefs = task.collect_preferences(n_samples=500, rng=rng)
    ref = SoftmaxPolicy(task.n_contexts, task.n_completions)
    policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
    train_dpo(policy, ref, prefs, n_steps=400, lr=0.5, beta=0.1)
    # Check that the policy argmax matches the true argmax.
    for c in range(task.n_contexts):
        pg_best = int(policy.theta[c].argmax())
        true_best = int(task.true_rewards[c].argmax())
        assert pg_best == true_best


def test_grpo_learns_optimal_policy():
    task = PreferenceTask(n_contexts=2, n_completions=3, seed=0)
    policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
    train_grpo(task, policy,
                n_iterations=100, group_size=8, n_prompts_per_iter=2,
                n_epochs=4, lr=0.2, eps=0.2,
                rng=np.random.default_rng(0))
    for c in range(task.n_contexts):
        pg_best = int(policy.theta[c].argmax())
        true_best = int(task.true_rewards[c].argmax())
        assert pg_best == true_best
