"""Group Relative Policy Optimization (Shao et al. 2024).

GRPO is PPO without a learned critic. For each prompt x, sample a group
of G completions y_1, ..., y_G from the current policy. Compute their
rewards r_1, ..., r_G (from an oracle or learned reward model). Use the
group mean (and optionally std) as the baseline:

    A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + ε)

Then apply the standard PPO clipped surrogate. The learned-critic baseline
from A2C/PPO is replaced by the per-prompt empirical baseline.

This is dramatically simpler engineering for LLMs: no separate value model
to train, no critic-actor learning rate balancing, no value loss to tune.
For high-variance reasoning rollouts where the value function would be hard
to learn anyway, the group-relative baseline often works as well as a
learned critic — sometimes better.

Used by DeepSeek-R1 for training reasoning models, and now widely adopted.
"""

from __future__ import annotations

import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy
from nano_agents.rlhf.task import PreferenceTask


def _grad_log_pi_softmax(policy: SoftmaxPolicy, context: int, completion: int) -> np.ndarray:
    p = policy.probs(context)
    g = np.zeros_like(policy.theta)
    g[context] = -p
    g[context, completion] += 1.0
    return g


def grpo_iteration_step(
    policy: SoftmaxPolicy,
    old_log_probs_by_pair: dict,
    contexts: np.ndarray,
    completions: np.ndarray,
    advantages: np.ndarray,
    lr: float,
    eps: float,
):
    """One epoch of clipped-surrogate updates on a pre-collected GRPO batch.

    contexts:    shape (N,)  — the prompts associated with each sample.
    completions: shape (N,)  — the sampled completions.
    advantages:  shape (N,)  — already group-normalized.
    """
    grad = np.zeros_like(policy.theta)
    n_clipped = 0
    n = len(contexts)
    for i in range(n):
        c, a = int(contexts[i]), int(completions[i])
        new_lp = float(np.log(policy.probs(c)[a] + 1e-30))
        old_lp = old_log_probs_by_pair[(i,)]
        ratio = float(np.exp(new_lp - old_lp))
        A = advantages[i]
        unclipped = ratio * A
        clipped = float(np.clip(ratio, 1 - eps, 1 + eps)) * A
        if unclipped <= clipped:
            grad += A * ratio * _grad_log_pi_softmax(policy, c, a)
        else:
            n_clipped += 1
    grad /= max(n, 1)
    policy.theta += lr * grad
    return n_clipped / max(n, 1)


def train_grpo(
    task: PreferenceTask,
    policy: SoftmaxPolicy,
    n_iterations: int,
    group_size: int = 8,
    n_prompts_per_iter: int = 4,
    n_epochs: int = 4,
    lr: float = 0.1,
    eps: float = 0.2,
    normalize_by_std: bool = True,
    use_true_reward: bool = True,
    reward_model=None,
    rng=None,
    track_kl_against_ref: bool = False,
    ref_policy: SoftmaxPolicy | None = None,
    kl_coef: float = 0.0,
):
    """Train policy via GRPO.

    use_true_reward: if True, use task.reward as the oracle. If False, must
        pass a fitted reward_model with a predict(c, a) method.
    track_kl_against_ref: if True, also apply a -kl_coef * KL(pi||pi_ref)
        penalty (the RLHF-style "reference KL"). Useful for the figure that
        demonstrates reward hacking.
    """
    rng = rng if rng is not None else np.random.default_rng()
    history = {
        "expected_return": np.zeros(n_iterations),
        "kl_to_ref": np.zeros(n_iterations) if track_kl_against_ref else None,
        "clip_fraction": np.zeros(n_iterations),
    }

    for it in range(n_iterations):
        # Collect: for each of n_prompts_per_iter prompts, sample G completions.
        contexts_list, completions_list, rewards_list = [], [], []
        # Sample contexts (with replacement to allow repeats).
        for _ in range(n_prompts_per_iter):
            c = int(rng.integers(task.n_contexts))
            probs_c = policy.probs(c)
            group_completions = rng.choice(
                task.n_completions, size=group_size, p=probs_c)
            for a in group_completions:
                if use_true_reward:
                    r = task.reward(c, int(a))
                else:
                    r = reward_model.predict(c, int(a))
                contexts_list.append(c)
                completions_list.append(int(a))
                rewards_list.append(r)
        contexts = np.array(contexts_list)
        completions = np.array(completions_list)
        rewards = np.array(rewards_list, dtype=float)

        # Group-relative advantages. Group = samples that share a context.
        advantages = np.zeros_like(rewards)
        for c in np.unique(contexts):
            mask = contexts == c
            r_in_group = rewards[mask]
            base = r_in_group.mean()
            adv = r_in_group - base
            if normalize_by_std:
                adv = adv / (r_in_group.std() + 1e-8)
            advantages[mask] = adv

        # Snapshot old log-probs.
        old_log_probs_by_pair = {
            (i,): float(np.log(policy.probs(int(contexts[i]))[int(completions[i])] + 1e-30))
            for i in range(len(contexts))
        }

        # Multiple epochs of PPO-style updates.
        clip_fracs = []
        for epoch in range(n_epochs):
            cf = grpo_iteration_step(policy, old_log_probs_by_pair,
                                       contexts, completions, advantages,
                                       lr=lr, eps=eps)
            clip_fracs.append(cf)

        # Optional reference-KL penalty.
        if track_kl_against_ref and ref_policy is not None and kl_coef > 0:
            # Apply gradient step on KL(π || π_ref) for each visited context.
            for c in np.unique(contexts):
                p = policy.probs(int(c))
                p_ref = ref_policy.probs(int(c))
                # KL = sum p * log(p / p_ref). Gradient w.r.t. theta[c, :]:
                # d KL / d theta[c, a] = p_a * (log(p_a/p_ref_a) - KL)
                kl_term = float(np.sum(p * (np.log(p + 1e-30) - np.log(p_ref + 1e-30))))
                g = p * (np.log(p + 1e-30) - np.log(p_ref + 1e-30) - kl_term)
                policy.theta[int(c)] -= lr * kl_coef * g

        # Diagnostics: expected return under the current policy.
        cur_policy_probs = np.array(
            [policy.probs(c) for c in range(task.n_contexts)])
        history["expected_return"][it] = task.expected_reward_under_policy(
            cur_policy_probs)
        history["clip_fraction"][it] = float(np.mean(clip_fracs))
        if track_kl_against_ref and ref_policy is not None:
            kl_total = 0.0
            for c in range(task.n_contexts):
                p = policy.probs(c)
                p_ref = ref_policy.probs(c)
                kl_total += float(
                    np.sum(p * (np.log(p + 1e-30) - np.log(p_ref + 1e-30))))
            history["kl_to_ref"][it] = kl_total / task.n_contexts

    return history
