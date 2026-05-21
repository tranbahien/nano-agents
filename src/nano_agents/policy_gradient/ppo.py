"""Proximal Policy Optimization (Schulman et al. 2017).

The core idea: maximize the clipped surrogate objective

    L^CLIP(θ) = E[ min( r(θ) A,  clip(r(θ), 1-ε, 1+ε) A ) ]

where r(θ) = π_θ(a|s) / π_{θ_old}(a|s) is the importance ratio. The clip
prevents the new policy from moving too far from the data-generating policy
without an explicit KL constraint.

Two practical benefits over vanilla policy gradient:
  1. We can take MANY gradient steps per batch (epochs > 1) because the
     clip bounds how much each step can change the policy.
  2. The variance of the gradient is bounded, since ratios are bounded.

For our tabular softmax policy, the clip is straightforward to implement
because we have closed-form expressions for log pi and its gradient.
"""

from __future__ import annotations

import numpy as np

from nano_agents.mdp import GridWorld
from nano_agents.policy_gradient.advantage import gae
from nano_agents.policy_gradient.baselines import TabularBaseline
from nano_agents.policy_gradient.policies import SoftmaxPolicy
from nano_agents.policy_gradient.reinforce import collect_trajectory


def collect_batch(env: GridWorld, policy: SoftmaxPolicy, n_trajectories: int,
                    rng, max_steps: int = 200, start_state=None):
    """Collect a batch of trajectories under the current policy."""
    non_terminal = [s for s in env.states if not env.is_terminal(s)]
    batch = []  # list of trajectories
    for _ in range(n_trajectories):
        start = (start_state if start_state is not None
                  else non_terminal[rng.integers(len(non_terminal))])
        traj = collect_trajectory(env, policy, start, rng, max_steps=max_steps)
        batch.append(traj)
    return batch


def compute_batch_advantages(batch, baseline: TabularBaseline, gamma: float,
                               lam: float):
    """Compute GAE advantages for every step in a batch of trajectories.

    Returns a flat list of (state_idx, action, advantage, target_return) tuples
    suitable for shuffling and mini-batched updates.
    """
    samples = []  # (s, a, advantage, target_return)
    for traj in batch:
        if len(traj) == 0:
            continue
        states = np.array([s for (s, _, _) in traj])
        rewards = [r for (_, _, r) in traj]
        values = baseline.values(states)
        adv = gae(rewards, values, next_value=0.0, gamma=gamma, lam=lam)
        target = adv + values  # critic regression target
        for t, (s, a, _) in enumerate(traj):
            samples.append((int(s), int(a), float(adv[t]),
                             float(target[t])))
    return samples


def ppo_update_step(
    policy: SoftmaxPolicy,
    old_log_probs: np.ndarray,
    samples: list,
    indices: np.ndarray,
    advantages: np.ndarray,
    lr: float,
    eps: float,
    ent_coef: float = 0.0,
    n_trajectories: int = 1,
):
    """Apply one gradient step of the PPO-clipped surrogate on the given indices.

    The gradient is normalized per trajectory, not per sample, to match A2C's
    convention (where each step contributes to the gradient and the per-batch
    sum scales naturally with trajectory length).

    Returns diagnostics: (mean_ratio, kl_estimate, clip_fraction).
    """
    grad = np.zeros_like(policy.theta)
    kl_sum = 0.0
    n_clipped = 0
    ratio_sum = 0.0
    for idx in indices:
        s, a, _, _ = samples[idx]
        p = policy.probs(s)
        new_log_prob = float(np.log(p[a] + 1e-30))
        old_log_prob = old_log_probs[idx]
        ratio = float(np.exp(new_log_prob - old_log_prob))
        A = advantages[idx]

        unclipped_obj = ratio * A
        clipped_obj = float(np.clip(ratio, 1.0 - eps, 1.0 + eps)) * A

        if unclipped_obj <= clipped_obj:
            # Unclipped branch is active; gradient is A * r * grad_log_pi.
            score = np.zeros_like(policy.theta)
            score[s] = -p
            score[s, a] += 1.0
            grad += A * ratio * score
        else:
            # Clipped branch is active; gradient is zero in this region.
            n_clipped += 1

        if ent_coef > 0:
            log_p = np.log(p + 1e-30)
            H = -float(np.sum(p * log_p))
            ent_score = np.zeros_like(policy.theta)
            ent_score[s] = p * (-log_p - H)
            grad += ent_coef * ent_score

        ratio_sum += ratio
        kl_sum += old_log_prob - new_log_prob

    n = len(indices)
    # Normalize per trajectory: matches A2C's per-step accumulation when there
    # are multiple steps per trajectory.
    grad /= max(n_trajectories, 1)
    policy.theta += lr * grad
    return (ratio_sum / n, kl_sum / n, n_clipped / n)


def train_ppo(
    env: GridWorld,
    policy: SoftmaxPolicy,
    baseline: TabularBaseline,
    n_iterations: int,
    n_trajectories_per_iter: int = 16,
    n_epochs: int = 4,
    lr_actor: float = 0.05,
    lr_critic: float = 0.2,
    gamma: float = 0.95,
    lam: float = 0.95,
    eps: float = 0.2,
    ent_coef: float = 0.0,
    normalize_adv: bool = True,
    rng=None,
    start_state=None,
):
    """Train a tabular policy with PPO.

    Returns a history dict with per-iteration diagnostics.
    """
    rng = rng if rng is not None else np.random.default_rng()
    history = {
        "returns": np.zeros(n_iterations),
        "ratio_mean": np.zeros(n_iterations),
        "kl": np.zeros(n_iterations),
        "clip_fraction": np.zeros(n_iterations),
    }

    for it in range(n_iterations):
        # 1. Collect a batch under the current policy.
        batch = collect_batch(env, policy, n_trajectories_per_iter, rng,
                                start_state=start_state)
        samples = compute_batch_advantages(batch, baseline, gamma, lam)
        if len(samples) == 0:
            continue
        # Snapshot old log-probs for the samples we have.
        old_log_probs = np.array(
            [float(np.log(policy.probs(s)[a] + 1e-30)) for (s, a, _, _) in samples])
        advantages = np.array([s[2] for s in samples])
        targets = np.array([s[3] for s in samples])
        states = np.array([s[0] for s in samples])

        if normalize_adv and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Multiple epochs of clipped-surrogate updates.
        ratio_means, kls, clip_fracs = [], [], []
        for epoch in range(n_epochs):
            indices = np.arange(len(samples))
            rng.shuffle(indices)
            r, k, cf = ppo_update_step(policy, old_log_probs, samples, indices,
                                          advantages, lr=lr_actor, eps=eps,
                                          ent_coef=ent_coef,
                                          n_trajectories=n_trajectories_per_iter)
            ratio_means.append(r); kls.append(k); clip_fracs.append(cf)

        # 3. Critic update on the same batch.
        for idx in range(len(samples)):
            s = states[idx]
            target = targets[idx]
            baseline.update(s, target, lr_critic)

        # 4. Diagnostics.
        all_returns = [float(sum(r for (_, _, r) in traj)) for traj in batch]
        history["returns"][it] = float(np.mean(all_returns))
        history["ratio_mean"][it] = float(np.mean(ratio_means))
        history["kl"][it] = float(np.mean(kls))
        history["clip_fraction"][it] = float(np.mean(clip_fracs))

    return history
