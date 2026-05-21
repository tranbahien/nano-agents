"""Figure: PPO (with learned reward model) vs DPO (direct).

On the same synthetic preference task, train two methods to convergence
and plot their expected reward over time, all measured against the
*true* reward function the task is hiding.

The contrast worth seeing:
  - DPO is much simpler — just supervised gradient descent on preferences.
  - PPO has two phases: train reward model on preferences, then train
    policy on the reward model.
  - Both should reach similar final performance if everything works.
  - DPO is often more sample-efficient because no reward-model error
    accumulates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy
from nano_agents.rlhf import (
    PreferenceTask,
    TabularRewardModel,
    dpo_loss,
    train_dpo,
    train_grpo,
)


def main() -> None:
    task = PreferenceTask(n_contexts=4, n_completions=4, reward_scale=2.0, seed=0)
    pref_counts = [25, 100, 400, 1600]
    n_seeds = 5

    ppo_returns = np.zeros((len(pref_counts), n_seeds))
    dpo_returns = np.zeros((len(pref_counts), n_seeds))
    optimal_return = task.expected_reward_under_policy(
        np.eye(task.n_completions)[task.greedy_policy()])

    for i, n_prefs in enumerate(pref_counts):
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 7)
            prefs = task.collect_preferences(n_samples=n_prefs, rng=rng)

            # PPO-with-reward-model pipeline.
            rm = TabularRewardModel(task.n_contexts, task.n_completions)
            rm.fit(prefs, n_steps=600, lr=0.5)
            ppo_policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
            train_grpo(
                task, ppo_policy,
                n_iterations=200, group_size=8, n_prompts_per_iter=2,
                n_epochs=4, lr=0.2, eps=0.2,
                use_true_reward=False, reward_model=rm,
                rng=np.random.default_rng(seed * 13 + 3))
            ppo_probs = np.array([ppo_policy.probs(c) for c in range(task.n_contexts)])
            ppo_returns[i, seed] = task.expected_reward_under_policy(ppo_probs)

            # DPO directly on preferences. Use larger lr / more steps because
            # the DPO loss decays logarithmically in the policy log-ratio
            # margin and needs many steps to drive the policy to a sharp
            # solution.
            ref = SoftmaxPolicy(task.n_contexts, task.n_completions)  # uniform reference
            dpo_policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
            train_dpo(dpo_policy, ref, prefs, n_steps=1500, lr=1.0, beta=0.1)
            dpo_probs = np.array([dpo_policy.probs(c) for c in range(task.n_contexts)])
            dpo_returns[i, seed] = task.expected_reward_under_policy(dpo_probs)
        print(f"  n_prefs = {n_prefs}: "
              f"PPO = {ppo_returns[i].mean():.3f}, "
              f"DPO = {dpo_returns[i].mean():.3f}, "
              f"optimal = {optimal_return:.3f}")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.errorbar(pref_counts, ppo_returns.mean(axis=1),
                 yerr=ppo_returns.std(axis=1) / np.sqrt(n_seeds),
                 fmt="o-", color="#c44e52", linewidth=2,
                 markersize=10, capsize=4,
                 label="PPO (reward model → GRPO with that model)")
    ax.errorbar(pref_counts, dpo_returns.mean(axis=1),
                 yerr=dpo_returns.std(axis=1) / np.sqrt(n_seeds),
                 fmt="s-", color="#3a7ebf", linewidth=2,
                 markersize=10, capsize=4,
                 label=r"DPO (direct, $\beta = 0.1$)")
    ax.axhline(optimal_return, color="#222", linestyle="--", linewidth=1,
                alpha=0.6, label=f"optimal (E[r*] = {optimal_return:.2f})")

    ax.set_xscale("log")
    ax.set_xlabel("# preference pairs available")
    ax.set_ylabel("expected reward under learned policy")
    ax.set_title(
        f"PPO (with reward model) vs DPO (direct) on the same task and data.\n"
        f"mean ± SEM over {n_seeds} seeds",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02d_ppo_vs_dpo.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
