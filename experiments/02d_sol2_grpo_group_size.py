"""Solution 2d.2: GRPO group size sweep.

The group size G in GRPO controls the quality of the group-relative
baseline estimate. Smaller G → noisier baseline → noisier gradient.
Larger G → more accurate baseline but more compute per iteration.

We sweep G and measure final true reward at fixed compute budget
(i.e., fixed total samples = G × n_iterations × n_prompts_per_iter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy
from nano_agents.rlhf import PreferenceTask, train_grpo


def main() -> None:
    task = PreferenceTask(n_contexts=4, n_completions=6, reward_scale=2.0, seed=0)
    n_seeds = 5
    total_samples_budget = 12000  # fixed total

    group_sizes = [2, 4, 8, 16, 32]
    n_prompts_per_iter = 2

    final_returns = np.zeros((len(group_sizes), n_seeds))
    optimal = task.expected_reward_under_policy(
        np.eye(task.n_completions)[task.greedy_policy()])

    for i, G in enumerate(group_sizes):
        # Compute number of iterations to keep total samples constant.
        samples_per_iter = G * n_prompts_per_iter
        n_iterations = total_samples_budget // samples_per_iter

        for seed in range(n_seeds):
            policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
            hist = train_grpo(
                task, policy,
                n_iterations=n_iterations,
                group_size=G,
                n_prompts_per_iter=n_prompts_per_iter,
                n_epochs=4, lr=0.15, eps=0.2,
                rng=np.random.default_rng(seed + 11))
            probs = np.array([policy.probs(c) for c in range(task.n_contexts)])
            final_returns[i, seed] = task.expected_reward_under_policy(probs)
        print(f"  G = {G}: n_iters = {n_iterations}, "
              f"E[r] = {final_returns[i].mean():.3f} ± "
              f"{final_returns[i].std():.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(group_sizes, final_returns.mean(axis=1),
                 yerr=final_returns.std(axis=1) / np.sqrt(n_seeds),
                 fmt="o-", color="#3a7ebf", linewidth=2,
                 markersize=10, capsize=4,
                 markeredgecolor="white", markeredgewidth=1.2)
    ax.axhline(optimal, color="#222", linestyle="--", linewidth=1,
                alpha=0.6, label=f"optimal = {optimal:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("group size G")
    ax.set_ylabel("expected true reward")
    ax.set_title(
        f"GRPO group size at FIXED total sample budget = {total_samples_budget}.\n"
        f"All G perform similarly here — the benefit of large G scales with reward variance.",
        fontsize=11,
    )
    ax.legend(loc="lower right")
    ax.grid(which="both", alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(group_sizes)
    ax.set_xticklabels([str(g) for g in group_sizes])

    fig.tight_layout()
    out = Path("figures/02d_sol2_grpo_group_size.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
