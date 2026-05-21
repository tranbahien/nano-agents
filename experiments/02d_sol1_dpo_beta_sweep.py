"""Solution 2d.1: DPO β sweep.

β controls how much the policy can move from the reference. Small β →
high temperature in the optimal DPO solution → soft policy. Large β →
low temperature → sharp policy.

Sweep β and show both:
  - The implicit temperature of the resulting policy (entropy)
  - The expected true reward achieved
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy
from nano_agents.rlhf import PreferenceTask, train_dpo


def policy_entropy_per_context(policy: SoftmaxPolicy) -> np.ndarray:
    H = np.zeros(policy.nS)
    for c in range(policy.nS):
        p = policy.probs(c)
        H[c] = -np.sum(p * np.log(p + 1e-30))
    return H


def main() -> None:
    task = PreferenceTask(n_contexts=4, n_completions=4, reward_scale=2.0, seed=0)
    n_seeds = 5
    n_prefs = 500
    n_steps = 1500
    lr = 1.0

    betas = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    returns = np.zeros((len(betas), n_seeds))
    entropies = np.zeros((len(betas), n_seeds))

    optimal = task.expected_reward_under_policy(
        np.eye(task.n_completions)[task.greedy_policy()])

    for i, beta in enumerate(betas):
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 7)
            prefs = task.collect_preferences(n_samples=n_prefs, rng=rng)
            ref = SoftmaxPolicy(task.n_contexts, task.n_completions)
            policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
            train_dpo(policy, ref, prefs, n_steps=n_steps, lr=lr, beta=beta)
            probs = np.array([policy.probs(c) for c in range(task.n_contexts)])
            returns[i, seed] = task.expected_reward_under_policy(probs)
            entropies[i, seed] = float(np.mean(policy_entropy_per_context(policy)))
        print(f"  β = {beta}: E[r] = {returns[i].mean():.3f}, "
              f"entropy = {entropies[i].mean():.3f}")

    H_uniform = np.log(task.n_completions)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

    axes[0].errorbar(betas, returns.mean(axis=1),
                       yerr=returns.std(axis=1) / np.sqrt(n_seeds),
                       fmt="o-", color="#3a7ebf", linewidth=2,
                       markersize=9, capsize=4,
                       markeredgecolor="white", markeredgewidth=1.2)
    axes[0].axhline(optimal, color="#222", linestyle="--", linewidth=1,
                     alpha=0.6, label=f"true optimum ({optimal:.2f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"DPO $\beta$")
    axes[0].set_ylabel("expected true reward")
    axes[0].set_title(r"Inverted-U: $\beta$ too small can't move, too large breaks optimization")
    axes[0].legend(loc="lower right")
    axes[0].grid(which="both", alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].errorbar(betas, entropies.mean(axis=1),
                       yerr=entropies.std(axis=1) / np.sqrt(n_seeds),
                       fmt="o-", color="#c44e52", linewidth=2,
                       markersize=9, capsize=4,
                       markeredgecolor="white", markeredgewidth=1.2)
    axes[1].axhline(H_uniform, color="#222", linestyle="--", linewidth=1,
                     alpha=0.6, label=f"uniform entropy ({H_uniform:.2f})")
    axes[1].axhline(0.0, color="#888", linestyle=":", linewidth=1, alpha=0.4,
                     label="deterministic (entropy = 0)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"DPO $\beta$")
    axes[1].set_ylabel(r"average policy entropy")
    axes[1].set_title(r"Entropy minimum at the same $\beta$ where return peaks")
    axes[1].legend(loc="upper right")
    axes[1].grid(which="both", alpha=0.3); axes[1].set_axisbelow(True)

    fig.suptitle(
        f"DPO β controls the implicit temperature of the trained policy.\n"
        f"mean ± SEM over {n_seeds} seeds, {n_prefs} preferences each",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02d_sol1_dpo_beta_sweep.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
