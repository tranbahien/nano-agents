"""Figure: GRPO group-relative baseline mechanism.

Two-panel demonstration:

  Left: For one context, sample G=8 completions from the current policy.
        Plot their rewards. Show:
          - The group mean (baseline)
          - The advantages = rewards - mean
          - (Optional) Normalized advantages = (r - mean) / std

  Right: Compare three baselines for variance reduction on the same task:
          - No baseline (vanilla policy gradient)
          - Group mean only
          - Group mean + std normalization

        Plot empirical gradient variance over training iterations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.policy_gradient import SoftmaxPolicy
from nano_agents.rlhf import PreferenceTask, train_grpo


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    # ----- Left: visualize the baseline for one prompt -----
    task = PreferenceTask(n_contexts=4, n_completions=8, reward_scale=2.0, seed=1)
    rng = np.random.default_rng(0)
    c = 2
    policy = SoftmaxPolicy(task.n_contexts, task.n_completions)
    G = 8
    probs_c = policy.probs(c)
    sampled = rng.choice(task.n_completions, size=G, p=probs_c)
    rewards = np.array([task.reward(c, a) for a in sampled])
    mean_r = rewards.mean()
    std_r = rewards.std()
    advantages = rewards - mean_r
    normalized = advantages / (std_r + 1e-8)

    ax = axes[0]
    x = np.arange(G)
    width = 0.35
    ax.bar(x - width / 2, rewards, width, color="#dd8452",
            label="reward $r_i$", edgecolor="white", linewidth=0.7)
    ax.bar(x + width / 2, advantages, width, color="#3a7ebf",
            label=r"$A_i = r_i - \mathrm{mean}(r)$", edgecolor="white",
            linewidth=0.7)
    ax.axhline(mean_r, color="#c44e52", linestyle="--", linewidth=2,
                label=f"group mean = {mean_r:.2f}")
    ax.axhline(0, color="#222", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("group sample $i$")
    ax.set_ylabel("reward / advantage")
    ax.set_title(f"GRPO baseline for one prompt (G = {G} completions)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([f"y_{i}" for i in range(G)])

    # ----- Right: variance reduction comparison -----
    task2 = PreferenceTask(n_contexts=4, n_completions=4, seed=0)
    n_iterations = 60
    n_seeds = 5

    configs = [
        ("no baseline", "#888888", "none"),
        ("group mean only", "#dd8452", "mean"),
        ("group mean + std", "#3a7ebf", "mean_std"),
    ]
    all_returns = {name: np.zeros((n_seeds, n_iterations))
                    for (name, _, _) in configs}

    for name, color, mode in configs:
        for seed in range(n_seeds):
            policy = SoftmaxPolicy(task2.n_contexts, task2.n_completions)
            # Custom training: re-implement minimal GRPO with switchable baseline.
            rng = np.random.default_rng(seed + 10)
            G = 6
            n_prompts = 3
            for it in range(n_iterations):
                ctxs = []; comps = []; rs = []
                for _ in range(n_prompts):
                    c = int(rng.integers(task2.n_contexts))
                    pc = policy.probs(c)
                    sg = rng.choice(task2.n_completions, size=G, p=pc)
                    for a in sg:
                        ctxs.append(c); comps.append(int(a))
                        rs.append(task2.reward(c, int(a)))
                ctxs = np.array(ctxs); comps = np.array(comps); rs = np.array(rs)
                # Baselines.
                adv = np.zeros_like(rs)
                if mode == "none":
                    adv = rs.copy()
                else:
                    for c in np.unique(ctxs):
                        mask = ctxs == c
                        r_in = rs[mask]
                        if mode == "mean":
                            adv[mask] = r_in - r_in.mean()
                        else:
                            adv[mask] = (r_in - r_in.mean()) / (r_in.std() + 1e-8)
                # Apply a single policy gradient step (no clipping for the
                # comparison — we want baseline effect alone).
                grad = np.zeros_like(policy.theta)
                for i in range(len(ctxs)):
                    c, a, A = int(ctxs[i]), int(comps[i]), float(adv[i])
                    p = policy.probs(c)
                    g = np.zeros_like(policy.theta)
                    g[c] = -p
                    g[c, a] += 1.0
                    grad += A * g
                grad /= len(ctxs)
                policy.theta += 0.2 * grad
                # Record expected return.
                probs = np.array(
                    [policy.probs(cx) for cx in range(task2.n_contexts)])
                all_returns[name][seed, it] = task2.expected_reward_under_policy(probs)

    ax = axes[1]
    for name, color, _ in configs:
        ret = all_returns[name]
        ax.plot(ret.mean(axis=0), color=color, linewidth=2, label=name)
        ax.fill_between(np.arange(n_iterations),
                          ret.mean(axis=0) - ret.std(axis=0),
                          ret.mean(axis=0) + ret.std(axis=0),
                          color=color, alpha=0.18)
    # Optimal reference.
    opt = task2.expected_reward_under_policy(
        np.eye(task2.n_completions)[task2.greedy_policy()])
    ax.axhline(opt, color="#222", linestyle="--", linewidth=1, alpha=0.5,
                label=f"optimal = {opt:.2f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("expected reward under current policy")
    ax.set_title("Three baseline choices: all reach similar performance here")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.suptitle(
        "GRPO mechanism. Group mean baseline (left) gives advantages. "
        "On this small clean task all three baselines are competitive — "
        "the benefit of GRPO's group baseline scales with reward variance.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02d_grpo_mechanism.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
