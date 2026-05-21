"""Figure: reward model learning from preferences.

Two-panel visualization:

  Left: the Bradley-Terry sigmoid relating reward differences to preference
        probabilities. Adds annotated examples for small / medium / large
        reward gaps.

  Right: empirical recovery of the true reward function from preference
        data. Plot the learned reward model's predictions vs the true
        rewards, showing convergence as more preferences are seen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.rlhf import PreferenceTask, TabularRewardModel


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # ----- Left: Bradley-Terry sigmoid -----
    ax = axes[0]
    diff = np.linspace(-5, 5, 500)
    p = 1.0 / (1.0 + np.exp(-diff))
    ax.plot(diff, p, color="#3a7ebf", linewidth=2.5)
    ax.fill_between(diff, p, alpha=0.18, color="#3a7ebf")
    ax.axhline(0.5, color="#222", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(0.0, color="#222", linestyle=":", linewidth=1, alpha=0.5)

    # Annotated examples.
    for d in [-2, 0, 2]:
        pp = 1.0 / (1.0 + np.exp(-d))
        ax.scatter([d], [pp], s=90, color="#c44e52", zorder=5,
                    edgecolor="white", linewidth=1.5)
        ax.annotate(f"Δr={d}\nP(y_w ≻ y_l)={pp:.2f}",
                    (d, pp), textcoords="offset points",
                    xytext=(12, -22 if pp > 0.5 else 8),
                    fontsize=10, color="#c44e52")

    ax.set_xlabel(r"reward difference $\Delta r = r(y_w) - r(y_l)$")
    ax.set_ylabel(r"$P(y_w \succ y_l) = \sigma(\Delta r)$")
    ax.set_title("Bradley-Terry preference model")
    ax.set_xlim(-5, 5); ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    # ----- Right: reward model recovery -----
    ax = axes[1]
    task = PreferenceTask(n_contexts=4, n_completions=4, reward_scale=2.0,
                            seed=0)
    sample_sizes = [10, 50, 200, 2000]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(sample_sizes)))

    # Flatten the reward grids.
    true_flat = task.true_rewards.flatten()
    for n, color in zip(sample_sizes, colors):
        rng = np.random.default_rng(0)
        prefs = task.collect_preferences(n_samples=n, rng=rng)
        rm = TabularRewardModel(task.n_contexts, task.n_completions)
        rm.fit(prefs, n_steps=600, lr=0.5)
        # Center the learned reward per context (eliminates the additive
        # ambiguity that makes the predictions hard to read).
        learned = rm.r - rm.r.mean(axis=1, keepdims=True)
        true_centered = task.true_rewards - task.true_rewards.mean(axis=1, keepdims=True)
        ax.scatter(true_centered.flatten(), learned.flatten(),
                    color=color, s=80, alpha=0.75, label=f"{n} preferences",
                    edgecolor="white", linewidth=0.8)

    # Identity line.
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#222", linestyle="--",
             linewidth=1.2, alpha=0.6, label="perfect recovery")
    ax.set_xlabel("true reward (centered per context)")
    ax.set_ylabel("learned reward (centered per context)")
    ax.set_title("Reward model recovery with more data")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "Preferences encode a sigmoid of reward differences (left). "
        "With enough preference data, the reward model converges to the "
        "true reward (right).",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02d_bradley_terry.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
