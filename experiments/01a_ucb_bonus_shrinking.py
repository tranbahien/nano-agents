"""Figure: UCB1 confidence bonuses shrinking over time.

Visualizes how the exploration bonus shrinks as arms are pulled more,
illustrating the mechanism behind "optimism in the face of uncertainty".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.bandits import BernoulliBandit, UCB1


# Update matplotlib configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"], 
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True 
})



def run_and_snapshot(T_targets, true_mus, seed=0):
    """Run UCB1 and snapshot the state at each T_target."""
    np.random.seed(seed)
    bandit = BernoulliBandit(true_mus)
    agent = UCB1(bandit.K)
    snaps = {}
    T_max = max(T_targets)
    for t in range(1, T_max + 1):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        if t in T_targets:
            bonus = np.sqrt(2.0 * np.log(agent.t) / np.maximum(agent.counts, 1))
            snaps[t] = (agent.counts.copy(), agent.means.copy(), bonus.copy())
    return snaps


def main() -> None:
    true_mus = [0.3, 0.5, 0.7, 0.75, 0.8]
    K = len(true_mus)
    T_targets = [50, 500, 5000]

    snaps = run_and_snapshot(T_targets, true_mus, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    arms = np.arange(K)

    for ax, T in zip(axes, T_targets):
        counts, means, bonus = snaps[T]
        # Empirical mean (solid bar) + bonus (lighter on top)
        ax.bar(arms, means, color="#3a7ebf", label=r"$\hat\mu_k$ (exploit)", edgecolor="white")
        ax.bar(arms, bonus, bottom=means, color="#f4a261",
               label=r"$\sqrt{2\log t / n_k}$ (explore)", edgecolor="white")

        # True means as horizontal markers
        for k in range(K):
            ax.hlines(true_mus[k], k - 0.4, k + 0.4,
                      colors="#222", linestyles="--", linewidth=1.5)

        # Pull counts as small text above each bar
        for k in range(K):
            total = means[k] + bonus[k]
            ax.text(k, total + 0.04, f"$n_k\\!=\\!{int(counts[k])}$",
                    ha="center", va="bottom", fontsize=8.5, color="#555")

        ax.set_xticks(arms)
        ax.set_xticklabels([f"Arm {k+1}" for k in range(K)])
        ax.set_title(f"t = {T}", fontsize=12)
        ax.set_ylim(0, 1.75)
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("UCB1 index")
    axes[0].hlines(0, 0, 0, color="#222", linestyles="--", linewidth=1.5,
                   label=r"true $\mu_k$")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "UCB1 in action: the exploration bonus shrinks as $n_k$ grows.\n"
        "By t=5000, the bonuses are nearly zero and decisions track empirical means.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out = Path("figures/01a_ucb_bonus_shrinking.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
