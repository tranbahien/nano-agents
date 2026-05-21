"""Figure: Arm pull distribution over time for each algorithm.

Shows the fraction of cumulative pulls allocated to each arm as a function
of time, for ε-greedy, UCB1, and Thompson Sampling. Reveals the qualitative
difference in exploration strategies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.bandits import (
    BernoulliBandit,
    EpsilonGreedy,
    ThompsonSampling,
    UCB1,
)


def run_track_pulls(agent, bandit, T):
    """Run one episode and return cumulative pull counts per arm over time."""
    K = bandit.K
    pulls = np.zeros((T, K))
    counts = np.zeros(K)
    for t in range(T):
        k = agent.select()
        r = bandit.pull(k)
        agent.update(k, r)
        counts[k] += 1
        pulls[t] = counts
    return pulls / pulls.sum(axis=1, keepdims=True)  # normalize to fractions


def average_runs(make_agent, bandit_factory, T, n_runs=30, seed=0):
    np.random.seed(seed)
    K = bandit_factory().K
    avg = np.zeros((T, K))
    for _ in range(n_runs):
        bandit = bandit_factory()
        fracs = run_track_pulls(make_agent(), bandit, T)
        avg += fracs / n_runs
    return avg


def main() -> None:
    true_mus = [0.3, 0.5, 0.7, 0.75, 0.8]
    K = len(true_mus)
    T, n_runs = 5000, 30

    def bandit_factory():
        return BernoulliBandit(true_mus)

    configs = [
        ("ε-greedy (0.1)", lambda: EpsilonGreedy(K, eps=0.1)),
        ("UCB1", lambda: UCB1(K)),
        ("Thompson", lambda: ThompsonSampling(K)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, K))
    t_axis = np.arange(1, T + 1)

    for ax, (name, ctor) in zip(axes, configs):
        fracs = average_runs(ctor, bandit_factory, T, n_runs=n_runs)
        # Stacked area plot
        ax.stackplot(t_axis, fracs.T, colors=colors,
                     labels=[f"Arm {k+1} ($\\mu={true_mus[k]}$)" for k in range(K)],
                     edgecolor="white", linewidth=0.2)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("t")
        ax.set_xlim(1, T)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("fraction of pulls")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)

    fig.suptitle(
        "Cumulative pull fractions per arm. Best arm (top, brightest) is arm 5.\n"
        "ε-greedy keeps exploring forever; UCB1 and Thompson concentrate on the best arm.",
        fontsize=12, y=1.06,
    )
    fig.tight_layout()

    out = Path("figures/01a_arm_pull_distribution.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
