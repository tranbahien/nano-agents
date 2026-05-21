"""Solution experiment 1a.1: gap sensitivity.

Run all three algorithms on an EASY bandit ([0.1, 0.5, 0.9]) and a HARD
bandit ([0.50, 0.51, 0.52]) at the same horizon. Reveals how algorithm
quality matters most when gaps are small.
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
    run_many,
)


def main() -> None:
    T, n_runs = 5000, 200
    K = 3

    configs = {
        "Easy gaps: [0.1, 0.5, 0.9]": [0.1, 0.5, 0.9],
        "Hard gaps: [0.50, 0.51, 0.52]": [0.50, 0.51, 0.52],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, (title, probs) in zip(axes, configs.items()):
        def bf(probs=probs):
            return BernoulliBandit(probs)

        agents = {
            "ε-greedy (0.1)": lambda: EpsilonGreedy(K, 0.1),
            "UCB1": lambda: UCB1(K),
            "Thompson": lambda: ThompsonSampling(K),
        }
        results = run_many(agents, bf, T, n_runs=n_runs, seed=0)
        for name, r in results.items():
            ax.plot(r, label=name, linewidth=2)
        ax.set_xlabel("t")
        ax.set_ylabel("cumulative regret")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Gap sensitivity: algorithm quality matters most when arms are hard to tell apart",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out = Path("figures/01a_sol1_gap_sensitivity.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
