"""Reproduce the regret-curves figure from Post 1a.

Run:
    python experiments/01a_regret_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from nano_agents.bandits import (
    BernoulliBandit,
    EpsilonGreedy,
    ThompsonSampling,
    UCB1,
    run_many,
)


def main() -> None:
    T, n_runs = 5000, 100

    def bandit_factory():
        return BernoulliBandit([0.3, 0.5, 0.7, 0.75, 0.8])

    K = 5
    agents = {
        "ε-greedy (0.1)": lambda: EpsilonGreedy(K, eps=0.1),
        "UCB1": lambda: UCB1(K),
        "Thompson": lambda: ThompsonSampling(K),
    }

    results = run_many(agents, bandit_factory, T, n_runs=n_runs, seed=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, regret in results.items():
        ax.plot(regret, label=name, linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative regret")
    ax.set_title("Bernoulli bandit, 5 arms (mean over 100 runs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path("figures/01a_regret_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
