"""Figure: Cumulative regret of contextual bandit algorithms.

Compares LinUCB, Linear Thompson Sampling, contextual ε-greedy, and a
context-free UCB1 baseline on a synthetic linear contextual bandit.
The gap between context-free UCB1 and the others quantifies the value of
exploiting context.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.contextual import (
    ContextFreeUCB,
    ContextualEpsilonGreedy,
    LinearContextualBandit,
    LinearThompsonSampling,
    LinUCB,
    run_many_contextual,
)


def main() -> None:
    K, d = 5, 8
    T, n_runs = 2000, 25

    def bandit_factory(seed):
        return LinearContextualBandit(K=K, d=d, sigma=0.1, seed=seed)

    agents = {
        "Context-free UCB1": lambda: ContextFreeUCB(K, d),
        "Contextual ε-greedy (0.1)":
            lambda: ContextualEpsilonGreedy(K, d, eps=0.1,
                                            rng=np.random.default_rng()),
        "LinUCB (α=1.0)": lambda: LinUCB(K, d, alpha=1.0),
        "Linear Thompson": lambda: LinearThompsonSampling(
            K, d, v=0.5, rng=np.random.default_rng()),
    }

    results = run_many_contextual(agents, bandit_factory, T,
                                   n_runs=n_runs, seed=0)

    colors = {
        "Context-free UCB1": "#888888",
        "Contextual ε-greedy (0.1)": "#dd8452",
        "LinUCB (α=1.0)": "#3a7ebf",
        "Linear Thompson": "#55a467",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, regret in results.items():
        ax.plot(regret, label=name, linewidth=2, color=colors[name])
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative regret")
    ax.set_title(f"Linear contextual bandit, K={K} arms, d={d} features "
                 f"(mean over {n_runs} runs)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path("figures/01b_contextual_regret_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
