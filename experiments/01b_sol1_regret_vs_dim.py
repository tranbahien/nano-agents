"""Solution experiment 1b.1: regret scaling with dimension.

How does cumulative regret at fixed T scale with feature dimension d?
Theory predicts:
  LinUCB:        regret ~ d * sqrt(T)        (linear in d)
  LinTS:         regret ~ d^{3/2} * sqrt(T)  (super-linear in d)

We sweep d and fit a power law to verify.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.contextual import (
    LinearContextualBandit,
    LinearThompsonSampling,
    LinUCB,
    run_many_contextual,
)


def main() -> None:
    K = 5
    T, n_runs = 600, 8
    dims = [2, 5, 10, 20, 35]

    results = {"LinUCB": [], "Linear Thompson": []}
    for d in dims:
        print(f"  d = {d}...")

        def bf(seed, d=d):
            return LinearContextualBandit(K=K, d=d, sigma=0.1, seed=seed)

        agents = {
            "LinUCB": lambda d=d: LinUCB(K, d, alpha=1.0),
            "Linear Thompson": lambda d=d: LinearThompsonSampling(
                K, d, v=0.5, rng=np.random.default_rng()),
        }
        r = run_many_contextual(agents, bf, T, n_runs=n_runs, seed=0)
        results["LinUCB"].append(r["LinUCB"][-1])
        results["Linear Thompson"].append(r["Linear Thompson"][-1])

    dims = np.array(dims)
    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Fit power-law: regret ~ a * d^b. Linear regression in log-log.
    for name, color, marker in [("LinUCB", "#3a7ebf", "o"),
                                  ("Linear Thompson", "#55a467", "s")]:
        y = np.array(results[name])
        ax.loglog(dims, y, marker, color=color, markersize=9,
                  label=name, markeredgecolor="white", markeredgewidth=1)

        # Fit power law
        log_d = np.log(dims)
        log_y = np.log(y)
        b, log_a = np.polyfit(log_d, log_y, 1)
        a = np.exp(log_a)
        d_grid = np.logspace(np.log10(dims.min()), np.log10(dims.max()), 100)
        ax.loglog(d_grid, a * d_grid ** b, color=color, linestyle="--",
                  linewidth=1.5, alpha=0.6,
                  label=rf"  fit: $\propto d^{{{b:.2f}}}$")

    ax.set_xlabel(r"feature dimension $d$")
    ax.set_ylabel(f"cumulative regret at T = {T}")
    ax.set_title("Regret scaling with dimension (log-log)\n"
                 f"K={K}, mean over {n_runs} runs")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()

    out = Path("figures/01b_sol1_regret_vs_dim.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
