"""Solution experiment 1b.3: model misspecification.

The true reward function is non-linear (includes sin(omega_k x_1)), but
the algorithm assumes linearity. Regret will not approach zero asymptotically:
the linear approximation has a non-zero "approximation error" floor.

We compare:
  - LinUCB on the non-linear problem (misspecified)
  - LinUCB on a linear approximation of the same problem (oracle baseline)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.contextual import (
    LinearContextualBandit,
    LinUCB,
    NonlinearContextualBandit,
    run_contextual,
)


def main() -> None:
    K, d = 4, 4
    T, n_runs = 3000, 25

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Run LinUCB on the linear bandit (well-specified).
    linear_regrets = np.zeros(T)
    for run_idx in range(n_runs):
        bandit = LinearContextualBandit(K=K, d=d, sigma=0.1, seed=run_idx)
        linear_regrets += run_contextual(LinUCB(K, d, alpha=1.0), bandit, T) / n_runs

    # Run LinUCB on the non-linear bandit (misspecified).
    nonlinear_regrets = np.zeros(T)
    for run_idx in range(n_runs):
        bandit = NonlinearContextualBandit(K=K, d=d, sigma=0.1, seed=run_idx)
        nonlinear_regrets += run_contextual(LinUCB(K, d, alpha=1.0), bandit, T) / n_runs

    ax.plot(linear_regrets, color="#3a7ebf", linewidth=2,
            label="LinUCB on linear bandit (well-specified)")
    ax.plot(nonlinear_regrets, color="#c44e52", linewidth=2,
            label=r"LinUCB on $\sin(\omega_k x_1) + \theta_k^\top x_{[1:]}$ (misspecified)")

    # Fit a linear function to the misspecified regret (last 1500 steps)
    t_late = np.arange(T // 2, T)
    slope, intercept = np.polyfit(t_late, nonlinear_regrets[T // 2:], 1)
    ax.plot(t_late, slope * t_late + intercept, color="#c44e52",
            linestyle="--", alpha=0.5, linewidth=1.2,
            label=f"linear fit (slope = {slope:.3f}/step)")

    ax.set_xlabel("t")
    ax.set_ylabel("cumulative regret")
    ax.set_title(f"Misspecification: regret becomes linear when the linear model is wrong\n"
                 f"K={K}, d={d}, mean over {n_runs} runs")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path("figures/01b_sol3_misspecification.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Asymptotic per-step regret (misspecified): {slope:.4f}")


if __name__ == "__main__":
    main()
