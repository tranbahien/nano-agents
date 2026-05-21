"""Solution experiment 1b.2: posterior calibration.

Tracks the empirical coverage of LinUCB's 95% predictive interval over time
in two scenarios:

  (a) Matched noise: bandit noise = algorithm's implicit assumption (sigma = 1).
      Coverage should converge to ~95%.

  (b) Mismatched noise: bandit noise sigma_true = 0.1, algorithm still
      assumes sigma = 1 internally. The credible interval will be far too
      wide and coverage will exceed 95% (over-coverage = under-confidence).

This second case is a microcosm of LLM calibration: stated uncertainty
that doesn't match true reliability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.contextual import LinearContextualBandit, LinUCB


def run_with_coverage(agent, bandit, T, sigma_alg=1.0):
    coverage = np.zeros(T)
    cum_cov = 0
    for t in range(T):
        x = bandit.sample_context()
        k = agent.select(x)
        A_inv = np.linalg.inv(agent.A[k])
        theta_hat = A_inv @ agent.b[k]
        mu_pred = float(x @ theta_hat)
        param_var = float(x @ A_inv @ x)
        sigma_pred = np.sqrt(max(sigma_alg ** 2 + param_var, 1e-12))

        r = bandit.pull(x, k)
        lo, hi = mu_pred - 1.96 * sigma_pred, mu_pred + 1.96 * sigma_pred
        cum_cov += int(lo <= r <= hi)
        coverage[t] = cum_cov / (t + 1)
        agent.update(x, k, r)
    return coverage


def main() -> None:
    K, d = 5, 6
    T, n_runs = 1500, 20

    scenarios = [
        ("Matched: bandit σ = 1.0, algorithm assumes σ = 1.0", 1.0, 1.0),
        ("Mismatched: bandit σ = 0.1, algorithm assumes σ = 1.0", 0.1, 1.0),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colors = ["#3a7ebf", "#c44e52"]

    for (label, sigma_true, sigma_alg), color in zip(scenarios, colors):
        avg = np.zeros(T)
        for run_idx in range(n_runs):
            bandit = LinearContextualBandit(K=K, d=d, sigma=sigma_true,
                                             seed=run_idx)
            agent = LinUCB(K, d, alpha=1.0)
            avg += run_with_coverage(agent, bandit, T, sigma_alg=sigma_alg) / n_runs
        ax.plot(avg, color=color, linewidth=2, label=label)
        print(f"{label}: final coverage = {avg[-1]:.3f}")

    ax.axhline(0.95, color="#222", linestyle="--", linewidth=1.2,
                label="nominal 95%")
    ax.set_xlabel("t")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"Calibration of LinUCB's 95% predictive interval "
                 f"(K={K}, d={d}, {n_runs} runs)")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = Path("figures/01b_sol2_calibration.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
