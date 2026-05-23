"""Solution 3b.1: the optimal tool-use threshold.

When should the agent call the tool? Model the per-problem expected utility:

  - Compute internally: utility = P(correct | difficulty) - 0      (no tool cost)
  - Use the tool:       utility = 1                      - lambda   (tool cost)

Call the tool iff (1 - lambda) > P(correct | difficulty), i.e. iff
P(correct internally) < 1 - lambda. Since internal accuracy decreases with
difficulty, this is a threshold rule on difficulty.

We verify the analytical threshold against an empirical sweep.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.tools import (
    NoisyArithmeticModel,
    make_calculator,
    sample_problem,
)


def main() -> None:
    model = NoisyArithmeticModel(base_accuracy=0.99, decay=0.45, d0=2.0)
    calc = make_calculator()

    n_problems = 6000
    rng = np.random.default_rng(0)
    problems = [sample_problem(rng, max_digits=5) for _ in range(n_problems)]

    lambdas = np.linspace(0.0, 0.9, 25)
    empirical_best_thr = np.zeros(len(lambdas))
    analytical_thr = np.zeros(len(lambdas))

    # Candidate thresholds to search over.
    cand_thr = np.linspace(0, 22, 60)
    solve_rng = np.random.default_rng(1)

    for li, lam in enumerate(lambdas):
        # Empirical: for each threshold, compute mean utility, pick best.
        best_u = -np.inf
        best_t = 0
        for thr in cand_thr:
            total_u = 0.0
            for p in problems:
                if p.difficulty >= thr:
                    result, ok = calc(p.as_tuple())
                    total_u += (1.0 if (ok and result == p.answer) else 0.0) - lam
                else:
                    _, correct = model.solve_internally(p, solve_rng)
                    total_u += 1.0 if correct else 0.0
            u = total_u / n_problems
            if u > best_u:
                best_u = u
                best_t = thr
        empirical_best_thr[li] = best_t

        # Analytical: threshold where internal accuracy = 1 - lambda.
        # internal_accuracy(d) = base * exp(-decay * max(d - d0, 0)) = 1 - lam
        # Solve for d.
        target = 1.0 - lam
        if target >= model.base_accuracy:
            analytical_thr[li] = 0.0  # always compute internally
        elif target <= 0:
            analytical_thr[li] = 0.0  # always use tool (threshold at 0)
        else:
            # base * exp(-decay (d - d0)) = target
            d = model.d0 - np.log(target / model.base_accuracy) / model.decay
            analytical_thr[li] = max(d, 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(lambdas, empirical_best_thr, "o", color="#3a7ebf",
            markersize=8, label="empirical optimum (utility sweep)")
    ax.plot(lambdas, analytical_thr, "-", color="#c44e52", linewidth=2.5,
            label=r"analytical: $P(\mathrm{correct}) = 1 - \lambda$")
    ax.set_xlabel(r"tool cost $\lambda$")
    ax.set_ylabel("optimal tool-use threshold (difficulty)")
    ax.set_title(
        "Optimal tool-use threshold: call the tool when internal accuracy\n"
        "drops below 1 - (tool cost). Empirical matches the analytical rule.",
        fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/03b_sol1_optimal_threshold.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
