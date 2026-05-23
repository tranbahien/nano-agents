"""Solution 4b.2: reflection vs best-of-N vs self-consistency at equal budget.

Given a budget of model calls, is it better to spend them sequentially
(reflection) or in parallel (best-of-N, self-consistency)? We sweep the
budget with a moderately noisy self-critic. Reflection spends its budget
adaptively — stopping early on accepted answers, revising the rest — which
makes it efficient when the critic is decent. Self-consistency, which
*averages* over independent samples, is the most robust to critic noise
because it doesn't rely on the critic at all. Best-of-N sits between,
limited by the same noisy verifier reflection uses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.reflection import (
    ReflectiveQA,
    accuracy,
    best_of_n,
    reflect,
    self_consistency,
    single_shot,
)


def main() -> None:
    budgets = [1, 2, 3, 4, 6, 8]
    a = 0.4
    env = ReflectiveQA(n_problems=1500, n_answers=5, gen_accuracy=a,
                       detect_rate=0.7, false_alarm=0.25, revise_gain=0.0, seed=0)

    refl, bon, sc = [], [], []
    for n in budgets:
        refl.append(accuracy(env, reflect, n_trials=5, seed=1, max_rounds=n - 1))
        bon.append(accuracy(env, best_of_n, n_trials=5, seed=1, n=n))
        sc.append(accuracy(env, self_consistency, n_trials=5, seed=1, n=n))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(budgets, refl, "o-", color="#c44e52", linewidth=2, markersize=7,
            label="reflection (sequential)")
    ax.plot(budgets, bon, "s-", color="#dd8452", linewidth=2, markersize=7,
            label="best-of-N (parallel + critic)")
    ax.plot(budgets, sc, "^-", color="#3a7ebf", linewidth=2, markersize=7,
            label="self-consistency (parallel + vote)")
    ax.axhline(a, color="#888", linestyle="--", linewidth=1.5,
               label=f"single-shot ({a})")

    ax.set_xlabel("budget (max model calls)")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "Spending a fixed call budget: sequential vs parallel (critic d=0.70, f=0.25).\n"
        "Reflection is efficient with a decent critic; voting is most robust to critic noise.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(budgets)

    print("  budget refl bon sc")
    for i, n in enumerate(budgets):
        print(f"  {n}: {refl[i]:.2f} {bon[i]:.2f} {sc[i]:.2f}")
    fig.tight_layout()
    out = Path("figures/04b_sol2_reflect_vs_bon.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
