"""Solution 3b.3: robustness to unreliable tools.

A tool isn't always right — APIs return stale data, calculators have bugs,
search returns irrelevant results. We model a calculator with an error_rate
and ask: how does tool unreliability change the value of tool use?

Key question: at what tool error rate does using the tool stop being worth
it (vs computing internally)? This depends on problem difficulty: for hard
problems where internal accuracy is near zero, even a flaky tool helps; for
easy problems, a flaky tool can be *worse* than internal computation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.tools import (
    ArithmeticProblem,
    NoisyArithmeticModel,
    make_calculator,
)


def main() -> None:
    model = NoisyArithmeticModel(base_accuracy=0.99, decay=0.45, d0=2.0)
    tool_error_rates = np.linspace(0, 0.5, 11)
    # Three difficulty regimes (by operand digit count).
    digit_levels = [(1, "easy (1-digit)"), (3, "medium (3-digit)"),
                    (5, "hard (5-digit)")]
    n_problems = 2000

    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#55a467", "#dd8452", "#c44e52"]

    # The tool's accuracy depends only on its error rate, not difficulty,
    # so we compute it once.
    tool_acc = [1.0 - err for err in tool_error_rates]
    ax.plot(tool_error_rates, tool_acc, "o-", color="#3a7ebf", linewidth=2.5,
            markersize=8, label="calculator tool (any difficulty)", zorder=5)

    for (md, label), color in zip(digit_levels, colors):
        rng = np.random.default_rng(0)
        problems = []
        for _ in range(n_problems):
            a = int(rng.integers(10 ** (md - 1), 10 ** md))
            b = int(rng.integers(10 ** (md - 1), 10 ** md))
            op = str(rng.choice(["+", "-", "*"]))
            problems.append(ArithmeticProblem(a, op, b))
        solve_rng = np.random.default_rng(1)
        internal_acc = float(np.mean(
            [model.solve_internally(p, solve_rng)[1] for p in problems]))
        ax.axhline(internal_acc, color=color, linestyle="--", linewidth=2,
                    alpha=0.7, label=f"internal: {label}")
        # Mark the crossover where tool error rate makes tool == internal.
        crossover = 1.0 - internal_acc
        if 0 <= crossover <= tool_error_rates[-1]:
            ax.scatter([crossover], [internal_acc], s=140, color=color,
                        marker="X", zorder=6, edgecolor="white", linewidth=1.5)

    ax.set_xlabel("tool error rate")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "Tool use vs internal computation under unreliable tools.\n"
        "X marks where the tool's error rate makes it no better than internal.",
        fontsize=11)
    ax.legend(loc="center left", fontsize=9); ax.grid(alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 0.55)

    fig.tight_layout()
    out = Path("figures/03b_sol3_tool_errors.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
