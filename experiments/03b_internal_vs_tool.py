"""Figure: internal computation accuracy vs tool-augmented accuracy.

The motivating picture for tool use. Internal arithmetic degrades sharply
with problem difficulty; calling a calculator keeps accuracy flat at ~100%
regardless of difficulty.
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
    rng = np.random.default_rng(0)

    # Group problems by difficulty bucket; measure empirical accuracy.
    n_per_bucket = 500
    max_digits_range = [1, 2, 3, 4, 5]
    diff_buckets = {}
    internal_acc = {}
    tool_acc = {}

    for md in max_digits_range:
        problems = []
        # Sample problems with operands of exactly md digits for clean buckets.
        for _ in range(n_per_bucket):
            a = int(rng.integers(10 ** (md - 1), 10 ** md))
            b = int(rng.integers(10 ** (md - 1), 10 ** md))
            op = str(rng.choice(["+", "-", "*"]))
            from nano_agents.tools import ArithmeticProblem
            problems.append(ArithmeticProblem(a, op, b))
        avg_diff = float(np.mean([p.difficulty for p in problems]))
        diff_buckets[md] = avg_diff
        internal_acc[md] = float(np.mean(
            [model.solve_internally(p, rng)[1] for p in problems]))
        # Tool: always correct (exact calculator).
        tool_correct = 0
        for p in problems:
            result, ok = calc(p.as_tuple())
            tool_correct += int(ok and result == p.answer)
        tool_acc[md] = tool_correct / n_per_bucket

    diffs = [diff_buckets[md] for md in max_digits_range]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    # Left: accuracy vs operand digit count.
    ax = axes[0]
    ax.plot(max_digits_range, [internal_acc[md] for md in max_digits_range],
            "o-", color="#c44e52", linewidth=2, markersize=10,
            label="internal computation")
    ax.plot(max_digits_range, [tool_acc[md] for md in max_digits_range],
            "s-", color="#3a7ebf", linewidth=2, markersize=10,
            label="calculator tool")
    ax.set_xlabel("operand size (number of digits)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs problem size")
    ax.set_xticks(max_digits_range)
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)

    # Right: the model's internal-accuracy function (theoretical curve).
    ax = axes[1]
    d_grid = np.linspace(0, 16, 200)
    p_grid = [model.internal_accuracy(d) for d in d_grid]
    ax.plot(d_grid, p_grid, color="#c44e52", linewidth=2.5,
            label="P(correct internally)")
    ax.axhline(1.0, color="#3a7ebf", linewidth=2, linestyle="--",
                label="P(correct with tool)")
    ax.fill_between(d_grid, p_grid, 1.0, color="#3a7ebf", alpha=0.10,
                     label="value of the tool")
    ax.set_xlabel("problem difficulty")
    ax.set_ylabel("P(correct)")
    ax.set_title("The internal-accuracy model and the tool's value")
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.08)

    fig.suptitle(
        "Internal arithmetic degrades with difficulty; a calculator stays exact. "
        "The gap is the value of tool access.",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path("figures/03b_internal_vs_tool.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
