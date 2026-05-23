"""Solution 3b.4: ReAct vs answer-immediately.

The whole premise of tool use is that pausing to call a tool beats
answering directly from the model's internal computation. We compare two
agents across a difficulty range:

  - answer-immediately: always solve internally, answer in one shot.
  - ReAct (always-tool): always call the calculator, then answer.
  - ReAct (threshold):   call the tool only when difficulty >= threshold.

The threshold agent is the practical sweet spot: it matches always-tool
accuracy while making far fewer tool calls.
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
    calc = make_calculator()
    threshold = 4.0  # call tool when difficulty >= 4

    digit_levels = list(range(1, 6))
    n_problems = 1500

    acc_immediate = []
    acc_always_tool = []
    acc_threshold = []
    toolcalls_threshold = []
    toolcalls_always = []

    for md in digit_levels:
        rng = np.random.default_rng(md)
        problems = []
        for _ in range(n_problems):
            a = int(rng.integers(10 ** (md - 1), 10 ** md))
            b = int(rng.integers(10 ** (md - 1), 10 ** md))
            op = str(rng.choice(["+", "-", "*"]))
            problems.append(ArithmeticProblem(a, op, b))

        solve_rng = np.random.default_rng(99)
        # answer-immediately
        acc_immediate.append(float(np.mean(
            [model.solve_internally(p, solve_rng)[1] for p in problems])))
        # always-tool
        n_correct = sum(int(calc(p.as_tuple())[0] == p.answer) for p in problems)
        acc_always_tool.append(n_correct / n_problems)
        toolcalls_always.append(1.0)
        # threshold
        n_correct = 0
        n_tool = 0
        for p in problems:
            if p.difficulty >= threshold:
                n_tool += 1
                n_correct += int(calc(p.as_tuple())[0] == p.answer)
            else:
                n_correct += int(model.solve_internally(p, solve_rng)[1])
        acc_threshold.append(n_correct / n_problems)
        toolcalls_threshold.append(n_tool / n_problems)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left: accuracy.
    ax = axes[0]
    ax.plot(digit_levels, acc_immediate, "o-", color="#c44e52", linewidth=2,
            markersize=9, label="answer-immediately (no tools)")
    ax.plot(digit_levels, acc_always_tool, "s-", color="#3a7ebf", linewidth=2,
            markersize=9, label="ReAct (always tool)")
    ax.plot(digit_levels, acc_threshold, "^-", color="#55a467", linewidth=2,
            markersize=9, label=f"ReAct (tool iff difficulty>={threshold:.0f})")
    ax.set_xlabel("operand size (digits)"); ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by difficulty")
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(digit_levels)

    # Right: tool-call rate (cost).
    ax = axes[1]
    width = 0.35
    x = np.arange(len(digit_levels))
    ax.bar(x - width / 2, toolcalls_always, width, color="#3a7ebf",
            label="always tool", edgecolor="white")
    ax.bar(x + width / 2, toolcalls_threshold, width, color="#55a467",
            label="threshold", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(digit_levels)
    ax.set_xlabel("operand size (digits)")
    ax.set_ylabel("tool calls per problem")
    ax.set_title("Tool-call cost — threshold agent calls far less")
    ax.legend(loc="center right"); ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True); ax.set_ylim(0, 1.1)

    fig.suptitle(
        "ReAct beats answer-immediately on hard problems. A difficulty "
        "threshold matches always-tool accuracy at a fraction of the cost.",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path("figures/03b_sol4_react_vs_immediate.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
