"""Figure: the cost-accuracy tradeoff of a threshold tool-use policy.

Tool calls aren't free — they cost latency, tokens, money. A sensible
policy calls the tool only when the problem is hard enough that the
accuracy gain outweighs the cost. We model a threshold policy: "use the
tool iff difficulty >= threshold" and sweep the threshold.

  - Low threshold  -> use tools almost always -> high accuracy, high cost.
  - High threshold -> rarely use tools -> low cost, low accuracy.

The net utility (accuracy minus cost penalty) is maximized at an
intermediate threshold — the agent should call tools selectively.
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

    n_problems = 4000
    rng = np.random.default_rng(0)
    problems = [sample_problem(rng, max_digits=4) for _ in range(n_problems)]
    difficulties = np.array([p.difficulty for p in problems])

    thresholds = np.linspace(0, difficulties.max() + 1, 30)
    accuracy = np.zeros(len(thresholds))
    tool_rate = np.zeros(len(thresholds))

    solve_rng = np.random.default_rng(1)
    for i, thr in enumerate(thresholds):
        n_correct = 0
        n_tool = 0
        for p in problems:
            if p.difficulty >= thr:
                # Use the tool.
                result, ok = calc(p.as_tuple())
                n_tool += 1
                n_correct += int(ok and result == p.answer)
            else:
                # Compute internally.
                _, correct = model.solve_internally(p, solve_rng)
                n_correct += int(correct)
        accuracy[i] = n_correct / n_problems
        tool_rate[i] = n_tool / n_problems

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left: accuracy and tool-usage rate vs threshold.
    ax = axes[0]
    ax.plot(thresholds, accuracy, "o-", color="#3a7ebf", linewidth=2,
            markersize=5, label="accuracy")
    ax.plot(thresholds, tool_rate, "s-", color="#dd8452", linewidth=2,
            markersize=5, label="fraction of problems using the tool")
    ax.set_xlabel("tool-use threshold (call tool iff difficulty >= threshold)")
    ax.set_ylabel("rate")
    ax.set_title("Accuracy and tool-usage vs threshold")
    ax.legend(loc="center right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)

    # Right: net utility = accuracy - lambda * tool_rate, for a few lambdas.
    ax = axes[1]
    for lam, color in zip([0.1, 0.3, 0.6], ["#55a467", "#3a7ebf", "#c44e52"]):
        utility = accuracy - lam * tool_rate
        best_idx = int(np.argmax(utility))
        ax.plot(thresholds, utility, "-", color=color, linewidth=2,
                label=f"cost $\\lambda$ = {lam}")
        ax.scatter([thresholds[best_idx]], [utility[best_idx]], s=120,
                   color=color, zorder=5, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("tool-use threshold")
    ax.set_ylabel("net utility = accuracy $-\\ \\lambda \\cdot$ tool-rate")
    ax.set_title("Optimal threshold shifts with the cost of a tool call")
    ax.legend(loc="lower center"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.suptitle(
        "When tool calls cost something, the optimal policy uses them "
        "selectively — only when a problem is hard enough to be worth it.",
        fontsize=12, y=1.02)
    fig.tight_layout()
    out = Path("figures/03b_cost_accuracy.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
