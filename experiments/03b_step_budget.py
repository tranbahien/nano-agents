"""Figure: multi-hop success vs the step budget.

A multi-hop task of length L needs at least L tool calls to solve. If the
agent's step budget is below L, it cannot finish the chain. We sweep both
the step budget and the chain length to show the phase transition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.tools import build_example_graph, make_lookup, run_react_oracle


def run_with_budget(kg, task, lookup, max_steps):
    """Run the oracle but cap the number of tool calls at max_steps."""
    from nano_agents.tools import ReActTrace
    trace = ReActTrace()
    cur = task.start
    for r in task.chain:
        if trace.n_tool_calls >= max_steps:
            # Out of budget; forced to answer with whatever we have.
            trace.final_answer = cur
            trace.correct = (cur == task.answer)
            return trace
        result, ok = lookup((cur, r))
        trace.n_tool_calls += 1
        if not ok:
            return trace
        cur = result
    trace.final_answer = cur
    trace.correct = (cur == task.answer)
    return trace


def main() -> None:
    chain_lengths = [2, 4, 6, 8]
    step_budgets = list(range(0, 11))
    n_tasks = 200

    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(chain_lengths)))

    for cl, color in zip(chain_lengths, colors):
        success = np.zeros(len(step_budgets))
        for t in range(n_tasks):
            kg, task = build_example_graph(
                np.random.default_rng(t), chain_len=cl, seed=t)
            lookup = make_lookup(kg)
            for j, budget in enumerate(step_budgets):
                trace = run_with_budget(kg, task, lookup, budget)
                success[j] += int(trace.correct)
        success /= n_tasks
        ax.plot(step_budgets, success, "o-", color=color, linewidth=2,
                markersize=7, label=f"chain length {cl}")
        # Mark the required budget (chain length + 1 for the value hop).
        needed = cl + 1
        ax.axvline(needed, color=color, linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("step budget (max tool calls allowed)")
    ax.set_ylabel("task success rate")
    ax.set_title(
        "Multi-hop success has a sharp threshold at (chain length + 1).\n"
        "Dotted lines mark the minimum budget each chain needs.",
        fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(step_budgets)

    fig.tight_layout()
    out = Path("figures/03b_step_budget.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
