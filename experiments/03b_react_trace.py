"""Figure: an annotated ReAct trace on a multi-hop task.

Visualizes one episode: the agent follows a chain of relations through the
knowledge graph, alternating think / act / observe steps until it can
answer. This is the "belief building up one observation at a time" picture
from the POMDP framing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from nano_agents.tools import build_example_graph, make_lookup, run_react_oracle


def main() -> None:
    kg, task = build_example_graph(np.random.default_rng(0), chain_len=3, seed=3)
    lookup = make_lookup(kg)
    trace = run_react_oracle(kg, task, lookup)

    # Render the trace as a vertical sequence of labeled boxes.
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(trace.steps) + 1)
    ax.axis("off")

    kind_color = {
        "think":   "#fff2cc",
        "act":     "#fde2e2",
        "observe": "#e8f0fe",
        "answer":  "#e0f0e0",
    }
    kind_label = {
        "think":   "THINK",
        "act":     "ACT",
        "observe": "OBSERVE",
        "answer":  "ANSWER",
    }

    n = len(trace.steps)
    for i, (kind, content) in enumerate(trace.steps):
        y = n - i
        # Format content.
        if kind == "act":
            tool, args = content
            text = f"call {tool}{args}"
        elif kind == "observe":
            text = f"-> {content}"
        elif kind == "answer":
            text = f"answer = {content}"
        else:
            text = str(content)

        rect = patches.FancyBboxPatch(
            (1.8, y - 0.38), 7.6, 0.72,
            boxstyle="round,pad=0.02", facecolor=kind_color[kind],
            edgecolor="#222", linewidth=1.4, zorder=3)
        ax.add_patch(rect)
        # Kind tag.
        ax.text(1.0, y, kind_label[kind], ha="right", va="center",
                fontsize=10, fontweight="bold", color="#444")
        ax.text(2.1, y, text, ha="left", va="center", fontsize=10,
                color="#222", family="monospace")
        # Arrow to next.
        if i < n - 1:
            ax.annotate("", xy=(5.6, y - 0.42), xytext=(5.6, y - 0.62),
                         arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))

    result_txt = ("CORRECT" if trace.correct else "WRONG")
    result_color = "#55a467" if trace.correct else "#c44e52"
    ax.text(5.6, 0.3,
            f"Final: {trace.final_answer}  ({result_txt}, "
            f"true answer = {task.answer}, {trace.n_tool_calls} tool calls)",
            ha="center", fontsize=11, fontweight="bold", color=result_color)

    ax.set_title(
        f"A ReAct episode: {len(task.chain)}-hop lookup from {task.start}.\n"
        "Each tool result is an observation; the agent answers once the chain "
        "is complete.",
        fontsize=12)
    fig.tight_layout()
    out = Path("figures/03b_react_trace.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
