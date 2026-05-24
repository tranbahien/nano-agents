"""Figure: the adaptive controller beats every fixed single strategy.

A fixed strategy applies one capability to every question; the nanoAgent
chooses per question by value of information. Because each capability only
helps its home question type, no fixed strategy can win across the board --
but the adaptive controller routes each question to the right tool and beats
them all. This is the value of orchestration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.agent import NanoAgent, SimulatedWorld, run_agent


def fixed_accuracy(world, strategy, seed=1):
    """Accuracy of a strategy that applies one capability to every question."""
    rng = np.random.default_rng(seed)
    correct = 0
    for qid in range(world.n_questions):
        ans, _ = world.generate(qid, rng)
        if strategy == "tool":
            ans = world.use_tool(qid, rng)
        elif strategy == "retrieve":
            ans = world.retrieve(qid, rng)
        elif strategy == "reflect":
            _, ans = world.reflect(qid, ans, rng)
        elif strategy == "vote":
            ans = world.vote(qid, 5, rng)
        correct += world.is_correct(qid, ans)
    return correct / world.n_questions


def main() -> None:
    world = SimulatedWorld(n_questions=6000, overconfidence=1.0, seed=0)
    fixed = {s: fixed_accuracy(world, s) for s in
             ["answer", "tool", "retrieve", "reflect", "vote"]}
    adaptive, cost, _ = run_agent(world, NanoAgent(calibration_T=1.0), seed=1)

    labels = ["answer\nonly", "always\ntool", "always\nretrieve",
              "always\nreflect", "always\nvote", "nanoAgent\n(adaptive)"]
    vals = [fixed["answer"], fixed["tool"], fixed["retrieve"],
            fixed["reflect"], fixed["vote"], adaptive]
    colors = ["#888888", "#dd8452", "#3a7ebf", "#c44e52", "#9467bd", "#55a467"]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=10)

    ax.set_ylabel("accuracy")
    ax.set_title(
        "The adaptive controller beats every fixed strategy.\n"
        "Each capability helps only its home question type; routing wins.",
        fontsize=11)
    ax.set_ylim(0, 0.92); ax.grid(alpha=0.3, axis="y"); ax.set_axisbelow(True)

    print(f"  fixed={ {k: round(v,3) for k,v in fixed.items()} }; adaptive={adaptive:.3f}")
    fig.tight_layout()
    out = Path("figures/05b_agent_vs_fixed.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
