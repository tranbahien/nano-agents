"""Figure: the controller's routing emerges from value of information.

No rule says "use the tool on calculation." The controller computes the
expected gain of each capability and picks the best -- and that alone routes
each question type to its home capability. We show the first-action
distribution per question type as a heatmap.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.agent import NanoAgent, SimulatedWorld, run_agent

ACTIONS = ["answer", "tool", "retrieve", "reflect", "vote"]
TYPES = ["knowledge", "calculation", "reasoning"]


def main() -> None:
    world = SimulatedWorld(n_questions=6000, overconfidence=1.0, seed=0)
    _, _, traces = run_agent(world, NanoAgent(calibration_T=1.0), seed=1)

    mat = np.zeros((len(TYPES), len(ACTIONS)))
    counts = {t: Counter() for t in TYPES}
    for qid, tr in enumerate(traces):
        counts[world.types[qid]][tr.actions[0] if tr.actions else "answer"] += 1
    for i, t in enumerate(TYPES):
        total = sum(counts[t].values())
        for j, a in enumerate(ACTIONS):
            mat[i, j] = counts[t][a] / total

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ACTIONS))); ax.set_xticklabels(ACTIONS)
    ax.set_yticks(range(len(TYPES))); ax.set_yticklabels(TYPES)
    for i in range(len(TYPES)):
        for j in range(len(ACTIONS)):
            if mat[i, j] > 0.01:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.5 else "#222", fontsize=10)
    ax.set_xlabel("first action chosen"); ax.set_ylabel("question type")
    ax.set_title(
        "Routing emerges from value of information (no hard-coded rules).\n"
        "Each question type is sent to its home capability.", fontsize=11)
    fig.colorbar(im, ax=ax, label="fraction of questions")

    fig.tight_layout()
    out = Path("figures/05b_routing.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
