"""Figure: TwoGoalGridWorld layout.

Shows the 5x5 grid with start, walls, two goals (+1 close, +10 far), and
the action space. This is the running example used throughout Post 1c.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld


def main() -> None:
    env = TwoGoalGridWorld(slip=0.0)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Background grid: white cells, dark walls, colored terminals.
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.walls:
                color = "#222"
            elif (r, c) in env.terminals:
                tv = env.terminals[(r, c)]
                color = "#a5d6a7" if tv > 5 else "#fff59d"
            elif (r, c) == env.start:
                color = "#bbdefb"
            else:
                color = "#fafafa"
            ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor=color, edgecolor="#999",
                                            linewidth=1.0))

    # Labels.
    ax.text(env.start[1], env.start[0], "S", ha="center", va="center",
             fontsize=22, fontweight="bold", color="#1565c0")
    for (r, c), tv in env.terminals.items():
        ax.text(c, r, f"+{int(tv)}", ha="center", va="center",
                 fontsize=18, fontweight="bold",
                 color="#2e7d32" if tv > 5 else "#f57f17")

    # Action legend.
    ax.text(2.0, -1.6, "Actions: ↑ Up   → Right   ↓ Down   ← Left",
            ha="center", fontsize=11)
    ax.text(2.0, -1.95,
             f"Step reward {env.step_reward}, slip = {env.slip}",
            ha="center", fontsize=10, color="#666")

    ax.set_xlim(-0.7, env.cols - 0.3)
    ax.set_ylim(env.rows - 0.3, -2.3)  # invert so row 0 is at top
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels(range(env.cols), fontsize=9)
    ax.set_yticklabels(range(env.rows), fontsize=9)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.set_title(
        "TwoGoalGridWorld — the running example.\n"
        "Small goal +1 is close; big goal +10 is far. "
        "Walls block the direct path.",
        fontsize=12,
    )

    fig.tight_layout()
    out = Path("figures/01c_gridworld_setup.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
