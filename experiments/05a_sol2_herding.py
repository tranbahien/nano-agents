"""Solution 5a.2: herding -- when debate destroys the wisdom of the crowd.

We track majority-vote accuracy across debate rounds for three regimes of a
9-agent committee (p=0.6):

  - moderate conformity, no stubborn agents: debate accumulates evidence and
    accuracy climbs above the one-shot vote;
  - high conformity, no stubborn agents: the committee herds onto the early
    plurality and freezes near the one-shot value;
  - high conformity WITH a stubborn faction: the loud, unmoving minority drags
    the consensus toward its (only p-accurate) opinions, eroding the crowd's
    advantage.

Conformity and confident factions are how a committee throws away the
independence that made it valuable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.multiagent import Committee, debate


def main() -> None:
    p, n, rounds = 0.6, 9, 6
    committee = Committee(n_agents=n, n_answers=4, accuracy=p, correlation=0.0,
                          seed=0)
    regimes = [
        (0.4, 0.0, "#55a467", "moderate conformity, no stubborn"),
        (0.9, 0.0, "#3a7ebf", "high conformity, no stubborn"),
        (0.9, 0.44, "#c44e52", "high conformity + stubborn faction (4/9)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    xs = np.arange(rounds + 1)
    for conf, stub, color, label in regimes:
        rng = np.random.default_rng(1)
        accs = debate(committee, 4000, rng, rounds=rounds, conformity=conf,
                      stubborn_frac=stub)
        ax.plot(xs, accs, "o-", color=color, linewidth=2, markersize=6,
                label=label)
    ax.axhline(p, color="#888", linestyle=":", linewidth=1.5,
               label=f"single agent ({p})")

    ax.set_xlabel("debate round (0 = initial independent vote)")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title(
        "Herding in debate (9 agents, p=0.6). Moderate conformity helps;\n"
        "heavy conformity and a stubborn faction erode the crowd's advantage.",
        fontsize=11)
    ax.legend(loc="lower left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.set_xticks(xs)

    fig.tight_layout()
    out = Path("figures/05a_sol2_herding.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
