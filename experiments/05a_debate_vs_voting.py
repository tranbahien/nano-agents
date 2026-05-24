"""Figure: debate vs independent voting.

Debate -- letting agents see and revise toward each other -- is reflection
(Post 4b) with peer critics instead of self-critique. We compare the final
accuracy of multi-round debate, as conformity grows, against the one-shot
independent majority vote (round 0). Light conformity with genuine
reconsideration can match or slightly beat voting; heavy conformity *herds*
the committee onto whatever the early plurality was, collapsing the diversity
that made voting work and dragging accuracy back down.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.multiagent import Committee, debate
from nano_agents.multiagent.committee import committee_accuracy


def main() -> None:
    p, n = 0.58, 9
    conformities = np.linspace(0.0, 1.0, 11)
    committee = Committee(n_agents=n, n_answers=4, accuracy=p, correlation=0.0,
                          seed=0)

    one_shot = committee_accuracy(committee, n_items=4000, seed=1)
    finals = []
    for k in conformities:
        rng = np.random.default_rng(1)
        accs = debate(committee, 4000, rng, rounds=5, conformity=float(k))
        finals.append(accs[-1])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(conformities, finals, "o-", color="#c44e52", linewidth=2,
            markersize=6, label="debate (final round)")
    ax.axhline(one_shot, color="#55a467", linestyle="--", linewidth=2,
               label=f"one-shot independent vote ({one_shot:.2f})")
    ax.axhline(p, color="#888", linestyle=":", linewidth=1.5,
               label=f"single agent ({p})")

    ax.set_xlabel("conformity (probability an agent adopts the plurality each round)")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "Debate vs voting (9 agents, p=0.58). Moderate conformity + reconsideration\n"
        "beats one-shot voting; heavy conformity herds and the gain evaporates.",
        fontsize=11)
    ax.legend(loc="lower left"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    print(f"  one-shot={one_shot:.3f}; debate finals={np.round(finals,3)}")
    fig.tight_layout()
    out = Path("figures/05a_debate_vs_voting.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
