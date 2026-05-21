"""Figure: Belief update under repeated listening in the Tiger POMDP.

Starting from a uniform prior, the agent listens repeatedly. Each
observation updates the belief via Bayes' rule. We show belief trajectories
for three observation sequences: all-GL (consistent with TL), all-GR
(consistent with TR), and a noisy mix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.pomdp import TigerPOMDP, belief_update


def simulate_belief_trajectory(pomdp, observations):
    """Apply belief_update repeatedly with the Listen action."""
    b = np.array([0.5, 0.5])
    trajectory = [b.copy()]
    for o in observations:
        b = belief_update(b, a=2, o=o, P=pomdp.P, Z=pomdp.Z)
        trajectory.append(b.copy())
    return np.array(trajectory)


def main() -> None:
    pomdp = TigerPOMDP(listen_accuracy=0.85)
    T = 12

    # Three scenarios.
    scenarios = [
        ("All GL (tiger really on left)", [0] * T, "#3a7ebf"),
        ("All GR (tiger really on right)", [1] * T, "#dd8452"),
        ("Mixed: GL, GR, GL, GR, ...", [t % 2 for t in range(T)], "#55a467"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for label, obs, color in scenarios:
        traj = simulate_belief_trajectory(pomdp, obs)
        ax.plot(range(T + 1), traj[:, 0], "o-", color=color, linewidth=2,
                 markersize=7, label=label)

    ax.axhline(0.5, color="#222", linestyle=":", linewidth=1, alpha=0.5,
                label="initial prior")
    ax.set_xlabel("listen step")
    ax.set_ylabel(r"$b(\mathrm{TL}) = P(\mathrm{tiger\ left} \mid \mathrm{history})$")
    ax.set_title(
        "Belief evolution under repeated Listen actions.\n"
        f"Each Listen costs −1 and is {int(pomdp.listen_accuracy * 100)}% accurate.",
        fontsize=12,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.3, T + 0.3)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/01d_belief_update.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
