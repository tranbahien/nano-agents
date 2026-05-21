"""Solution experiment 1c.2: stochastic transitions.

How does slip probability change the optimal policy? Replace the
TwoGoalGridWorld's perimeter with a 'cliff' variant: the row just above
the +10 has a -50 penalty if you slip into it.

With slip=0, the agent walks confidently next to the cliff. With slip>0,
it backs away.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from nano_agents.mdp import GridWorld, value_iteration
from nano_agents.mdp.visualization import plot_policy


def make_cliff_world(slip: float):
    """4x6 world with cliff along the bottom edge between start and goal."""
    # Layout:
    #   . . . . . .
    #   . . . . . .
    #   S . . . . .
    #   . X X X X G
    # Cells along row 3 cols 1..4 are cliffs (-50, terminal).
    # Goal G at (3, 5) gives +10.
    # Start S at (2, 0).
    terminals = {(3, c): -50.0 for c in range(1, 5)}
    terminals[(3, 5)] = 10.0
    env = GridWorld(rows=4, cols=6, terminals=terminals, walls=None,
                     step_reward=-0.1, slip=slip)
    return env


def main() -> None:
    slips = [0.0, 0.1, 0.3]
    gamma = 0.95

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, slip in zip(axes, slips):
        env = make_cliff_world(slip)
        V, pi, _ = value_iteration(env, gamma=gamma)
        plot_policy(env, pi, V=V, ax=ax)
        start_idx = env.state_to_idx[(2, 0)]
        ax.set_title(rf"slip = {slip}    $V^\star$(start) = {V[start_idx]:.2f}",
                      fontsize=11)

    fig.suptitle(
        f"Cliff walking (γ = {gamma}). Cliff cells (red, value −50) line the row "
        "below the agent. With deterministic transitions the optimal path hugs the cliff. "
        "Slippery transitions push the policy onto safer routes.",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    out = Path("figures/01c_sol2_stochastic.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
