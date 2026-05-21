"""Solution experiment 1c.3: reward shaping.

Three reward designs for the same task (start at (0,0), goal at (4,4)):
  (a) +10 at goal, 0 elsewhere — sparse.
  (b) +10 at goal, -0.04 per step — small step penalty.
  (c) +10 at goal, distance-based shaping: r(s,s') = step_cost + (phi(s') - phi(s))
      with phi(s) = -manhattan_distance(s, goal). This is a 'potential-based'
      shaping known to preserve the optimal policy (Ng, Harada, Russell 1999).

For finite MDPs all three are well-defined; we look at the optimal policy and
the value-of-start under each. The point is that shaping changes V* magnitudes
but does NOT change the optimal policy when shaping is potential-based.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import GridWorld, value_iteration
from nano_agents.mdp.visualization import plot_policy


# We use a custom subclass to inject potential-based shaping at the
# step_outcomes level.
class ShapedGridWorld(GridWorld):
    def __init__(self, *args, goal, gamma_for_shaping=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal = goal
        self.gamma_for_shaping = float(gamma_for_shaping)

    def potential(self, s):
        if s in self.walls:
            return 0.0
        return -float(abs(s[0] - self.goal[0]) + abs(s[1] - self.goal[1]))

    def step_outcomes(self, s, a):
        outcomes = super().step_outcomes(s, a)
        if self.is_terminal(s):
            return outcomes
        # Add potential difference: phi(s') - gamma * phi(s).
        # We use gamma=gamma_for_shaping for the shaping rule.
        new_outcomes = []
        for ns, prob, r in outcomes:
            shaped = r + self.potential(ns) - self.gamma_for_shaping * self.potential(s)
            new_outcomes.append((ns, prob, shaped))
        return new_outcomes


def make_basic(n, step_reward, goal_reward):
    return GridWorld(rows=n, cols=n,
                      terminals={(n-1, n-1): goal_reward},
                      step_reward=step_reward, slip=0.0)


def main() -> None:
    n = 10
    gamma = 0.99  # large gamma → sparse rewards take longer to propagate
    goal = (n - 1, n - 1)
    goal_reward = 10.0
    tol = 1e-8

    scenarios = [
        ("Sparse: 0 per step", make_basic(n, step_reward=0.0, goal_reward=goal_reward)),
        ("Step penalty: −0.04 per step", make_basic(n, step_reward=-0.04, goal_reward=goal_reward)),
    ]
    env_shaped = ShapedGridWorld(rows=n, cols=n,
                                  terminals={goal: goal_reward},
                                  step_reward=-0.04, slip=0.0,
                                  goal=goal, gamma_for_shaping=gamma)
    scenarios.append(("Potential-shaped: −0.04 step + Δϕ", env_shaped))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (label, env) in zip(axes, scenarios):
        V, pi, hist = value_iteration(env, gamma=gamma, tol=tol, max_iters=5000)
        plot_policy(env, pi, V=V, ax=ax)
        ax.set_title(f"{label}\n"
                     f"VI iterations to tol={tol}: {hist['iters']}    "
                     rf"$V^\star$(start) = {V[env.state_to_idx[(0, 0)]]:.2f}",
                     fontsize=10)

    fig.suptitle(
        "Reward shaping preserves the optimal policy (arrows identical across panels) "
        "but changes value magnitudes.\n"
        "Synchronous value iteration converges in the same number of steps "
        "regardless — shaping helps sample-based methods, not VI.",
        fontsize=11, y=1.06,
    )
    fig.tight_layout()
    out = Path("figures/01c_sol3_reward_shaping.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
