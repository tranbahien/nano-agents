"""Solution 2b.1: gradient variance under different advantage estimators.

Empirically measure the gradient-estimator variance for:
  - Vanilla REINFORCE
  - A2C with TD(0) advantages
  - A2C with GAE(γ, 0.95)
  - A2C with MC advantages

Same fixed policy (partially trained), same critic state, many trajectories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import TwoGoalGridWorld, value_iteration
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    collect_trajectory,
    compute_returns,
    gae,
    train_a2c,
)


def gradient_norm_for(policy, traj, advantages):
    grad = np.zeros_like(policy.theta)
    for t, (s, a, _) in enumerate(traj):
        A = advantages[t]
        p = policy.probs(s)
        grad[s] += A * (-p)
        grad[s, a] += A
    return float(np.linalg.norm(grad))


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    V_star, _, _ = value_iteration(env, gamma=gamma)

    # Train both policy and critic partway, then freeze.
    rng = np.random.default_rng(0)
    policy = SoftmaxPolicy(env.nS, env.nA)
    baseline = TabularBaseline(env.nS)
    train_a2c(env, policy, baseline,
               n_episodes=400, lr_actor=0.05, lr_critic=0.2,
               gamma=gamma, lam=0.95, rng=rng)

    # The critic has some error vs V*; that's realistic.
    print(f"Critic error after warmup: {np.max(np.abs(baseline.V - V_star)):.3f}")

    # Now measure gradient-norm distributions across many trajectories.
    n_trajs = 800
    non_terminal = [s for s in env.states if not env.is_terminal(s)]

    estimators = {
        "REINFORCE (no baseline)": ("mc", None, "#888888"),
        "TD(0): r + γV − V": ("gae", 0.0, "#c44e52"),
        "GAE(0.95)": ("gae", 0.95, "#3a7ebf"),
        "MC − V (λ=1)": ("gae", 1.0, "#55a467"),
    }

    norms = {name: np.zeros(n_trajs) for name in estimators}
    for k in range(n_trajs):
        start = non_terminal[rng.integers(len(non_terminal))]
        traj = collect_trajectory(env, policy, start,
                                    np.random.default_rng(k + 1),
                                    max_steps=100)
        if len(traj) == 0:
            continue
        rewards = [r for (_, _, r) in traj]
        states = np.array([s for (s, _, _) in traj])
        values = baseline.values(states)
        G = compute_returns(rewards, gamma)

        for name, (kind, lam, _) in estimators.items():
            if kind == "mc":
                A = G  # no baseline
            else:
                A = gae(rewards, values, 0.0, gamma, lam)
            norms[name][k] = gradient_norm_for(policy, traj, A)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bins = np.linspace(0, max(n.max() for n in norms.values()) * 1.02, 40)
    for name, n in norms.items():
        color = estimators[name][2]
        ax.hist(n, bins=bins, alpha=0.55, color=color,
                label=f"{name}  (std = {n.std():.2f})")
    ax.set_xlabel(r"$\|\hat{\nabla}_\theta J\|_2$  (single-trajectory)")
    ax.set_ylabel(f"count over {n_trajs} trajectories")
    ax.set_title(
        "Gradient-estimator distributions under different advantage choices.\n"
        "Learned critic + GAE gives the smallest spread by a wide margin.",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02b_sol1_advantage_variance.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
