"""Figure: gradient variance with and without baseline.

Setup: take a fixed policy (a uniform-random one early in training, and a
near-optimal one late in training). For each, collect 500 single-trajectory
gradient estimates and look at the distribution of gradient norms.

The optimal-baseline theory predicts that subtracting V^pi(s) from G_t
minimizes the variance of the gradient estimator while keeping it unbiased.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import (
    GridWorld,
    TwoGoalGridWorld,
    simulate_step,
    value_iteration,
)
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    collect_trajectory,
    compute_returns,
    train_reinforce,
)


def gradient_estimate(env, policy, baseline, gamma, rng):
    """Compute a single-trajectory gradient estimate. Returns the full gradient (sparse)."""
    non_terminal = [s for s in env.states if not env.is_terminal(s)]
    start = non_terminal[rng.integers(len(non_terminal))]
    traj = collect_trajectory(env, policy, start, rng, max_steps=200)
    rewards = [r for (_, _, r) in traj]
    G = compute_returns(rewards, gamma)

    grad = np.zeros_like(policy.theta)
    for i, (s, a, _) in enumerate(traj):
        adv = G[i] if baseline is None else G[i] - baseline[s]
        p = policy.probs(s)
        grad[s] += adv * (-p)
        grad[s, a] += adv
    return grad


def gradient_variance_distribution(env, policy, baseline, gamma, rng, n=500):
    """Return the L2 norms of n independent gradient estimates."""
    norms = np.zeros(n)
    for i in range(n):
        g = gradient_estimate(env, policy, baseline, gamma, rng)
        norms[i] = float(np.linalg.norm(g))
    return norms


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    V_star, _, _ = value_iteration(env, gamma=gamma)

    # Two policies: a random (uniform) one, and a partially-trained one.
    policy_random = SoftmaxPolicy(env.nS, env.nA)

    policy_trained = SoftmaxPolicy(env.nS, env.nA)
    train_reinforce(env, policy_trained, n_episodes=300, lr=0.1, gamma=gamma,
                     baseline="mean", rng=np.random.default_rng(0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    for ax, policy, label in zip(axes, [policy_random, policy_trained],
                                   ["Untrained (uniform) policy",
                                    "Partially-trained policy"]):
        norms_no = gradient_variance_distribution(
            env, policy, None, gamma,
            rng=np.random.default_rng(0))
        norms_baseline = gradient_variance_distribution(
            env, policy, V_star, gamma,
            rng=np.random.default_rng(0))

        bins = np.linspace(0, max(norms_no.max(), norms_baseline.max()) * 1.05, 40)
        ax.hist(norms_no, bins=bins, color="#c44e52", alpha=0.6,
                 label=f"no baseline (std = {norms_no.std():.2f})")
        ax.hist(norms_baseline, bins=bins, color="#3a7ebf", alpha=0.6,
                 label=rf"$V^\star$ baseline (std = {norms_baseline.std():.2f})")
        ax.set_xlabel(r"$\|\hat{\nabla}_\theta J\|_2$  (single-trajectory)")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3); ax.set_axisbelow(True)

    axes[0].set_ylabel("count over 500 trajectories")

    fig.suptitle(
        r"A baseline only reduces variance if it tracks $V^{\pi_\theta}$. "
        r"$V^\star$ (the *optimal* policy's value) helps near convergence but hurts when far away.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = Path("figures/02a_gradient_variance.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
