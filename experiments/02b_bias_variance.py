"""Figure: bias-variance spectrum of advantage estimators.

Generates a fixed trajectory under a fixed policy and a fixed (somewhat
inaccurate) value function. Computes advantage estimates at different n
(for n-step returns) and different λ (for GAE), and plots their empirical
variance vs the bias they introduce relative to the true MC advantage.

The point: short n (or low λ) is low-variance but high-bias when V is
inaccurate; long n (or high λ) is unbiased but high-variance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.mdp import GridWorld, TwoGoalGridWorld, value_iteration
from nano_agents.policy_gradient import (
    SoftmaxPolicy,
    TabularBaseline,
    collect_trajectory,
    compute_returns,
    gae,
    n_step_returns,
    train_a2c,
)


def main() -> None:
    gamma = 0.95
    env = TwoGoalGridWorld(slip=0.0, step_reward=-0.04)
    V_star, _, _ = value_iteration(env, gamma=gamma)

    # Use a partially-trained policy with a fixed start state, so trajectory
    # noise comes only from the stochasticity of the policy.
    rng = np.random.default_rng(0)
    policy = SoftmaxPolicy(env.nS, env.nA)
    from nano_agents.policy_gradient import train_reinforce
    train_reinforce(env, policy, n_episodes=400, lr=0.05, gamma=gamma,
                     baseline="mean", rng=rng, start_state=env.start)

    sigma_noise = 1.5
    V_noisy = V_star + rng.normal(0, sigma_noise, size=env.nS)

    # Collect many trajectories all starting from env.start.
    n_trajs = 600

    def estimator_stats(make_estimator):
        adv0 = np.zeros(n_trajs)
        true0 = np.zeros(n_trajs)
        for k in range(n_trajs):
            traj = collect_trajectory(env, policy, env.start,
                                       np.random.default_rng(k + 100),
                                       max_steps=100)
            if len(traj) == 0:
                continue
            rewards = [r for (_, _, r) in traj]
            states = np.array([s for (s, _, _) in traj])
            values = V_noisy[states]
            G0 = compute_returns(rewards, gamma)[0]
            true0[k] = G0 - V_star[states[0]]
            adv0[k] = make_estimator(rewards, values, traj)[0]
        bias = float(np.mean(adv0 - true0))
        std = float(np.std(adv0))
        return bias, std

    # GAE.
    lam_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    gae_stats = []
    for lam in lam_values:
        def make(rewards, values, traj, lam=lam):
            return gae(rewards, values, 0.0, gamma, lam)
        gae_stats.append(estimator_stats(make))

    fig, ax = plt.subplots(figsize=(10, 5.2))

    biases = np.array([s[0] for s in gae_stats])
    stds = np.array([s[1] for s in gae_stats])
    sc = ax.scatter(stds, biases, c=lam_values,
                     cmap="plasma", s=120, zorder=3,
                     edgecolor="white", linewidth=1.5)
    for i, lam in enumerate(lam_values):
        ax.annotate(f"λ = {lam}", (stds[i], biases[i]),
                    textcoords="offset points", xytext=(9, 6),
                    fontsize=10)
    ax.plot(stds, biases, color="#888", linewidth=1, alpha=0.5, zorder=2)
    ax.axhline(0, color="#222", linestyle="--", linewidth=1, alpha=0.5,
                label="zero bias (Monte Carlo)")

    # Highlight the sweet spot: lowest absolute bias (the point closest to zero-bias line).
    best_i = int(np.argmin(np.abs(biases)))
    ax.annotate(
        f"closest to zero bias\nat moderate variance",
        xy=(stds[best_i], biases[best_i]),
        xytext=(stds[best_i] - 0.15, biases[best_i] + 0.4),
        fontsize=10, color="#222",
        arrowprops=dict(arrowstyle="->", color="#222"),
    )

    ax.set_xlabel(r"std (variance) of $\hat{A}_0$")
    ax.set_ylabel(r"bias of $\hat{A}_0$ vs $G_0 - V^\star$")
    ax.set_title(
        rf"GAE($\gamma, \lambda$) bias-variance trade-off"
        f" (critic noise σ = {sigma_noise}, fixed start)\n"
        r"$\lambda = 0$: TD(0)-like (low variance, biased).  "
        r"$\lambda = 1$: Monte Carlo (unbiased, high variance).",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path("figures/02b_bias_variance.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
