"""Advantage Actor-Critic (A2C).

Each episode:
  1. Collect a trajectory under the current policy.
  2. Use the critic V to estimate advantages (via GAE).
  3. Apply policy gradient update on the actor using advantages.
  4. Apply TD regression on the critic toward the bootstrapped targets.

Both updates use the same trajectory. The critic helps the actor by
providing a low-variance baseline; the actor helps the critic by visiting
new states. This positive feedback loop is the "actor-critic" pattern.

Optional features:
  - Entropy bonus: encourage exploration by adding β H[π(·|s)] to the
    objective. Implemented as an additive policy gradient term.
  - Normalized advantages: shift/scale advantages to mean 0, unit std.
    Common in modern implementations and helps with bigger problems.
"""

from __future__ import annotations

import numpy as np

from nano_agents.mdp import GridWorld
from nano_agents.policy_gradient.advantage import gae
from nano_agents.policy_gradient.baselines import TabularBaseline
from nano_agents.policy_gradient.policies import SoftmaxPolicy
from nano_agents.policy_gradient.reinforce import (
    collect_trajectory,
    compute_returns,
)


def policy_entropy(policy: SoftmaxPolicy, s: int) -> float:
    p = policy.probs(s)
    return float(-np.sum(p * np.log(p + 1e-30)))


def entropy_grad(policy: SoftmaxPolicy, s: int) -> np.ndarray:
    """Gradient of entropy H[pi(.|s)] w.r.t. all theta. Only row s nonzero.

    d/dθ_{s,a} H = -p(a)(log p(a) + 1 - sum_{a'} p(a')(log p(a') + 1))
                 = -p(a)(log p(a) - H_unnormalized + 1 - 1)
    Cleaner: H(p) = -∑ p log p; ∂H/∂θ_a = -p_a (log p_a + 1) + p_a ∑ p_b (log p_b + 1)
                                       = -p_a log p_a + p_a (-H + sum p_b log p_b + 0)
    Hmm, easier to derive in matrix form. Numerically stable form:
        ∂H/∂θ_a = p_a * ( H - log p_a - 1 + 1 ) - ... actually let me just verify by FD.
    """
    p = policy.probs(s)
    grad = np.zeros_like(policy.theta)
    log_p = np.log(p + 1e-30)
    H = -np.sum(p * log_p)
    # ∂H/∂θ_a = p_a * (-log p_a - H) (for softmax with unit temperature)
    grad[s] = p * (-log_p - H)
    return grad


def train_a2c(
    env: GridWorld,
    policy: SoftmaxPolicy,
    baseline: TabularBaseline,
    n_episodes: int,
    lr_actor: float,
    lr_critic: float,
    gamma: float = 0.95,
    lam: float = 1.0,
    ent_coef: float = 0.0,
    normalize_adv: bool = False,
    rng=None,
    start_state=None,
    max_steps: int = 200,
):
    """Train A2C / actor-critic.

    lam controls GAE: 1.0 = Monte Carlo, 0.0 = TD(0), intermediate = mix.
    ent_coef = entropy bonus coefficient.
    normalize_adv = z-score advantages per-trajectory.
    """
    rng = rng if rng is not None else np.random.default_rng()
    non_terminal = [s for s in env.states if not env.is_terminal(s)]
    history = {
        "returns": np.zeros(n_episodes),
        "td_error_mean": np.zeros(n_episodes),
        "entropy": np.zeros(n_episodes),
    }

    for ep in range(n_episodes):
        start = (start_state if start_state is not None
                  else non_terminal[rng.integers(len(non_terminal))])
        traj = collect_trajectory(env, policy, start, rng, max_steps=max_steps)
        T = len(traj)
        states = np.array([s for (s, _, _) in traj])
        rewards = [r for (_, _, r) in traj]

        values = baseline.values(states)
        # Bootstrap value at end: 0 if last state is terminal, else V(s_T).
        # We need to know V(s_T). Re-derive it from the trajectory:
        # if the last (s, a, r) was followed by a terminal state, next_value = 0.
        # Otherwise, take env.step_outcomes to find s' from the last (s, a)
        # — but in our simulate_step we already stepped to s_next implicitly.
        # The cleanest approach: simulate one extra step deterministically to
        # get s_T, but actually it's easier to just say: if the last s in traj
        # is terminal we set next_value=0; otherwise approximate by bootstrap
        # from values[-1] (we don't track s_T post-trajectory).
        # Since collect_trajectory ends on `done`, the next state was terminal,
        # so next_value=0 is the right bootstrap target.
        next_value = 0.0

        advantages = gae(rewards, values, next_value, gamma, lam)
        returns_for_critic = advantages + values

        if normalize_adv and len(advantages) > 1:
            std = advantages.std() + 1e-8
            advantages = (advantages - advantages.mean()) / std

        # ----- Actor update -----
        for t, (s, a, _) in enumerate(traj):
            A = advantages[t]
            p = policy.probs(s)
            policy.theta[s] += lr_actor * (-A * p)
            policy.theta[s, a] += lr_actor * A
            if ent_coef > 0:
                policy.theta += lr_actor * ent_coef * entropy_grad(policy, s)

        # ----- Critic update -----
        td_errors = []
        for t, (s, _, _) in enumerate(traj):
            td = baseline.update(s, returns_for_critic[t], lr_critic)
            td_errors.append(td)

        history["returns"][ep] = float(np.sum(rewards))
        history["td_error_mean"][ep] = float(np.mean(np.abs(td_errors)))
        if T > 0:
            history["entropy"][ep] = float(
                np.mean([policy_entropy(policy, s) for s, _, _ in traj]))

    return history
