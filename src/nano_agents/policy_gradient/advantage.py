"""Advantage estimation.

Given a trajectory and a value function V, we can build many different
estimators of the advantage A_t = Q(s_t, a_t) - V(s_t). They differ in
their bias-variance tradeoff:

  Monte Carlo:  A_t = G_t - V(s_t)                  — unbiased, high variance.
  TD(0):        A_t = r_{t+1} + γ V(s_{t+1}) - V(s_t)   — biased (if V wrong),
                                                       low variance.
  n-step:       A_t^(n) = sum_{k=0..n-1} γ^k r_{t+k+1} + γ^n V(s_{t+n}) - V(s_t)
  GAE(γ, λ):    A_t = sum_{k=0..} (γλ)^k δ_{t+k}     — exponentially weighted
                                                       combination of n-step.
                  where δ_t = r_{t+1} + γ V(s_{t+1}) - V(s_t).

GAE(γ, 0) = TD(0).
GAE(γ, 1) = Monte Carlo (in the discounted sense).
"""

from __future__ import annotations

import numpy as np


def n_step_returns(
    rewards: list[float],
    values: np.ndarray,
    next_value: float,
    gamma: float,
    n: int,
) -> np.ndarray:
    """Compute n-step returns from each timestep onward.

    R_t^(n) = sum_{k=0..min(n, T-t-1)-1} γ^k r_{t+k+1}
              + γ^min(n, T-t) V_bootstrap
    where V_bootstrap = values[t+n] if t+n < T, else next_value.

    rewards: length T list of immediate rewards (rewards[t] is r_{t+1}).
    values:  length T array of V(s_t).
    next_value: V(s_T), the bootstrap target after the trajectory ends.
                For terminal states this is 0.
    """
    T = len(rewards)
    returns = np.zeros(T)
    for t in range(T):
        G = 0.0
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            G += (gamma ** k) * rewards[idx]
        boot_idx = t + n
        if boot_idx < T:
            G += (gamma ** n) * values[boot_idx]
        else:
            # Bootstrap with next_value at the end of the trajectory.
            steps_remaining = T - t
            G += (gamma ** steps_remaining) * next_value
        returns[t] = G
    return returns


def gae(
    rewards: list[float],
    values: np.ndarray,
    next_value: float,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Compute Generalized Advantage Estimation (Schulman et al. 2015).

    A_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + ...
    where δ_t = r_{t+1} + γ V(s_{t+1}) - V(s_t).

    This is computed efficiently by working backward through the trajectory:
        A_T = δ_T
        A_t = δ_t + γλ A_{t+1}     (assuming non-terminal at t+1)

    Returns an array of length T = len(rewards) of advantages.
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_advantage = 0.0
    # Process in reverse.
    for t in reversed(range(T)):
        if t == T - 1:
            v_next = next_value
        else:
            v_next = values[t + 1]
        delta = rewards[t] + gamma * v_next - values[t]
        advantages[t] = delta + gamma * lam * last_advantage
        last_advantage = advantages[t]
    return advantages


def discounted_returns_from_advantages(
    advantages: np.ndarray, values: np.ndarray,
) -> np.ndarray:
    """Recover target returns (for critic regression) as A_t + V(s_t)."""
    return advantages + values
