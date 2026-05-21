"""Dynamic programming solvers for finite MDPs.

  value_iteration:  iterate the Bellman optimality operator until convergence.
  policy_iteration: alternate policy evaluation and policy improvement.

Both return (V, pi, history) where:
  V       is the converged state-value function (length nS),
  pi      is a deterministic policy (length nS, dtype int),
  history is a dict with diagnostic series (e.g. max-norm error per iter).
"""

from __future__ import annotations

import numpy as np

from .environments import GridWorld


def _bellman_q(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Return Q[s, a] = R[s, a] + gamma sum_s' P[s, a, s'] V[s']."""
    return R + gamma * (P @ V)


def value_iteration(
    env: GridWorld, gamma: float = 0.95, tol: float = 1e-8, max_iters: int = 5000,
):
    """Iterate V_{k+1}(s) = max_a [R(s, a) + gamma sum_s' P(s, a, s') V_k(s')]."""
    P, R = env.transition_tensors()
    V = np.zeros(env.nS)
    history = {"max_diff": [], "V_snapshots": []}
    for it in range(max_iters):
        Q = _bellman_q(V, P, R, gamma)
        V_new = Q.max(axis=1)
        diff = np.max(np.abs(V_new - V))
        history["max_diff"].append(diff)
        history["V_snapshots"].append(V.copy())
        V = V_new
        if diff < tol:
            break
    Q = _bellman_q(V, P, R, gamma)
    pi = Q.argmax(axis=1)
    history["V_snapshots"].append(V.copy())
    history["iters"] = it + 1
    return V, pi, history


def policy_evaluation(
    pi: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float,
    tol: float = 1e-10, max_iters: int = 100_000,
) -> np.ndarray:
    """Iteratively solve V^pi(s) = R(s, pi(s)) + gamma sum_s' P(s, pi(s), s') V^pi(s')."""
    nS = P.shape[0]
    V = np.zeros(nS)
    # Build per-state P_pi and R_pi for the deterministic policy.
    P_pi = P[np.arange(nS), pi]      # shape (nS, nS)
    R_pi = R[np.arange(nS), pi]      # shape (nS,)
    for _ in range(max_iters):
        V_new = R_pi + gamma * (P_pi @ V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def policy_iteration(
    env: GridWorld, gamma: float = 0.95, tol: float = 1e-10, max_iters: int = 1000,
):
    """Alternate exact policy evaluation and policy improvement until stable.

    Tie-breaking: only switch arm pi[s] if the new action's Q value is
    strictly greater than the current action's Q value (by `tol`). This
    avoids infinite loops when multiple actions are equally optimal.
    """
    P, R = env.transition_tensors()
    pi = np.zeros(env.nS, dtype=int)
    history = {"policy_changes_per_iter": [], "V_snapshots": []}
    for it in range(max_iters):
        V = policy_evaluation(pi, P, R, gamma, tol=tol)
        Q = _bellman_q(V, P, R, gamma)
        current_Q = Q[np.arange(env.nS), pi]
        best_Q = Q.max(axis=1)
        improvable = best_Q > current_Q + tol
        pi_new = pi.copy()
        pi_new[improvable] = Q[improvable].argmax(axis=1)
        changed = int(np.sum(pi_new != pi))
        history["policy_changes_per_iter"].append(changed)
        history["V_snapshots"].append(V.copy())
        if changed == 0:
            history["iters"] = it + 1
            return V, pi_new, history
        pi = pi_new
    history["iters"] = it + 1
    return V, pi, history


def bellman_optimality_residual(
    V: np.ndarray, env: GridWorld, gamma: float,
) -> float:
    """Max-norm residual ||T V - V||_inf, where T is the Bellman optimality operator."""
    P, R = env.transition_tensors()
    Q = _bellman_q(V, P, R, gamma)
    return float(np.max(np.abs(Q.max(axis=1) - V)))
