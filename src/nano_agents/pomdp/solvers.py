"""POMDP solvers.

Two functions:
  belief_update(b, a, o, P, Z): Bayesian update of belief b given (a, o).
  discretized_pomdp_value_iteration: discretize the belief simplex to a grid
    and solve the resulting approximate finite MDP. Returns the value function
    and policy as functions of the grid index.

Exact POMDP solving uses α-vectors (Sondik 1971) — the optimal value function
is piecewise linear and convex in belief. The discretized approach is an
approximation that converges to the exact solution as the grid is refined,
and is much simpler to implement.
"""

from __future__ import annotations

import numpy as np


def belief_update(b: np.ndarray, a: int, o: int, P: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Bayesian belief update.

    Given current belief b ∈ Δ(S), action a, observation o, returns the
    posterior belief b' over next states:

      b'(s') ∝ Z(o | s', a) * Σ_s P(s' | s, a) b(s)

    Returns a normalized probability vector. If P(o | b, a) = 0, returns
    a uniform belief as a fallback (this shouldn't happen with the right
    observation model).
    """
    # Predict: distribution over next states under action a.
    pred = b @ P[:, a, :]                     # shape (nS,)
    # Update: weight by observation likelihood.
    posterior = Z[:, a, o] * pred
    Z_norm = posterior.sum()
    if Z_norm <= 0:
        return np.ones_like(b) / len(b)
    return posterior / Z_norm


def discretized_pomdp_value_iteration(
    pomdp,
    n_belief: int = 101,
    gamma: float = 0.95,
    tol: float = 1e-7,
    max_iters: int = 5000,
):
    """Solve a 2-state POMDP by discretizing the belief simplex.

    For a 2-state POMDP, belief is fully described by b(0) ∈ [0, 1] and
    b(1) = 1 - b(0). We discretize b(0) into n_belief points and treat the
    result as a finite MDP whose 'states' are these belief points.

    For each (b_i, a), we compute the expected immediate reward and the
    distribution over next belief points (by snapping the updated belief
    to the nearest grid point, weighted by observation probability).

    Returns (V, pi, b_grid, history) with:
      V[i]     = optimal value at belief point i,
      pi[i]    = optimal action at belief point i,
      b_grid   = the 1D grid of belief values (length n_belief),
      history  = diagnostic dict with 'max_diff' per iteration.
    """
    assert pomdp.nS == 2, "This solver is specialized to 2-state POMDPs."
    P, Z, R = pomdp.P, pomdp.Z, pomdp.R
    nA = pomdp.nA
    nO = pomdp.nO

    # Belief grid: b(0) values from 0 to 1.
    b_grid = np.linspace(0.0, 1.0, n_belief)
    N = n_belief

    # Precompute the discretized transition tensor T[i, a, j] and reward Rb[i, a].
    T = np.zeros((N, nA, N))
    Rb = np.zeros((N, nA))
    for i, b0 in enumerate(b_grid):
        b = np.array([b0, 1.0 - b0])
        for a in range(nA):
            # Expected immediate reward.
            Rb[i, a] = b @ R[:, a]
            # For each possible observation o, compute the next belief and its prob.
            pred = b @ P[:, a, :]   # next-state distribution before observing
            for o in range(nO):
                # P(o | b, a) = Σ_{s'} pred(s') Z(o | s', a)
                p_o = float((pred * Z[:, a, o]).sum())
                if p_o <= 0:
                    continue
                # Posterior belief after observing o.
                posterior = Z[:, a, o] * pred / p_o
                # Snap to nearest grid point (linear interpolation between two
                # neighbors would be more accurate; nearest-point is simpler).
                b0_next = float(posterior[0])
                j = int(round(b0_next * (N - 1)))
                j = max(0, min(N - 1, j))
                T[i, a, j] += p_o

    # Standard value iteration on the discretized MDP.
    V = np.zeros(N)
    history = {"max_diff": []}
    for it in range(max_iters):
        Q = Rb + gamma * (T @ V)             # shape (N, nA)
        V_new = Q.max(axis=1)
        diff = float(np.max(np.abs(V_new - V)))
        history["max_diff"].append(diff)
        V = V_new
        if diff < tol:
            break
    Q = Rb + gamma * (T @ V)
    pi = Q.argmax(axis=1)
    history["iters"] = it + 1
    return V, pi, b_grid, history
