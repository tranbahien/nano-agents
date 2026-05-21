"""Contextual bandit algorithms.

LinUCB and Linear Thompson Sampling, plus context-aware ε-greedy and a
context-free UCB1 baseline (to demonstrate the value of using context).

All algorithms use the disjoint parameterization: each arm k has its own
parameter vector theta_k in R^d. Maintain A_k = lambda*I + X_k^T X_k and
b_k = X_k^T r_k where X_k is the matrix of contexts on which arm k was
pulled and r_k the corresponding rewards. The posterior mean is then
theta_hat_k = A_k^{-1} b_k, and the predictive variance of x^T theta_k is
x^T A_k^{-1} x.

See: posts/01b-contextual-bandits.qmd
"""

from __future__ import annotations

import numpy as np


class _DisjointLinearBase:
    """Shared bookkeeping for disjoint linear contextual algorithms."""

    def __init__(self, K: int, d: int, lam: float = 1.0):
        self.K = K
        self.d = d
        self.lam = float(lam)
        # A_k = lambda * I (the prior precision); b_k = 0.
        self.A = np.array([self.lam * np.eye(d) for _ in range(K)])
        self.b = np.zeros((K, d))

    def update(self, x: np.ndarray, k: int, r: float) -> None:
        # Rank-1 update of A_k: A_k <- A_k + x x^T.
        # For online use we keep only A_k; inversion is done on demand.
        # See Sherman-Morrison for a fully-online O(d^2) update of A_k^{-1}.
        self.A[k] += np.outer(x, x)
        self.b[k] += r * x

    def theta_hat(self, k: int) -> np.ndarray:
        """Posterior mean for arm k."""
        return np.linalg.solve(self.A[k], self.b[k])


class LinUCB(_DisjointLinearBase):
    """Disjoint LinUCB (Li et al., 2010).

    Selects arg max_k [ x^T theta_hat_k + alpha * sqrt(x^T A_k^{-1} x) ].
    The bonus is the predictive standard deviation of x^T theta_k scaled
    by alpha, which controls exploration aggressiveness.
    """

    def __init__(self, K: int, d: int, alpha: float = 1.0, lam: float = 1.0):
        super().__init__(K, d, lam=lam)
        self.alpha = float(alpha)

    def select(self, x: np.ndarray) -> int:
        scores = np.empty(self.K)
        for k in range(self.K):
            A_inv = np.linalg.inv(self.A[k])
            theta_k = A_inv @ self.b[k]
            mean = x @ theta_k
            bonus = self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0))
            scores[k] = mean + bonus
        return int(np.argmax(scores))


class LinearThompsonSampling(_DisjointLinearBase):
    """Disjoint Linear Thompson Sampling (Agrawal & Goyal, 2013).

    Posterior over theta_k is N(theta_hat_k, v^2 A_k^{-1}). Sample
    tilde_theta_k ~ posterior for each arm; play arg max_k x^T tilde_theta_k.

    The exploration parameter v rescales posterior variance. v=1 corresponds
    to the standard Bayesian Thompson posterior under unit noise; some
    analyses recommend v > 1 for tight regret bounds.
    """

    def __init__(
        self,
        K: int,
        d: int,
        v: float = 1.0,
        lam: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(K, d, lam=lam)
        self.v = float(v)
        self.rng = rng if rng is not None else np.random.default_rng()

    def select(self, x: np.ndarray) -> int:
        scores = np.empty(self.K)
        for k in range(self.K):
            A_inv = np.linalg.inv(self.A[k])
            theta_hat = A_inv @ self.b[k]
            # Symmetrize for numerical stability before sampling.
            cov = 0.5 * (A_inv + A_inv.T) * (self.v ** 2)
            theta_tilde = self.rng.multivariate_normal(theta_hat, cov)
            scores[k] = x @ theta_tilde
        return int(np.argmax(scores))


class ContextualEpsilonGreedy(_DisjointLinearBase):
    """Context-aware ε-greedy: linear regression on each arm, then ε-greedy.

    Used as a baseline to isolate the benefit of confidence/uncertainty
    quantification (LinUCB, LinTS) over plain regression-then-greedy.
    """

    def __init__(self, K: int, d: int, eps: float = 0.1, lam: float = 1.0,
                 rng: np.random.Generator | None = None):
        super().__init__(K, d, lam=lam)
        self.eps = float(eps)
        self.rng = rng if rng is not None else np.random.default_rng()

    def select(self, x: np.ndarray) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.K))
        scores = np.array([x @ self.theta_hat(k) for k in range(self.K)])
        return int(np.argmax(scores))


class ContextFreeUCB:
    """UCB1 that ignores context — pretends rewards are i.i.d. per arm.

    Useful for showing how much regret is left on the table when an
    algorithm is blind to the contextual structure.
    """

    def __init__(self, K: int, d: int):
        self.K = K
        self.counts = np.zeros(K)
        self.means = np.zeros(K)
        self.t = 0

    def select(self, x: np.ndarray) -> int:
        self.t += 1
        for k in range(self.K):
            if self.counts[k] == 0:
                return k
        bonus = np.sqrt(2.0 * np.log(self.t) / self.counts)
        return int(np.argmax(self.means + bonus))

    def update(self, x: np.ndarray, k: int, r: float) -> None:
        self.counts[k] += 1
        self.means[k] += (r - self.means[k]) / self.counts[k]
