"""Policy parameterizations.

Two flavors:
  SoftmaxPolicy: tabular policy over discrete actions, parameterized by logits
                 theta[s, a]. Gives pi(a|s) = softmax_a(theta[s, :]).
  GaussianPolicy: 1D Gaussian policy with parameterized mean and (optionally
                  learnable) log-std. For continuous action problems.
"""

from __future__ import annotations

import numpy as np


class SoftmaxPolicy:
    """Tabular softmax policy: pi(a|s) = softmax(theta[s, :])_a.

    The score (gradient of log-prob) has a clean closed form:

        d log pi(a|s) / d theta[s', a'] = (1[a=a'] - pi(a'|s)) * 1[s=s']

    so the gradient is nonzero only at row s of theta.
    """

    def __init__(self, nS: int, nA: int, temperature: float = 1.0):
        self.nS = nS
        self.nA = nA
        self.temperature = float(temperature)
        self.theta = np.zeros((nS, nA))

    def probs(self, s: int) -> np.ndarray:
        z = self.theta[s] / self.temperature
        z = z - z.max()  # for numerical stability
        e = np.exp(z)
        return e / e.sum()

    def sample(self, s: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.nA, p=self.probs(s)))

    def log_prob(self, s: int, a: int) -> float:
        return float(np.log(self.probs(s)[a] + 1e-30))

    def grad_log_prob(self, s: int, a: int) -> np.ndarray:
        """Gradient of log pi(a|s) w.r.t. self.theta, shape (nS, nA).

        Only row s is nonzero: it is (e_a - pi(.|s)) / temperature.
        """
        g = np.zeros_like(self.theta)
        p = self.probs(s)
        g[s] = -p / self.temperature
        g[s, a] += 1.0 / self.temperature
        return g

    def greedy(self) -> np.ndarray:
        """Deterministic greedy policy: argmax of theta per row."""
        return self.theta.argmax(axis=1)


class GaussianPolicy:
    """1D Gaussian policy on a continuous action.

    pi(a|s) = N(mu(s), sigma^2). For pedagogical simplicity:
      - mu(s) = a state-conditional parameter (stored as a vector if we discretize s,
        or a function of s in continuous-state versions).
      - sigma fixed at construction (could be learnable; we keep it fixed for clarity).

    For the experiments here we use a single state (a continuous bandit), so mu is
    a scalar.
    """

    def __init__(self, mu_init: float = 0.0, sigma: float = 1.0):
        self.mu = float(mu_init)
        self.sigma = float(sigma)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mu, self.sigma))

    def log_prob(self, a: float) -> float:
        return float(
            -0.5 * np.log(2 * np.pi * self.sigma ** 2)
            - 0.5 * ((a - self.mu) / self.sigma) ** 2
        )

    def grad_log_prob_mu(self, a: float) -> float:
        """d log pi(a) / d mu = (a - mu) / sigma^2."""
        return float((a - self.mu) / (self.sigma ** 2))
