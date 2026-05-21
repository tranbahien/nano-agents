"""Bandit algorithms: ε-greedy, UCB1, Thompson Sampling.

See: posts/01a-multi-armed-bandits.qmd for derivations.
"""

from __future__ import annotations

import numpy as np


class EpsilonGreedy:
    """ε-greedy: pull arg max with probability 1-ε, else uniform."""

    def __init__(self, K: int, eps: float = 0.1):
        self.K = K
        self.eps = eps
        self.counts = np.zeros(K)
        self.means = np.zeros(K)

    def select(self) -> int:
        if np.random.rand() < self.eps:
            return int(np.random.randint(self.K))
        return int(np.argmax(self.means))

    def update(self, k: int, r: float) -> None:
        self.counts[k] += 1
        # Running mean update — Welford-style for numerical stability.
        self.means[k] += (r - self.means[k]) / self.counts[k]


class UCB1:
    """UCB1 (Auer, Cesa-Bianchi, Fischer, 2002).

    Selects arg max_k [ mean_k + sqrt(2 log t / n_k) ].
    Achieves O(log T) regret, matching the Lai–Robbins lower bound up to constants.
    """

    def __init__(self, K: int):
        self.K = K
        self.counts = np.zeros(K)
        self.means = np.zeros(K)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        # Play each arm at least once.
        for k in range(self.K):
            if self.counts[k] == 0:
                return k
        bonus = np.sqrt(2.0 * np.log(self.t) / self.counts)
        return int(np.argmax(self.means + bonus))

    def update(self, k: int, r: float) -> None:
        self.counts[k] += 1
        self.means[k] += (r - self.means[k]) / self.counts[k]


class ThompsonSampling:
    """Bernoulli Thompson Sampling with Beta(alpha, beta) prior.

    Posterior is Beta(alpha + successes, beta + failures) by conjugacy.
    Probability of selecting arm k equals the posterior probability that it is optimal.
    """

    def __init__(self, K: int, alpha: float = 1.0, beta: float = 1.0):
        self.K = K
        self.alpha = np.full(K, float(alpha))
        self.beta = np.full(K, float(beta))

    def select(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, k: int, r: float) -> None:
        self.alpha[k] += r
        self.beta[k] += 1.0 - r


class GaussianUCB:
    """UCB for Gaussian rewards with known variance.

    Bonus is sqrt(2 sigma^2 log t / n_k), the right scale-up of UCB1 for
    sub-Gaussian rewards with parameter sigma.
    """

    def __init__(self, K: int, sigma: float = 1.0):
        self.K = K
        self.sigma = float(sigma)
        self.counts = np.zeros(K)
        self.means = np.zeros(K)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        for k in range(self.K):
            if self.counts[k] == 0:
                return k
        bonus = self.sigma * np.sqrt(2.0 * np.log(self.t) / self.counts)
        return int(np.argmax(self.means + bonus))

    def update(self, k: int, r: float) -> None:
        self.counts[k] += 1
        self.means[k] += (r - self.means[k]) / self.counts[k]


class GaussianThompsonSampling:
    """Gaussian Thompson Sampling with Normal-Inverse-Gamma conjugate prior.

    The prior over (mu_k, sigma_k^2) is NIG(mu_0, kappa_0, alpha_0, beta_0).
    After n observations with empirical mean x_bar and SSE = sum (x_i - x_bar)^2:

        kappa_n = kappa_0 + n
        mu_n    = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        alpha_n = alpha_0 + n / 2
        beta_n  = beta_0 + 0.5 * SSE + 0.5 * kappa_0 * n * (x_bar - mu_0)^2 / kappa_n

    The marginal posterior of mu is Student-t_{2 alpha_n}(mu_n, beta_n / (alpha_n kappa_n)).
    Sampling: draw sigma^2 ~ InvGamma(alpha_n, beta_n), then mu | sigma^2 ~ N(mu_n, sigma^2 / kappa_n).

    Default prior (mu_0=0, kappa_0=1, alpha_0=1, beta_0=1) is moderately weak.
    """

    def __init__(
        self,
        K: int,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.K = K
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        # Welford-style running stats per arm.
        self.n = np.zeros(K)
        self.mean = np.zeros(K)
        self.M2 = np.zeros(K)  # sum of squared deviations from running mean
        self.rng = rng if rng is not None else np.random.default_rng()

    def _posterior_params(self, k: int):
        n = self.n[k]
        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * self.mean[k]) / kappa_n
        alpha_n = self.alpha0 + n / 2.0
        delta = self.mean[k] - self.mu0
        beta_n = self.beta0 + 0.5 * self.M2[k] + 0.5 * self.kappa0 * n * delta ** 2 / kappa_n
        return mu_n, kappa_n, alpha_n, beta_n

    def select(self) -> int:
        samples = np.empty(self.K)
        for k in range(self.K):
            mu_n, kappa_n, alpha_n, beta_n = self._posterior_params(k)
            # InvGamma(alpha, beta): sigma^2 = beta / Gamma(alpha, scale=1).
            sigma2 = beta_n / self.rng.gamma(shape=alpha_n, scale=1.0)
            mu_tilde = self.rng.normal(mu_n, np.sqrt(sigma2 / kappa_n))
            samples[k] = mu_tilde
        return int(np.argmax(samples))

    def update(self, k: int, r: float) -> None:
        # Welford's online update of mean and M2 (sum of squared deviations).
        self.n[k] += 1
        delta = r - self.mean[k]
        self.mean[k] += delta / self.n[k]
        delta2 = r - self.mean[k]
        self.M2[k] += delta * delta2
