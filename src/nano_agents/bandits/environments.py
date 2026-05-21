"""Bandit environments."""

from __future__ import annotations

import numpy as np


class BernoulliBandit:
    """K-armed Bernoulli bandit.

    Each arm k yields reward 1 with probability probs[k], else 0.
    """

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)
        self.K = len(self.probs)
        self.best = float(self.probs.max())

    def pull(self, k: int) -> int:
        return int(np.random.binomial(1, self.probs[k]))

    def regret_of(self, k: int) -> float:
        """Instantaneous regret of pulling arm k."""
        return self.best - float(self.probs[k])


class GaussianBandit:
    """K-armed Gaussian bandit with known variance.

    Each arm k yields reward N(means[k], sigma^2).
    """

    def __init__(self, means, sigma: float = 1.0):
        self.means = np.asarray(means, dtype=float)
        self.sigma = float(sigma)
        self.K = len(self.means)
        self.best = float(self.means.max())

    def pull(self, k: int) -> float:
        return float(np.random.normal(self.means[k], self.sigma))

    def regret_of(self, k: int) -> float:
        return self.best - float(self.means[k])


class DriftingBernoulliBandit:
    """Non-stationary Bernoulli bandit with sinusoidally drifting arm means.

    Arm k's mean at time t is base[k] + amplitude * sin(2*pi*(t + phase[k])/period).
    Means are clipped to [0.05, 0.95] to keep them valid Bernoulli probabilities.
    Track a global timer that advances on each pull.
    """

    def __init__(self, base, amplitude: float = 0.3, period: float = 2000.0,
                 phases=None, seed: int = 0):
        self.base = np.asarray(base, dtype=float)
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.K = len(self.base)
        if phases is None:
            rng = np.random.default_rng(seed)
            phases = rng.uniform(0, period, size=self.K)
        self.phases = np.asarray(phases, dtype=float)
        self.t = 0

    def _means(self) -> np.ndarray:
        return np.clip(
            self.base + self.amplitude * np.sin(2 * np.pi * (self.t + self.phases) / self.period),
            0.05,
            0.95,
        )

    @property
    def probs(self) -> np.ndarray:
        return self._means()

    @property
    def best(self) -> float:
        return float(self._means().max())

    def pull(self, k: int) -> int:
        mu = self._means()[k]
        r = int(np.random.binomial(1, mu))
        self.t += 1
        return r

    def regret_of(self, k: int) -> float:
        m = self._means()
        return float(m.max() - m[k])
