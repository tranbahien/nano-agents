"""Contextual bandit environments."""

from __future__ import annotations

import numpy as np


class LinearContextualBandit:
    """K-armed linear contextual bandit (disjoint parameterization).

    Each arm k has a true parameter theta_k in R^d. For context x,
    the expected reward of pulling arm k is x^T theta_k. Realized reward
    is x^T theta_k + N(0, sigma^2).
    """

    def __init__(self, K: int, d: int, sigma: float = 0.1, seed: int = 0):
        self.K = K
        self.d = d
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

        # Random arm parameters on the unit sphere — keeps reward scales comparable.
        thetas = self.rng.standard_normal((K, d))
        thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)
        self.thetas = thetas

    def sample_context(self) -> np.ndarray:
        """Sample a unit-norm context vector."""
        x = self.rng.standard_normal(self.d)
        x /= np.linalg.norm(x)
        return x

    def pull(self, x: np.ndarray, k: int) -> float:
        """Realized reward of pulling arm k under context x."""
        return float(x @ self.thetas[k] + self.rng.normal(0.0, self.sigma))

    def expected_reward(self, x: np.ndarray, k: int) -> float:
        return float(x @ self.thetas[k])

    def best_action(self, x: np.ndarray) -> int:
        return int(np.argmax(self.thetas @ x))

    def best_reward(self, x: np.ndarray) -> float:
        return float(np.max(self.thetas @ x))

    def regret_of(self, x: np.ndarray, k: int) -> float:
        return self.best_reward(x) - self.expected_reward(x, k)


class NonlinearContextualBandit:
    """Contextual bandit with non-linear reward functions.

    Used to study misspecification: linear algorithms will fit the best linear
    approximation but cannot reach zero regret because the truth isn't linear.

    Each arm k has a true reward function f_k(x) of the form:
        f_k(x) = sin(omega_k . x_1) + theta_k . x[1:]
    where omega_k is the non-linear frequency for arm k. The first feature x_0
    enters non-linearly; the remaining features are linear.
    """

    def __init__(self, K: int, d: int, sigma: float = 0.1, seed: int = 0):
        self.K = K
        self.d = d
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

        # Non-linear frequencies (one per arm) — control how wiggly each arm is.
        self.omegas = self.rng.uniform(1.5, 3.0, size=K)
        # Linear part: theta_k of dimension d-1 (excluding the nonlinear x[0]).
        thetas = self.rng.standard_normal((K, d - 1))
        thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)
        self.thetas = thetas

    def sample_context(self) -> np.ndarray:
        x = self.rng.standard_normal(self.d)
        x /= np.linalg.norm(x)
        return x

    def expected_reward(self, x: np.ndarray, k: int) -> float:
        nonlinear = np.sin(self.omegas[k] * x[0])
        linear = float(x[1:] @ self.thetas[k])
        return float(nonlinear + linear)

    def pull(self, x: np.ndarray, k: int) -> float:
        return self.expected_reward(x, k) + float(self.rng.normal(0, self.sigma))

    def best_action(self, x: np.ndarray) -> int:
        rewards = np.array([self.expected_reward(x, k) for k in range(self.K)])
        return int(np.argmax(rewards))

    def best_reward(self, x: np.ndarray) -> float:
        rewards = np.array([self.expected_reward(x, k) for k in range(self.K)])
        return float(np.max(rewards))

    def regret_of(self, x: np.ndarray, k: int) -> float:
        return self.best_reward(x) - self.expected_reward(x, k)
