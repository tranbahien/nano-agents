"""Tabular reward model trained to match preference data.

Given preference triples (context, winner, loser), fit a tabular reward
function r̂(c, k) by maximum likelihood under the Bradley-Terry model:

    P(y_w ≻ y_l | x) = sigmoid(r̂(x, y_w) - r̂(x, y_l))

The negative log-likelihood is

    NLL = -log sigmoid(r̂(x, y_w) - r̂(x, y_l))
        = log(1 + exp(-(r̂_w - r̂_l)))
        = softplus(-(r̂_w - r̂_l))

The gradient w.r.t. r̂(x, k) is straightforward:

    ∂NLL/∂r̂_w = -sigmoid(r̂_l - r̂_w)     (push w up)
    ∂NLL/∂r̂_l = +sigmoid(r̂_l - r̂_w)     (push l down)

Note: rewards are only identified up to an additive constant per context.
We don't normalize — just fit. This is also true of LLM reward models.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x):
    # Numerically stable sigmoid for scalar/array.
    return np.where(x >= 0,
                     1.0 / (1.0 + np.exp(-x)),
                     np.exp(x) / (1.0 + np.exp(x)))


class TabularRewardModel:
    """Reward model with one parameter per (context, completion) cell."""

    def __init__(self, n_contexts: int, n_completions: int):
        self.n_contexts = n_contexts
        self.n_completions = n_completions
        self.r = np.zeros((n_contexts, n_completions))

    def predict(self, context: int, completion: int) -> float:
        return float(self.r[context, completion])

    def step(self, preferences: list[tuple], lr: float):
        """One full-batch gradient step on Bradley-Terry NLL.

        Returns the mean NLL on the batch (for monitoring).
        """
        grad = np.zeros_like(self.r)
        total_loss = 0.0
        for (c, w, l) in preferences:
            diff = self.r[c, w] - self.r[c, l]
            sig = _sigmoid(diff)
            # NLL = -log sigmoid(diff) = log(1 + exp(-diff)).
            total_loss += float(-np.log(sig + 1e-30))
            # grad of NLL w.r.t. r[c, w] = -(1 - sig) = sig - 1
            # grad of NLL w.r.t. r[c, l] = (1 - sig) = -(sig - 1)
            grad[c, w] += sig - 1.0
            grad[c, l] += 1.0 - sig
        grad /= max(len(preferences), 1)
        self.r -= lr * grad
        return total_loss / max(len(preferences), 1)

    def fit(self, preferences: list[tuple], n_steps: int = 500,
            lr: float = 0.5) -> list[float]:
        """Run multiple gradient steps; return loss history."""
        losses = []
        for _ in range(n_steps):
            losses.append(self.step(preferences, lr))
        return losses
