"""Value baselines for policy gradient methods.

For tabular state spaces the baseline is just a learnable vector V[s]
updated by TD regression. For function-approximation settings this would
be a neural network with a regression loss.
"""

from __future__ import annotations

import numpy as np


class TabularBaseline:
    """A learnable value function V[s] for use as a policy gradient baseline.

    Updates: V[s] ← V[s] + lr * (target - V[s])
    where the target is typically the discounted return G_t (Monte Carlo)
    or a TD bootstrap r + gamma * V[s'].

    The implementation is intentionally minimal: just a vector + an update rule.
    The choice of *target* lives in the training loop, not here.
    """

    def __init__(self, nS: int, init_value: float = 0.0):
        self.nS = nS
        self.V = np.full(nS, float(init_value))

    def __call__(self, s_idx: int) -> float:
        return float(self.V[s_idx])

    def values(self, s_indices) -> np.ndarray:
        return self.V[np.asarray(s_indices)]

    def update(self, s_idx: int, target: float, lr: float) -> float:
        """Apply one regression step toward `target`. Return the TD error."""
        td = float(target) - float(self.V[s_idx])
        self.V[s_idx] += lr * td
        return td
