"""Synthetic model of confidence calibration.

Companion to posts/04c-calibration.qmd.

Each item carries a latent *calibrated* logit l ~ Normal(mu, spread). The true
probability the model's answer is correct is q = sigmoid(l), and the outcome
is drawn correct ~ Bernoulli(q). The model then *reports* a confidence by
scaling that logit by a sharpness factor beta:

    report_logit = beta * l,    confidence = sigmoid(report_logit).

  - beta = 1  -> reported confidence == true P(correct): perfectly calibrated.
  - beta > 1  -> reported confidence is sharper than warranted: OVERCONFIDENT
                 (the failure mode of modern neural nets / LLMs).
  - beta < 1  -> reported confidence too timid: underconfident.

Because miscalibration here is exactly a logit rescaling, the textbook fix --
temperature scaling, dividing the logit by a learned T -- recovers perfect
calibration at T = beta. That is the point of the post: the same temperature
knob from decoding (Post 3a) reappears as the calibration knob.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps: float = 1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class CalibrationDataset:
    """A bank of (reported confidence, correctness) pairs with tunable sharpness.

    Parameters
    ----------
    n_items : int
        Number of predictions.
    sharpness : float
        beta above. >1 overconfident, <1 underconfident, 1 calibrated.
    mean_logit : float
        Centre of the latent logit distribution; controls base accuracy
        (accuracy ~ sigmoid(mean_logit) for small spread).
    spread : float
        Std of the latent logit distribution; controls the range of difficulty.
    seed : int
    """

    def __init__(
        self,
        n_items: int = 4000,
        sharpness: float = 1.0,
        mean_logit: float = 0.4,
        spread: float = 1.6,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.sharpness = float(sharpness)
        self.latent_logit = rng.normal(mean_logit, spread, size=n_items)
        self.q_true = sigmoid(self.latent_logit)               # true P(correct)
        self.correct = (rng.random(n_items) < self.q_true).astype(int)
        self.report_logit = self.sharpness * self.latent_logit
        self.confidence = sigmoid(self.report_logit)           # reported

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean())

    def split(self, frac: float = 0.5, seed: int = 0):
        """Split into (calibration, test) halves for fitting a temperature."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.correct))
        k = int(frac * len(idx))
        cal, test = idx[:k], idx[k:]
        return (
            (self.report_logit[cal], self.correct[cal]),
            (self.report_logit[test], self.correct[test]),
        )
