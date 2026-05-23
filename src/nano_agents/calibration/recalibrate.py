"""Post-hoc recalibration and the agentic payoff.

`temperature_scale` fits a single temperature T > 0 by minimizing NLL of
sigmoid(logit / T) on held-out data -- the standard one-parameter fix for
overconfident networks (Guo et al. 2017). With the synthetic model of
model.py the optimum lands at T = sharpness, recovering perfect calibration.

`gated_decision_accuracy` implements the agentic payoff that motivates the
whole post: an agent that takes a fallback action (retrieve, reflect, defer)
whenever its confidence is below a threshold. How well that gate works
depends entirely on whether confidence tracks correctness -- i.e. calibration.
"""

from __future__ import annotations

import numpy as np

from .metrics import nll
from .model import sigmoid


def apply_temperature(report_logit, T: float):
    """Recalibrate by dividing logits by temperature T, then sigmoid."""
    return sigmoid(np.asarray(report_logit, float) / T)


def temperature_scale(report_logit, correct, grid=None) -> float:
    """Fit T>0 minimizing NLL of sigmoid(logit/T). Returns the best T.

    A simple, robust grid search (the objective is 1-D and smooth).
    """
    report_logit = np.asarray(report_logit, float)
    correct = np.asarray(correct, float)
    if grid is None:
        grid = np.linspace(0.2, 5.0, 481)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        loss = nll(apply_temperature(report_logit, T), correct)
        if loss < best_nll:
            best_nll, best_T = loss, float(T)
    return best_T


def gated_decision_accuracy(
    confidence,
    correct,
    threshold: float,
    fallback_accuracy: float = 0.9,
):
    """Accuracy of a confidence-gated policy.

    For each item: if reported confidence >= threshold, KEEP the model's own
    answer (correct as given); otherwise take a FALLBACK action (retrieve /
    reflect / defer) that succeeds with probability `fallback_accuracy`,
    independent of the original answer.

    Returns
    -------
    accuracy : float
        Overall accuracy of the gated policy.
    fallback_rate : float
        Fraction of items routed to the fallback.
    """
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    keep = confidence >= threshold
    kept_correct = correct[keep].sum() if keep.any() else 0.0
    n_fallback = int((~keep).sum())
    fallback_correct = fallback_accuracy * n_fallback
    accuracy = (kept_correct + fallback_correct) / len(correct)
    return float(accuracy), float(n_fallback / len(correct))
