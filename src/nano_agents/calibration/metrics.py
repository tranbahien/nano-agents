"""Calibration metrics: reliability curve, ECE, Brier score, NLL.

All take a reported `confidence` array in [0, 1] and a binary `correct`
array. The reliability curve and ECE are the workhorses; Brier and NLL are
proper scoring rules that fold calibration and sharpness together.
"""

from __future__ import annotations

import numpy as np


def reliability_curve(confidence, correct, n_bins: int = 10, adaptive: bool = False):
    """Bin predictions by confidence and return per-bin (conf, acc, count).

    Parameters
    ----------
    adaptive : bool
        If False, equal-width bins on [0, 1]. If True, equal-mass bins
        (each holds ~the same number of items), which reduces estimator
        variance when confidence piles up near 1.

    Returns
    -------
    bin_conf, bin_acc, bin_count : np.ndarray
        Mean confidence, accuracy, and count per non-empty bin.
    """
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    if adaptive:
        edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    confs, accs, counts = [], [], []
    for lo, hi, i in zip(edges[:-1], edges[1:], range(len(edges) - 1)):
        last = i == len(edges) - 2
        mask = (confidence >= lo) & (confidence <= hi if last else confidence < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        confs.append(confidence[mask].mean())
        accs.append(correct[mask].mean())
        counts.append(n)
    return np.array(confs), np.array(accs), np.array(counts)


def ece(confidence, correct, n_bins: int = 10, adaptive: bool = False) -> float:
    """Expected Calibration Error: bin-size-weighted mean |accuracy - confidence|."""
    conf, acc, count = reliability_curve(confidence, correct, n_bins, adaptive)
    if len(count) == 0:
        return 0.0
    weights = count / count.sum()
    return float(np.sum(weights * np.abs(acc - conf)))


def max_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """Worst-case per-bin calibration gap (MCE)."""
    conf, acc, _ = reliability_curve(confidence, correct, n_bins)
    if len(conf) == 0:
        return 0.0
    return float(np.max(np.abs(acc - conf)))


def brier_score(confidence, correct) -> float:
    """Mean squared error between confidence and outcome (proper scoring rule)."""
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    return float(np.mean((confidence - correct) ** 2))


def nll(confidence, correct, eps: float = 1e-12) -> float:
    """Negative log-likelihood (log loss) of the reported confidences."""
    confidence = np.clip(np.asarray(confidence, float), eps, 1.0 - eps)
    correct = np.asarray(correct, float)
    return float(-np.mean(correct * np.log(confidence)
                          + (1 - correct) * np.log(1 - confidence)))
