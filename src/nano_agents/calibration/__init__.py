"""Calibration and uncertainty.

Companion to posts/04c-calibration.qmd.
"""

from .metrics import (
    brier_score,
    ece,
    max_calibration_error,
    nll,
    reliability_curve,
)
from .model import CalibrationDataset, logit, sigmoid
from .recalibrate import (
    apply_temperature,
    gated_decision_accuracy,
    temperature_scale,
)

__all__ = [
    "CalibrationDataset",
    "sigmoid",
    "logit",
    "reliability_curve",
    "ece",
    "max_calibration_error",
    "brier_score",
    "nll",
    "temperature_scale",
    "apply_temperature",
    "gated_decision_accuracy",
]
