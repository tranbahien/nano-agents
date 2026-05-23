"""Tests for the calibration subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.calibration import (
    CalibrationDataset,
    apply_temperature,
    brier_score,
    ece,
    gated_decision_accuracy,
    logit,
    nll,
    reliability_curve,
    sigmoid,
    temperature_scale,
)


def test_sigmoid_logit_inverse():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(sigmoid(logit(p)), p, atol=1e-9)


def test_calibrated_model_has_low_ece():
    ds = CalibrationDataset(n_items=8000, sharpness=1.0, seed=0)
    assert ece(ds.confidence, ds.correct, n_bins=15) < 0.03


def test_overconfident_model_has_high_ece():
    ds = CalibrationDataset(n_items=8000, sharpness=2.5, seed=0)
    # Overconfident: confidence systematically exceeds accuracy.
    conf, acc, _ = reliability_curve(ds.confidence, ds.correct, n_bins=15)
    high = conf > 0.5
    assert np.mean(acc[high] < conf[high]) > 0.7      # accuracy below confidence
    assert ece(ds.confidence, ds.correct, n_bins=15) > 0.05


def test_temperature_recovers_sharpness():
    beta = 2.0
    ds = CalibrationDataset(n_items=20000, sharpness=beta, seed=1)
    (cal_logit, cal_y), _ = ds.split(frac=0.5, seed=2)
    T = temperature_scale(cal_logit, cal_y)
    assert abs(T - beta) < 0.25       # fitted T ~ true sharpness


def test_temperature_scaling_lowers_ece():
    ds = CalibrationDataset(n_items=20000, sharpness=2.5, seed=3)
    (cal_logit, cal_y), (test_logit, test_y) = ds.split(frac=0.5, seed=4)
    raw = ece(sigmoid_safe(test_logit), test_y, n_bins=15)
    T = temperature_scale(cal_logit, cal_y)
    fixed = ece(apply_temperature(test_logit, T), test_y, n_bins=15)
    assert fixed < raw
    assert fixed < 0.03


def sigmoid_safe(x):
    return sigmoid(x)


def test_brier_and_nll_reward_calibration():
    good = CalibrationDataset(n_items=8000, sharpness=1.0, seed=0)
    bad = CalibrationDataset(n_items=8000, sharpness=3.0, seed=0)
    # Same underlying correctness; calibrated confidences score better.
    assert nll(good.confidence, good.correct) < nll(bad.confidence, bad.correct)
    assert brier_score(good.confidence, good.correct) <= \
        brier_score(bad.confidence, bad.correct) + 1e-9


def test_ece_bounds():
    ds = CalibrationDataset(n_items=2000, sharpness=2.0, seed=0)
    e = ece(ds.confidence, ds.correct)
    assert 0.0 <= e <= 1.0


def test_adaptive_binning_runs():
    ds = CalibrationDataset(n_items=3000, sharpness=2.0, seed=0)
    conf, acc, count = reliability_curve(ds.confidence, ds.correct,
                                         n_bins=10, adaptive=True)
    assert count.sum() == len(ds.correct)
    assert len(conf) == len(acc) == len(count)


def test_gating_better_with_calibration_at_fixed_threshold():
    # Temperature scaling preserves the *ranking* of items by confidence, so it
    # does not change the oracle-threshold accuracy. Its value is making a
    # FIXED, semantically-meaningful threshold behave: at tau=0.8 an
    # overconfident model keeps many wrong answers (conf > 0.8 but wrong),
    # while a calibrated model routes them to the (better) fallback.
    fb, tau = 0.88, 0.8
    cal = CalibrationDataset(n_items=8000, sharpness=1.0, seed=5)
    over = CalibrationDataset(n_items=8000, sharpness=3.0, seed=5)
    acc_cal, _ = gated_decision_accuracy(cal.confidence, cal.correct, tau, fb)
    acc_over, _ = gated_decision_accuracy(over.confidence, over.correct, tau, fb)
    assert acc_cal > acc_over + 0.01


def test_temperature_preserves_ranking():
    # Monotonic rescaling: order of items by confidence is unchanged.
    ds = CalibrationDataset(n_items=2000, sharpness=2.5, seed=0)
    fixed = apply_temperature(ds.report_logit, 2.5)
    assert np.array_equal(np.argsort(ds.confidence), np.argsort(fixed))
