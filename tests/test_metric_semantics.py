# -*- coding: utf-8 -*-
"""Unit tests for the canonical metric semantics (metric_semantics.py).

Run:  python -m pytest tests/test_metric_semantics.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "08_infrastructure"))

import metric_semantics as M  # noqa: E402


# ---------------- Task B: binary probability metrics ----------------
def test_binary_brier_known_value():
    p = np.array([1.0, 0.0, 0.5])
    e = np.array([1.0, 0.0, 0.0])
    assert M.brier_binary(p, e) == pytest.approx((0 + 0 + 0.25) / 3)


def test_logloss_known_value():
    p = np.array([0.9, 0.2])
    e = np.array([1.0, 0.0])
    expected = -0.5 * (math.log(0.9) + math.log(0.8))
    assert M.logloss_binary(p, e) == pytest.approx(expected)


def test_brier_rejects_non_binary_event():
    with pytest.raises(ValueError):
        M.brier_binary(np.array([0.5]), np.array([0.73]))


# ---------------- Task A: continuous metrics ----------------
def test_mse_cont_known_value():
    pred = np.array([0.2, 0.8])
    y = np.array([0.0, 1.0])
    assert M.mse_cont(pred, y) == pytest.approx(0.04)
    assert M.rmse_cont(pred, y) == pytest.approx(0.2)


def test_mse_cont_clip_vs_raw_differ():
    pred = np.array([1.3, 0.5])
    y = np.array([1.0, 0.5])
    assert M.mse_cont(pred, y, clip=True) == pytest.approx(0.0)
    assert M.mse_cont(pred, y, clip=False) == pytest.approx(0.045)


# ---------------- ECE / binning ----------------
def test_ece_last_bin_includes_one():
    bins = M.reliability_bins(np.array([1.0]), np.array([1.0]))
    assert len(bins) == 1
    assert bins[0]["bin"] == 10 and bins[0]["n"] == 1


def test_ece_known_value():
    # bin 1: score ~0.05, event 0 -> 0.05*0.5 ; bin 10: score 1.0, event 1 -> 0
    p = np.array([0.05, 1.0])
    e = np.array([0.0, 1.0])
    assert M.ece(p, e) == pytest.approx(0.5 * 0.05)


def test_ece_probability_equals_score_reliability_for_inrange():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    e = (rng.uniform(0, 1, 500) < p).astype(float)
    assert M.ece_probability(p, e) == pytest.approx(M.score_reliability(p, e))


# ---------------- validity handling ----------------
def test_probability_out_of_range_raises():
    with pytest.raises(ValueError):
        M.check_probability(np.array([0.5, 1.2]))
    with pytest.raises(ValueError):
        M.brier_binary(np.array([0.5, 1.2]), np.array([0.0, 1.0]))


def test_nan_probability_raises():
    with pytest.raises(ValueError):
        M.check_probability(np.array([0.5, np.nan]))


def test_finite_mask_filters_rows():
    m = M.finite_mask(np.array([1.0, np.nan, 3.0]),
                      np.array([0.0, 1.0, 2.0]))
    assert m.tolist() == [True, False, True]


# ---------------- applicability flags ----------------
def test_non_probability_model_is_flagged():
    y = np.array([0.0, 0.9, 0.1])
    pred = np.array([0.1, 0.8, 0.2])
    a = M.summarize(pred, y, "A_PersA", "SCS")
    d = M.summarize(pred, y, "D_TSCF", "SCS")
    assert a["probability_calibration_valid"] is False
    assert "brier_binary" not in a and "score_reliability" in a
    assert d["probability_calibration_valid"] is True
    assert "brier_binary" in d and "ece_probability" in d
