# -*- coding: utf-8 -*-
"""Canonical metric semantics for TACT.

The project has TWO prediction tasks and they must never share a metric name.

Task A — continuous thermal-risk estimation
    Target   : y(t+4) = clip(DHW(t+4)/10, 0, 1)   (continuous, [0, 1])
    Models   : PersA, ungated residual, TSAR, XGBoost
    Metrics  : mse_cont, rmse_cont, mae_cont, r2_cont, spearman
    "Brier" is NOT used for this task.

Task B — severe-event probability prediction
    Event    : 1[y(t+4) >= 0.8]
    Models   : TSCF only (sigmoid head trained with weighted BCE)
    Metrics  : brier_binary, logloss_binary, ece_probability, ROC/PR/CSI

Descriptive fallback for non-probability outputs
    ``score_reliability`` uses the same numeric formula as ECE
    (mean |observed event rate - mean score| over score bins) but is
    explicitly NOT probability calibration. It answers "does the score
    numerically track event frequency?", not "is P(event) calibrated?".

Binning convention (single, documented)
    10 equal-width bins over [0, 1]; the last bin INCLUDES 1.0.
    Predictions are clipped to [0, 1] only where stated; the fraction of
    out-of-unit predictions is always reported, never silently dropped.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

SEVERE_THRESHOLD = 0.8
N_BINS = 10

# model -> (task, is_probability)
MODEL_SPEC = {
    "A_PersA": {"task": "A_continuous", "is_probability": False},
    "B_ungated": {"task": "A_continuous", "is_probability": False},
    "C_anchored": {"task": "A_continuous", "is_probability": False},
    "D_TSCF": {"task": "B_probability", "is_probability": True},
    "XGB_FULL": {"task": "A_continuous", "is_probability": False},
    "XGB_THERMAL": {"task": "A_continuous", "is_probability": False},
    "XGB_NO_THERMAL": {"task": "A_continuous", "is_probability": False},
}


def binary_event(y, threshold: float = SEVERE_THRESHOLD) -> np.ndarray:
    """event = 1[y >= threshold] (float 0/1)."""
    return (np.asarray(y, dtype=float) >= threshold).astype(float)


def clip_unit(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


def finite_mask(*arrays) -> np.ndarray:
    """Element-wise validity mask: all arrays finite at that index.

    All metrics in this project are computed on the shared finite record set;
    callers must apply this mask before ``summarize`` (never impute).
    """
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(np.asarray(a, dtype=float))
    return m


# --------------------------------------------------------------------------
# Task A — continuous
# --------------------------------------------------------------------------
def mse_cont(pred, y, clip: bool = True) -> float:
    """Squared error against the continuous risk coordinate.

    ``clip=True`` reproduces the project's evaluation convention (predictions
    clipped to [0, 1]).  ``clip=False`` keeps the raw model output.  Both are
    reported by ``summarize`` so clipping is never hidden.
    """
    p = clip_unit(pred) if clip else np.asarray(pred, dtype=float)
    t = np.asarray(y, dtype=float)
    return float(np.mean((p - t) ** 2))


def rmse_cont(pred, y, clip: bool = True) -> float:
    return float(np.sqrt(mse_cont(pred, y, clip)))


def mae_cont(pred, y, clip: bool = True) -> float:
    p = clip_unit(pred) if clip else np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(p - np.asarray(y, dtype=float))))


def r2_cont(pred, y, clip: bool = True) -> float:
    p = clip_unit(pred) if clip else np.asarray(pred, dtype=float)
    t = np.asarray(y, dtype=float)
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def spearman_cont(pred, y) -> float:
    return float(spearmanr(np.asarray(pred, dtype=float),
                           np.asarray(y, dtype=float)).statistic)


# --------------------------------------------------------------------------
# Task B — probability
# --------------------------------------------------------------------------
def check_probability(p, where: str = "") -> np.ndarray:
    """Validate that ``p`` is a genuine probability array.

    Raises on NaN/inf or values outside [0, 1].  Callers that must clip must
    do so explicitly and record ``frac_out_of_unit``.
    """
    a = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{where}: non-finite probability values")
    lo, hi = float(a.min()), float(a.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"{where}: probability outside [0,1] (min={lo:.4f}, max={hi:.4f}); "
            "clip explicitly and record the out-of-unit fraction")
    return a


def _as_event(event) -> np.ndarray:
    """Require an explicit 0/1 label vector (never a continuous target)."""
    e = np.asarray(event, dtype=float)
    if not np.all(np.isin(e, (0.0, 1.0))):
        raise ValueError("event must be a 0/1 label array; use binary_event(y) first")
    return e


def brier_binary(prob, event, require_unit: bool = True) -> float:
    """Standard binary Brier score: mean((p - event)**2), event in {0,1}."""
    p = check_probability(prob) if require_unit else np.asarray(prob, dtype=float)
    e = _as_event(event)
    return float(np.mean((p - e) ** 2))


def logloss_binary(prob, event, eps: float = 1e-12) -> float:
    """Binary log loss (natural log)."""
    p = np.clip(check_probability(prob), eps, 1 - eps)
    e = _as_event(event)
    return float(-np.mean(e * np.log(p) + (1 - e) * np.log(1 - p)))


def bin_index(p, n_bins: int = N_BINS) -> np.ndarray:
    """10 equal-width bins over [0,1]; last bin includes 1.0."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return idx


def reliability_bins(pred, event, n_bins: int = N_BINS, clip: bool = True):
    """Per-bin n, mean score, observed event rate.

    Returns a list of dicts; empty bins are omitted (never imputed).
    """
    p = clip_unit(pred) if clip else np.asarray(pred, dtype=float)
    e = np.asarray(event, dtype=float)
    idx = bin_index(p, n_bins)
    out = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        out.append({"bin": b + 1, "n": int(m.sum()),
                    "mean_score": float(p[m].mean()),
                    "obs_event_rate": float(e[m].mean())})
    return out


def ece(pred, event, n_bins: int = N_BINS, clip: bool = True) -> float:
    """Weighted |observed event rate - mean score| over score bins.

    Numerically identical for probability and non-probability scores; the
    INTERPRETATION differs (see module docstring).  Use ``ece_probability``
    for TSCF and ``score_reliability`` otherwise.
    """
    p = clip_unit(pred) if clip else np.asarray(pred, dtype=float)
    e = np.asarray(event, dtype=float)
    n = len(p)
    idx = bin_index(p, n_bins)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / n) * abs(e[m].mean() - p[m].mean())
    return float(total)


def ece_probability(prob, event, n_bins: int = N_BINS) -> float:
    """Probability ECE — only for genuine event-probability outputs."""
    check_probability(prob, "ece_probability")
    return ece(prob, event, n_bins=n_bins, clip=False)


def score_reliability(pred, event, n_bins: int = N_BINS) -> float:
    """Descriptive score-frequency agreement — NOT probability calibration."""
    return ece(pred, event, n_bins=n_bins, clip=True)


# --------------------------------------------------------------------------
# Combined summary
# --------------------------------------------------------------------------
def summarize(pred, y, model: str, domain: str) -> dict:
    """Full, explicitly-typed metric block for one model x domain."""
    spec = MODEL_SPEC[model]
    p = np.asarray(pred, dtype=float)
    t = np.asarray(y, dtype=float)
    ev = binary_event(t)
    pc = clip_unit(p)
    frac_out = float(np.mean((p < 0.0) | (p > 1.0)))
    out = {
        "domain": domain,
        "model": model,
        "task": spec["task"],
        "is_probability_output": spec["is_probability"],
        "n": int(len(t)),
        "n_severe": int(ev.sum()),
        "prevalence": float(ev.mean()),
        "pred_min": float(p.min()),
        "pred_max": float(p.max()),
        "frac_out_of_unit": frac_out,
        # Task A (always reported)
        "mse_cont_raw": mse_cont(p, t, clip=False),
        "mse_cont": mse_cont(p, t, clip=True),
        "rmse_cont": rmse_cont(p, t, clip=True),
        "mae_cont": mae_cont(p, t, clip=True),
        "r2_cont": r2_cont(p, t, clip=True),
        "spearman_cont": spearman_cont(p, t),
        # ranking (valid for both tasks; computed on raw scores)
        "roc_severe": _auc(ev, p),
        "pr_severe": _ap(ev, p),
    }
    if spec["is_probability"]:
        out.update({
            "brier_binary": brier_binary(p, ev, require_unit=True),
            "logloss_binary": logloss_binary(p, ev),
            "ece_probability": ece_probability(p, ev),
            "probability_calibration_valid": True,
        })
    else:
        out.update({
            "brier_binary_descriptive": brier_binary(pc, ev, require_unit=True),
            "score_reliability": score_reliability(p, ev),
            "probability_calibration_valid": False,
            "note": ("continuous-output model: Brier/ECE are descriptive only "
                     "and are not evidence of probability calibration"),
        })
    out["reliability_bins"] = reliability_bins(p, ev)
    return out


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))
