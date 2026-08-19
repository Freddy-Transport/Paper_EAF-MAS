"""Metrics used by EAF-MAS v2 visualization tables and figures."""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(actual, pred) -> float:
    a, p = _arr(actual), _arr(pred)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual, pred) -> float:
    a, p = _arr(actual), _arr(pred)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def wape(actual, pred) -> float:
    a, p = _arr(actual), _arr(pred)
    denom = max(float(np.sum(np.abs(a))), EPS)
    return float(np.sum(np.abs(a - p)) / denom * 100.0)


def smape(actual, pred) -> float:
    a, p = _arr(actual), _arr(pred)
    denom = np.abs(a) + np.abs(p) + EPS
    return float(np.mean(2.0 * np.abs(a - p) / denom) * 100.0) if a.size else float("nan")


def metric_bundle(actual, pred) -> dict:
    return {
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "wape": wape(actual, pred),
        "smape": smape(actual, pred),
    }


def wape_gain(actual, raw, adjusted) -> float:
    return float(wape(actual, raw) - wape(actual, adjusted))


def relative_wape_gain(actual, raw, adjusted) -> float:
    raw_wape = wape(actual, raw)
    return float(wape_gain(actual, raw, adjusted) / max(raw_wape, EPS))


def beneficial_flag(gain: float, neutral_threshold: float = 0.01) -> bool:
    return bool(float(gain) > float(neutral_threshold))


def harmful_flag(gain: float, neutral_threshold: float = 0.01) -> bool:
    return bool(float(gain) < -float(neutral_threshold))


def neutral_flag(gain: float, neutral_threshold: float = 0.01) -> bool:
    return bool(abs(float(gain)) <= float(neutral_threshold))


def calibration_regret(raw_wape: float, adjusted_wape: float) -> float:
    return max(0.0, float(adjusted_wape) - float(raw_wape))


def correction_magnitude(raw, adjusted):
    return _arr(adjusted) - _arr(raw)


def relative_correction(raw, adjusted):
    r = _arr(raw)
    return correction_magnitude(raw, adjusted) / np.maximum(np.abs(r), EPS)


def bound_utilization(raw, adjusted, correction_bound: float):
    bound = max(float(correction_bound), EPS)
    return np.abs(relative_correction(raw, adjusted)) / bound
