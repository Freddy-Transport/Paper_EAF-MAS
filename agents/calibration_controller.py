"""Calibration controller for safe bounded event corrections."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


class CalibrationController:
    def __init__(
        self,
        source_threshold: float = 0.60,
        geo_threshold: float = 0.60,
        temporal_threshold: float = 0.70,
        residual_threshold: float = 0.50,
        correction_bound: float = 0.05,
    ):
        self.thresholds = {
            "source_validity_score": float(source_threshold),
            "geo_consistency_score": float(geo_threshold),
            "temporal_alignment_score": float(temporal_threshold),
            "residual_support_score": float(residual_threshold),
        }
        self.correction_bound = float(correction_bound)

    def apply(
        self,
        raw_forecast: Sequence[Sequence[float]],
        proposed_adjusted: Sequence[Sequence[float]],
        correction: Sequence[Sequence[float]],
        channel_names: Sequence[str],
        adjusted_channel_candidates: Sequence[str],
        event_channel_mask,
        evidence_audit: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw = np.asarray(raw_forecast, dtype=np.float32)
        proposed = np.asarray(proposed_adjusted, dtype=np.float32)
        corr = np.asarray(correction, dtype=np.float32)
        names = list(channel_names)
        candidates = list(adjusted_channel_candidates or [])
        mask = np.asarray(event_channel_mask, dtype=bool)
        if mask.ndim == 2:
            mask_by_channel = mask.any(axis=1)
        else:
            mask_by_channel = mask.reshape(-1)
        if mask_by_channel.shape[0] != len(names):
            mask_by_channel = np.zeros(len(names), dtype=bool)
        failures = []
        for key, threshold in self.thresholds.items():
            if float(evidence_audit.get(key, 0.0) or 0.0) < threshold:
                failures.append(key)
        severe = list(evidence_audit.get("severe_conflict_flags") or [])
        if severe:
            failures.append("severe_conflict_flags:" + ",".join(severe))
        max_abs = float(np.max(np.abs(corr))) if corr.size else 0.0
        mean_abs = float(np.mean(np.abs(corr))) if corr.size else 0.0
        active_cells = int(np.sum(np.abs(corr) > 1e-8)) if corr.size else 0
        if max_abs > self.correction_bound + 1e-6:
            failures.append("correction magnitude exceeds predefined bound")
        outside = [ch for ch in candidates if ch in names and not bool(mask_by_channel[names.index(ch)])]
        missing = [ch for ch in candidates if ch not in names]
        if outside or missing:
            failures.append("adjusted channels outside event-station-channel mask")
        controller_allowed = not failures and bool(candidates)
        final = proposed if controller_allowed else raw
        return {
            "controller_allowed": bool(controller_allowed),
            "abstain": not bool(controller_allowed),
            "final_adjusted_forecast": final.astype(float).tolist(),
            "adjusted_channels": candidates if controller_allowed else [],
            "abstention_reason": "" if controller_allowed else "; ".join(failures or ["no adjusted channel candidate passed controller"]),
            "correction_stats": {
                "max_abs_correction": max_abs,
                "mean_abs_correction": mean_abs,
                "active_correction_cells": active_cells,
                "correction_bound": self.correction_bound,
            },
            "thresholds": dict(self.thresholds),
            "audit_scores": {k: evidence_audit.get(k) for k in self.thresholds},
            "included_units": list(evidence_audit.get("included_units") or []),
            "excluded_units": list(evidence_audit.get("excluded_units") or []),
        }
