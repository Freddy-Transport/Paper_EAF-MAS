"""Calibration controller for safe bounded event corrections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Sequence

import numpy as np


def _finite_array(value, label):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _unit_scalar(value, label):
    array = _finite_array(value, label)
    if array.ndim != 0 or not 0 <= float(array) <= 1:
        raise ValueError(f"{label} must be a scalar in [0, 1]")
    return float(array)


def _names(value, label):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of unique nonempty names")
    try:
        names = list(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence of names") from exc
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must contain unique names")
    return names


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
            "source_validity_score": _unit_scalar(source_threshold, "source_threshold"),
            "geo_consistency_score": _unit_scalar(geo_threshold, "geo_threshold"),
            "temporal_alignment_score": _unit_scalar(temporal_threshold, "temporal_threshold"),
            "residual_support_score": _unit_scalar(residual_threshold, "residual_threshold"),
        }
        self.correction_bound = _unit_scalar(correction_bound, "correction_bound")

    def apply(
        self,
        raw_forecast: Sequence[Sequence[float]],
        proposed_adjusted: Sequence[Sequence[float]],
        correction: Sequence[Sequence[float]],
        channel_names: Sequence[str],
        adjusted_channel_candidates: Sequence[str],
        event_channel_mask,
        evidence_audit: Dict[str, Any],
        *,
        operating_mode: str = "EAF-MAS-C",
        has_admissible_events: bool = True,
        cell_classes=None,
        bound_matrix=None,
        rho_partial=None,
    ) -> Dict[str, Any]:
        """Apply bounded corrections to (channel, hour) cells only.

        The seven positional arguments are retained. ``proposed_adjusted`` is
        validated for shape/finiteness but deliberately ignored, including when
        inconsistent with raw * (1 + correction); legacy rounded proposals must
        never become an alternate path around the mask or bounds.

        Classes default to core. Core uses the full configured bound, secondary
        requires an explicit bound_matrix or rho_partial, an absolute relative
        bound in [0, correction_bound], not a fractional multiplier. Weak/conflict
        always use zero. An explicit matrix may tighten any bound
        but cannot exceed correction_bound. No score-to-class threshold is inferred.

        Each required audit score is a scalar or an array broadcastable exactly
        to raw.shape using NumPy rules (channel scores need shape (C, 1)). Optional
        evidence_audit['cell_scores'] uses the same keys and rules and adds gates
        without replacing required aggregate scores. Missing required scores
        fail closed. Returned correction is the effective, masked correction;
        corrected_mask marks actual numerical changes, not merely eligibility.
        Clipping alone does not reject a proposal. ``partial_apply`` means a mix
        of changed and unchanged eligible cells (candidate/event intersection
        with positive bounds, before auditing). Outside-support proposals and
        secondary-class membership alone do not imply partial application.
        No-event and explanation-only decisions are ``no_event_keep_raw`` and
        ``explain_only``; otherwise decisions are ``abstain`` or ``apply``.
        Correction statistics describe actual applied corrections; proposal
        magnitudes are retained separately in diagnostics.
        """
        raw = _finite_array(raw_forecast, "raw_forecast")
        if raw.ndim != 2 or 0 in raw.shape:
            raise ValueError("raw_forecast must be a nonempty 2D (channel, hour) array")
        if (raw < 0).any():
            raise ValueError("raw_forecast must be nonnegative")
        proposed = _finite_array(proposed_adjusted, "proposed_adjusted")
        corr = _finite_array(correction, "correction")
        mask_values = _finite_array(event_channel_mask, "event_channel_mask")
        for label, array in (("proposed_adjusted", proposed), ("correction", corr),
                             ("event_channel_mask", mask_values)):
            if array.ndim != 2 or array.shape != raw.shape:
                raise ValueError(f"{label} must be 2D with shape {raw.shape}")
        if not np.isin(mask_values, [0, 1]).all():
            raise ValueError("event_channel_mask must contain only boolean or 0/1 values")
        event_mask = mask_values.astype(bool)
        names = _names(channel_names, "channel_names")
        candidates = _names(
            [] if adjusted_channel_candidates is None else adjusted_channel_candidates,
            "adjusted_channel_candidates",
        )
        if len(names) != raw.shape[0]:
            raise ValueError("channel_names must match the channel dimension of raw_forecast")
        if not set(candidates).issubset(names):
            raise ValueError("adjusted_channel_candidates must be a subset of channel_names")
        if operating_mode not in ("EAF-MAS-C", "EAF-MAS-X"):
            raise ValueError("operating_mode must be EAF-MAS-C or EAF-MAS-X")
        if not isinstance(has_admissible_events, (bool, np.bool_)):
            raise ValueError("has_admissible_events must be boolean")

        try:
            classes = (np.full(raw.shape, "core") if cell_classes is None
                       else np.asarray(cell_classes))
        except (TypeError, ValueError) as exc:
            raise ValueError("cell_classes must be a 2D array matching raw_forecast") from exc
        if classes.shape != raw.shape or not np.isin(
            classes, ["core", "secondary", "weak", "conflict"]
        ).all():
            raise ValueError("cell_classes must match raw shape and use core/secondary/weak/conflict")
        secondary = classes == "secondary"
        rho = None if rho_partial is None else _unit_scalar(rho_partial, "rho_partial")
        if rho is not None and rho > self.correction_bound and not np.isclose(
            rho, self.correction_bound, rtol=1e-7, atol=0,
        ):
            raise ValueError("rho_partial must not exceed correction_bound")
        if secondary.any() and bound_matrix is None and rho is None:
            raise ValueError("secondary cells require explicit bound_matrix or rho_partial")
        bounds = np.full(raw.shape, self.correction_bound)
        if rho is not None:
            bounds[secondary] = min(rho, self.correction_bound)
        if bound_matrix is not None:
            supplied_bounds = _finite_array(bound_matrix, "bound_matrix")
            if supplied_bounds.shape != raw.shape:
                raise ValueError(f"bound_matrix must be 2D with shape {raw.shape}")
            too_large = (supplied_bounds > self.correction_bound) & ~np.isclose(
                supplied_bounds, self.correction_bound, rtol=1e-7, atol=0,
            )
            if ((supplied_bounds < 0) | too_large).any():
                raise ValueError("bound_matrix must be between zero and correction_bound")
            # A float32 representation of the bound may round slightly upward.
            bounds = np.minimum(bounds, supplied_bounds)
        bounds[np.isin(classes, ["weak", "conflict"])] = 0

        audit = {} if evidence_audit is None else evidence_audit
        if not isinstance(audit, Mapping):
            raise ValueError("evidence_audit must be a mapping or None")
        failures = []
        audit_mask = np.ones(raw.shape, dtype=bool)
        audit_scores = {}
        candidate_mask = np.asarray([name in candidates for name in names])[:, None]
        eligible_mask = event_mask & candidate_mask & (bounds > 0)

        def score_gate(value, label, threshold):
            scores = _finite_array(value, label)
            if ((scores < 0) | (scores > 1)).any():
                raise ValueError(f"{label} must be in [0, 1]")
            try:
                scores = np.broadcast_to(scores, raw.shape)
            except ValueError as exc:
                raise ValueError(f"{label} must broadcast exactly to {raw.shape}") from exc
            passed = scores >= threshold
            if (eligible_mask & ~passed).any():
                failures.append(label)
            return passed

        for key, threshold in self.thresholds.items():
            if audit.get(key) is None:
                failures.append(key)
                audit_mask[:] = False
                audit_scores[key] = None
            else:
                audit_mask &= score_gate(audit[key], key, threshold)
                audit_scores[key] = _finite_array(audit[key], key).tolist()
        cell_scores = audit.get("cell_scores", {})
        if not isinstance(cell_scores, Mapping):
            raise ValueError("cell_scores must be a mapping of audit score names to arrays")
        for key, value in cell_scores.items():
            if key not in self.thresholds:
                raise ValueError(f"unknown cell_scores key: {key}")
            audit_mask &= score_gate(value, f"cell_scores.{key}", self.thresholds[key])
        severe = _names(audit.get("severe_conflict_flags") or [], "severe_conflict_flags")
        if severe:
            failures.append("severe_conflict_flags:" + ",".join(severe))
            audit_mask[:] = False
        if candidates and not (event_mask & candidate_mask).any():
            failures.append("adjusted channels outside event-station-channel mask")
        mask = eligible_mask & audit_mask
        if not has_admissible_events:
            decision = "no_event_keep_raw"
            failures.append("no admissible events")
            mask[:] = False
        elif operating_mode == "EAF-MAS-X":
            decision = "explain_only"
            failures.append("explanation-only operating mode")
            mask[:] = False
        else:
            decision = "abstain"

        clipped = np.clip(corr, -bounds, bounds)
        effective = np.where(mask, clipped, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            final = raw * (1.0 + effective)
        if not np.isfinite(final).all():
            raise ValueError("bounded correction produced a nonfinite final forecast")
        corrected_mask = final != raw
        effective = np.where(corrected_mask, effective, 0.0)
        controller_allowed = bool(corrected_mask.any())
        requested_mask = eligible_mask & (corr != 0) & (raw > 0)
        if controller_allowed:
            partial = bool((eligible_mask & ~corrected_mask).any())
            decision = "partial_apply" if partial else "apply"
        proposal_max_abs = float(np.max(np.abs(corr)))
        # Scale before summation to avoid overflow for large finite proposals.
        proposal_mean_abs = (float(np.mean(np.abs(corr) / proposal_max_abs) * proposal_max_abs)
                             if proposal_max_abs else 0.0)
        applied_max_abs = float(np.max(np.abs(effective)))
        return {
            "controller_allowed": bool(controller_allowed),
            "abstain": not bool(controller_allowed),
            "final_adjusted_forecast": final.astype(float).tolist(),
            "adjusted_channels": [name for name in candidates if corrected_mask[names.index(name)].any()],
            "abstention_reason": "" if controller_allowed else "; ".join(failures or ["no adjusted channel candidate passed controller"]),
            "correction_stats": {
                "max_abs_correction": applied_max_abs,
                "mean_abs_correction": float(np.mean(np.abs(effective))),
                "active_correction_cells": int(np.count_nonzero(corrected_mask)),
                "correction_bound": self.correction_bound,
                "max_abs_applied_correction": applied_max_abs,
            },
            "thresholds": dict(self.thresholds),
            "audit_scores": audit_scores,
            "included_units": list(audit.get("included_units") or []),
            "excluded_units": list(audit.get("excluded_units") or []),
            "decision": decision,
            "mask": mask.tolist(),
            "bound_matrix": bounds.tolist(),
            "correction": effective.tolist(),
            "corrected_mask": corrected_mask.tolist(),
            "diagnostics": {
                "proposed_adjusted_ignored": True,
                "proposal_max_abs_correction": proposal_max_abs,
                "proposal_mean_abs_correction": proposal_mean_abs,
                "audit_failures": failures,
                "requested_effect_cells": int(np.count_nonzero(requested_mask)),
                "eligible_cells": int(np.count_nonzero(eligible_mask)),
                "permitted_cells": int(np.count_nonzero(mask)),
                "corrected_cells": int(np.count_nonzero(corrected_mask)),
                "clipped_cells": int(np.count_nonzero(mask & (clipped != corr))),
            },
        }
