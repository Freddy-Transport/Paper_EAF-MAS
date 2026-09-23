"""Shared audited numerical path. Skill selection is downstream of this module."""
from __future__ import annotations

from copy import deepcopy
import re
import numpy as np
import pandas as pd
import torch

from agents.audit_inputs import assert_forecast_time, digest
from agents.calibration_controller import CalibrationController
from agents.event_adapter import FEATURE_NAMES, build_event_feature_cube
from agents.forecast_evidence_auditor import ForecastEvidenceAuditor

FEATURE_SCHEMA = "event_residual_support.v2"
FORMAL_FEATURES = list(FEATURE_NAMES) + ["residual_support", "residual_median", "residual_log_count"]


def validate_memory(cases, origin):
    origin = pd.Timestamp(origin)
    if pd.isna(origin):
        raise ValueError("Invalid forecast origin")
    seen = set()
    result = []
    for case in cases:
        required = {"case_id", "split", "support_end", "channel_name", "event_type",
                    "median_correction", "n_eff", "iqr", "prediction_source_sha256"}
        if not required.issubset(case):
            raise ValueError("Historical residual case lacks provenance or required statistics")
        if case["case_id"] in seen:
            raise ValueError("Duplicate historical residual case")
        seen.add(case["case_id"])
        if case["split"] not in {"train", "val"}:
            raise ValueError("Numerical memory must originate from train/validation")
        if not isinstance(case["prediction_source_sha256"], str) or not re.fullmatch(r"[a-fA-F0-9]{64}", case["prediction_source_sha256"]):
            raise ValueError("Residual prediction-source hash required")
        if not np.isfinite([case["median_correction"], case["iqr"], case["n_eff"]]).all():
            raise ValueError("Invalid residual statistics")
        if case["n_eff"] <= 0 or case["iqr"] < 0:
            raise ValueError("Invalid residual support population")
        # Earlier origins may only use already observed historical windows.
        end = pd.Timestamp(case["support_end"])
        if pd.isna(end):
            raise ValueError("Historical support end is unknown")
        if end >= origin:
            continue
        row = deepcopy(case)
        row["median_lp_moment_correction"] = row["median_correction"]
        result.append(row)
    return result


def numerical_context(record, memory, config):
    """Construct true case-backed support and a cell-level gate without skill input."""
    raw = np.asarray(record["raw_forecast"], dtype=np.float64)
    names, times = record["channel_names"], record["timestamps"]
    if raw.shape != (len(names), len(times)) or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid raw forecast tensor")
    if len(set(names)) != len(names) or not times:
        raise ValueError("Unique channels and nonempty hourly horizon required")
    dates = pd.to_datetime(times)
    if dates[0] != pd.Timestamp(record["forecast_origin"]) or any(
        dates[i+1] - dates[i] != pd.Timedelta(hours=1) for i in range(len(dates)-1)
    ):
        raise ValueError("Forecast timestamps must begin at origin and advance hourly")
    events = deepcopy(record["events"])
    assert_forecast_time(events)
    assert_forecast_time(record["evidence_sources"])
    assert_forecast_time(record["model_assisted_summaries"])
    cases = validate_memory(memory, record["forecast_origin"])
    cube = build_event_feature_cube(raw, times, names, events, record["channel_meta"])
    matched = cube[..., 0] > 0
    support = np.zeros(raw.shape)
    median = np.zeros(raw.shape)
    count = np.zeros(raw.shape)
    case_ids = [[[] for _ in times] for _ in names]
    audit = {k: np.zeros(raw.shape) for k in ["source_validity_score", "geo_consistency_score",
             "temporal_alignment_score", "semantic_consistency_score", "residual_support_score"]}
    auditor = ForecastEvidenceAuditor(**config["audit_thresholds"])
    conflicts = np.zeros(raw.shape, dtype=bool)
    # Per-event cubes preserve event/channel/time linkage in overlapping windows.
    event_cubes = [build_event_feature_cube(raw, times, names, [e], record["channel_meta"])[..., 0] > 0
                   for e in events]
    for i, name in enumerate(names):
        cache = {}
        for j in np.flatnonzero(matched[i]):
            key = tuple(k for k, mask in enumerate(event_cubes) if mask[i, j])
            if key not in cache:
                relevant = [events[k] for k in key]
                types = {str(e.get("event_type") or e.get("event_category") or "").lower() for e in relevant}
                selected = [c for c in cases if c["channel_name"] == name and str(c["event_type"]).lower() in types]
                result = auditor.audit(record["forecast_origin"], len(times), relevant,
                                       record["evidence_sources"], record["model_assisted_summaries"],
                                       [name], [name], selected, [])
                cache[key] = result, selected
            result, selected = cache[key]
            for key in audit:
                audit[key][i, j] = result[key]
            conflicts[i, j] = bool(result["severe_conflict_flags"])
            support[i, j] = result["residual_support_score"]
            if selected:
                median[i, j] = np.median([c["median_correction"] for c in selected])
                count[i, j] = sum(c["n_eff"] for c in selected)
            case_ids[i][j] = [c["case_id"] for c in selected]
    weights = config["gate_weights"]
    if set(weights) != set(audit) or not np.isfinite(list(weights.values())).all():
        raise ValueError("All gate score weights must be explicitly configured")
    if any(v < 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Gate weights must be nonnegative and sum to one")
    score = sum(audit[k] * float(v) for k, v in weights.items())
    low, high = float(config["secondary_threshold"]), float(config["core_threshold"])
    if not 0 <= low <= high <= 1:
        raise ValueError("Invalid configured gate thresholds")
    eligible = matched & (support >= config["audit_thresholds"]["residual_threshold"]) & (count > 0)
    classes = np.full(raw.shape, "weak", dtype=object)
    classes[eligible & (score >= low)] = "secondary"
    classes[eligible & (score >= high)] = "core"
    classes[conflicts] = "conflict"
    mask = np.isin(classes, ["core", "secondary"])
    full, partial = float(config["rho_full"]), float(config["rho_partial"])
    if not 0 <= partial <= full <= 1:
        raise ValueError("Bounds must satisfy 0 <= partial <= full <= 1")
    bounds = np.where(classes == "core", full, np.where(classes == "secondary", partial, 0))
    features = np.concatenate([cube, np.stack([support, median, np.log1p(count)], axis=-1)], axis=-1)
    return dict(raw=raw, matched=matched, mask=mask, classes=classes, bounds=bounds,
                score=score, audit=audit, conflicts=conflicts, features=features,
                R_num=dict(case_ids=case_ids, support=support.tolist(), median=median.tolist(),
                           count=count.tolist()), historical_cases=cases)


def calibrate(record, context, proposal, config, *, mode="EAF-MAS-C"):
    controller = CalibrationController(**config["audit_thresholds"], correction_bound=config["rho_full"])
    audit = dict(context["audit"])
    # Conflict cells already have a zero bound; unrelated cells remain eligible.
    result = controller.apply(context["raw"], context["raw"] * (1 + proposal), proposal,
                              record["channel_names"], record["channel_names"], context["mask"], audit,
                              cell_classes=context["classes"], bound_matrix=context["bounds"],
                              operating_mode=mode, has_admissible_events=bool(record["events"]))
    result["R_num"] = context["R_num"]
    result["gate_trace"] = dict(all_cells=context["raw"].size,
        horizon_event_candidates=context["raw"].size if record["events"] else 0,
        event_station_matched=int(context["matched"].sum()),
        residual_supported=int(((np.asarray(context["R_num"]["count"]) > 0) &
             (np.asarray(context["R_num"]["support"]) >= config["audit_thresholds"]["residual_threshold"])).sum()),
        gate_eligible=int(context["mask"].sum()), controller_allowed=int(np.asarray(result["mask"]).sum()),
        corrected=int(np.asarray(result["corrected_mask"]).sum()))
    result["numerical_state_digest"] = digest({"raw": record["raw_forecast"], "R_num": context["R_num"],
         "gate": context["mask"].tolist(), "proposal": np.asarray(proposal).tolist(),
         "final": result["final_adjusted_forecast"], "decision": result["decision"]})
    return result


def run_numerical(record, memory, config, adapter, adapter_config):
    if adapter_config.get("feature_schema") != FEATURE_SCHEMA or adapter_config.get("feature_names") != FORMAL_FEATURES:
        raise ValueError("Adapter schema mismatch; explicit residual-MLP refit required")
    if adapter_config.get("output_transform") != "identity":
        raise ValueError("Formal residual proposal requires the refitted identity-output MLP")
    context = numerical_context(record, memory, config)
    proposal = adapter_proposal(adapter, context["features"], config.get("device", "cpu"))
    return context, calibrate(record, context, proposal, config, mode=config.get("operating_mode", "EAF-MAS-C"))


def adapter_proposal(adapter, features, device="cpu"):
    """Return g_phi before controller clipping, shared across ablation variants."""
    if getattr(adapter, "output_transform", None) != "identity":
        raise ValueError("Legacy adapter output parameterization requires an explicit refit")
    with torch.no_grad():
        proposal = adapter.to(device).eval()(torch.as_tensor(features, dtype=torch.float32, device=device))
    result = proposal.cpu().numpy().astype(float)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite adapter proposal")
    return result
