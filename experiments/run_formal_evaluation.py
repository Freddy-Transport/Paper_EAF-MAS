#!/usr/bin/env python3
"""Evaluate private, provenance-locked forecasts through the shared audited path."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.audit_inputs import load_bundle, new_output_dir, digest, verify_adapter_provenance, configuration_digest
from agents.event_adapter import load_adapter
from agents.formal_workflow import FEATURE_SCHEMA, FORMAL_FEATURES, run_numerical, adapter_proposal
from agents.skillbench import LocalJSONGenerator, evaluate_modes, aggregate


def wape(actual, forecast, mask):
    denominator = np.abs(actual[mask]).sum()
    return float(100 * np.abs(actual[mask] - forecast[mask]).sum() / denominator) if denominator > 0 else None


def validate_outcomes(record, shape):
    y = np.asarray(record["outcomes"], dtype=float)
    if y.shape != shape or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Missing/invalid retrospective outcomes")
    return y


def ablation_rows(record, context, state, adapter, config, index):
    raw = context["raw"]
    proposal = adapter_proposal(adapter, context["features"], config.get("device", "cpu"))
    global_proposal = proposal  # Identical g_phi proposals across all controller variants.
    full = np.asarray(state["correction"])
    count = int(np.asarray(state["corrected_mask"]).sum())
    rng = np.random.default_rng(config["random_seed"] + index)
    candidates = np.flatnonzero((raw * global_proposal != 0).ravel())
    if len(candidates) < count:
        raise ValueError("Random gate cannot match full controller's actual intervention count")
    random_mask = np.zeros(raw.size, dtype=bool)
    random_mask[rng.choice(candidates, size=count, replace=False)] = True
    random_mask = random_mask.reshape(raw.shape)
    bound = config["rho_full"]
    thresholds = {"source_validity_score":config["audit_thresholds"]["source_threshold"],
                  "geo_consistency_score":config["audit_thresholds"]["geo_threshold"],
                  "temporal_alignment_score":config["audit_thresholds"]["temporal_threshold"],
                  "residual_support_score":config["audit_thresholds"]["residual_threshold"]}
    audit_pass = np.ones(raw.shape, dtype=bool)
    non_residual_pass = np.ones(raw.shape, dtype=bool)
    for key, threshold in thresholds.items():
        audit_pass &= context["audit"][key] >= threshold
        if key != "residual_support_score": non_residual_pass &= context["audit"][key] >= threshold
    audit_pass &= ~context["conflicts"]
    non_residual_pass &= ~context["conflicts"]
    score_bounds = np.where(context["score"] >= config["core_threshold"], bound,
                           np.where(context["score"] >= config["secondary_threshold"], config["rho_partial"], 0))
    corrections = {"raw": np.zeros_like(raw), "full_controller": full,
        "no_event_gate": np.where(audit_pass, np.clip(proposal, -score_bounds, score_bounds), 0),
        "no_residual_support": np.where(context["matched"] & non_residual_pass,
                                        np.clip(proposal, -score_bounds, score_bounds), 0),
        "no_audit_guard": np.where(context["mask"], np.clip(proposal, -context["bounds"], context["bounds"]), 0),
        "no_bound": np.where(np.asarray(state["mask"]), proposal, 0),
        "always_apply": np.clip(global_proposal, -bound, bound),
        "random_gate": np.where(random_mask, np.clip(global_proposal, -bound, bound), 0)}
    y = validate_outcomes(record, raw.shape)
    # Evaluation masks come from the locked population, not the tested gate.
    active = np.asarray(record["event_active_mask"])
    if active.shape != raw.shape or not np.isin(active, [0, 1]).all():
        raise ValueError("Explicit fixed event-active evaluation mask required")
    active = active.astype(bool)
    venue = np.zeros_like(active)
    for name in record["venue_channels"]:
        if name not in record["channel_names"]:
            raise ValueError("Unknown venue channel")
        venue[record["channel_names"].index(name)] = True
    base_event, base_non = wape(y, raw, active), wape(y, raw, ~active)
    rows = []
    for name, delta in corrections.items():
        final = raw * (1 + delta)
        event, non = wape(y, final, active), wape(y, final, ~active)
        rows.append(dict(request_id=record["request_id"], variant=name,
            model_identity=record["model_identity"], groups=record["groups"],
            WAPE=wape(y, final, np.ones_like(active)), venue_associated_WAPE=wape(y, final, venue),
            event_active_WAPE=event, raw_event_active_WAPE=base_event,
            gain=base_event-event if base_event is not None and event is not None else None,
            non_event_degradation=non-base_non if non is not None and base_non is not None else None,
            corrected_cells=int((final != raw).sum()), total_cells=raw.size,
            max_relative_correction=float(np.abs(delta).max())))
    return rows


def summarize(rows):
    output = []
    def summary(items, label):
        gains = [r["gain"] for r in items if r["gain"] is not None]
        means = {key: float(np.mean([r[key] for r in items if r[key] is not None]))
                 if any(r[key] is not None for r in items) else None
                 for key in ("WAPE", "venue_associated_WAPE", "event_active_WAPE", "raw_event_active_WAPE", "gain", "non_event_degradation")}
        return dict(group=label, n=len(items), valid_gain_n=len(gains), **means,
             improved_forecasts_pct=100*sum(g > 0 for g in gains)/len(gains) if gains else None,
             degraded_forecasts_pct=100*sum(g < 0 for g in gains)/len(gains) if gains else None,
             worst_regret=max([max(-g, 0) for g in gains], default=None),
             corrected_cells=sum(r["corrected_cells"] for r in items),
             max_relative_correction=max(r["max_relative_correction"] for r in items))
    for variant in dict.fromkeys(r["variant"] for r in rows):
        group = [r for r in rows if r["variant"] == variant]
        output.append(dict(variant=variant, **summary(group, "all")))
        if variant == "full_controller":
            labels = {(k, v) for r in group for k, v in r["groups"].items()}
            for key, value in sorted(labels):
                selected = [r for r in group if r["groups"].get(key) == value]
                output.append(dict(variant=variant, group_type=key, **summary(selected, value)))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--explanations", action="store_true")
    p.add_argument("--library")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model")
    args = p.parse_args()
    manifest, records, memory = load_bundle(args.manifest, args.split)
    config = json.loads(Path(args.config).read_text())
    adapter, adapter_config = load_adapter(args.adapter, config.get("device", "cpu"))
    if adapter_config.get("feature_schema") != FEATURE_SCHEMA or adapter_config.get("feature_names") != FORMAL_FEATURES:
        raise ValueError("Residual feature schema changed; refit adapter before formal evaluation")
    verify_adapter_provenance(adapter_config, manifest, config)
    library, generator = [], None
    if args.explanations:
        if not args.library or not args.model:
            p.error("Actual explanation evaluation requires a fixed library and model backend")
        library_record = json.loads(Path(args.library).read_text())
        if library_record.get("split") != "val" or library_record.get("input_manifest_digest") != digest(manifest):
            raise ValueError("Fixed library must come from validation replay of the locked population")
        if library_record.get("configuration_digest") != configuration_digest(config) or library_record.get("adapter_sha256") != hashlib.sha256((Path(args.adapter)/"event_adapter.pt").read_bytes()).hexdigest():
            raise ValueError("Skill library was promoted under a different numerical configuration")
        library = library_record["promoted_skills"]
        generator = LocalJSONGenerator(args.base_url, args.model)
    out = new_output_dir(args.output)
    (out/"run_manifest.json").write_text(json.dumps(dict(inputs=manifest, config=config,
        adapter_config=adapter_config, checkpoint_sha256=hashlib.sha256((Path(args.adapter)/"event_adapter.pt").read_bytes()).hexdigest(), library_digest=digest(library),
        model_identity=manifest["model_identity"], input_manifest_digest=digest(manifest)), indent=2))
    metrics, explanations = [], []
    for index, record in enumerate(records):
        context, state = run_numerical(record, memory, config, adapter, adapter_config)
        metrics.extend(ablation_rows(record, context, state, adapter, config, index))
        with (out/"numerical_traces.jsonl").open("a") as f:
            f.write(json.dumps(dict(request_id=record["request_id"], **state), allow_nan=False)+"\n")
        if generator:
            batch = evaluate_modes(record, state, context["historical_cases"], library, generator)
            explanations.extend(batch)
            with (out/"explanations.jsonl").open("a") as f:
                for row in batch:
                    f.write(json.dumps(row, allow_nan=False)+"\n")
    (out/"calibration_rows.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out/"calibration_summary.json").write_text(json.dumps(summarize(metrics), indent=2, allow_nan=False))
    (out/"skillbench_summary.json").write_text(json.dumps(aggregate(explanations), indent=2, allow_nan=False))
    print("Completed; all generated artifacts remain in the private output directory.")


if __name__ == "__main__":
    main()
