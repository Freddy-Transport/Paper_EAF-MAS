"""Private, hash-verified inputs for reproducible evaluations.

No dataset, prediction, fallback forecast, or result constant is distributed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

SCHEMA = "eafmas.audit-inputs.v1"
FORBIDDEN = {"actual", "actuals", "ground_truth", "realized_ridership", "post_hoc_metrics",
             "posthoc_metrics", "actual_wape", "test_errors", "target", "targets", "outcomes",
             "metrics", "raw_metrics", "labels", "future_actuals"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def assert_forecast_time(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN:
                raise ValueError(f"Retrospective field at inference boundary: {key}")
            assert_forecast_time(child)
    elif isinstance(value, list):
        for child in value:
            assert_forecast_time(child)


def load_bundle(manifest_path, split, *, allow_synthetic=False):
    """Return private records after checking file hashes and declared population.

The model identity is retained verbatim. A legacy LP cache is never renamed.
Hashes establish artifact identity, not proof that historical collection was causal.
"""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or split not in {"train", "val", "test"}:
        raise ValueError("Explicit versioned manifest and train/val/test split required")
    if manifest.get("synthetic", True) and not allow_synthetic:
        raise ValueError("Formal evaluation requires real artifacts; synthetic fallback disabled")
    if not manifest.get("model_identity") or not manifest.get("split_definition"):
        raise ValueError("Missing model provenance or split definition")
    if not manifest.get("source_revision"):
        raise ValueError("Missing input-producing source revision")
    loaded, population = {}, {}
    all_ids, all_origins = set(), set()
    for key in ("train", "val", "test", "residual_memory"):
        entry = manifest["files"][key]
        source = (path.parent / entry["path"]).resolve()
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Input hash mismatch: {key}")
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(rows) != entry["count"]:
            raise ValueError(f"Population mismatch: {key}")
        if key != "residual_memory":
            boundary = manifest["split_definition"][key]
            start, end = datetime.fromisoformat(boundary["start"]), datetime.fromisoformat(boundary["end"])
            if start >= end:
                raise ValueError("Split boundary must be an ordered half-open interval")
            population[key] = (start, end)
            for row in rows:
                origin = datetime.fromisoformat(row["forecast_origin"])
                times = [datetime.fromisoformat(t) for t in row["timestamps"]]
                if row["split"] != key or not times or any(not start <= t < end for t in times) or not start <= origin < end:
                    raise ValueError("Forecast window disagrees with locked split boundaries")
                if row["request_id"] in all_ids or origin in all_origins:
                    raise ValueError("Duplicate forecast request/origin across splits")
                all_ids.add(row["request_id"]); all_origins.add(origin)
                if row["model_identity"] != manifest["model_identity"]:
                    raise ValueError("Forecast model identity disagrees with manifest")
        if key in {split, "residual_memory"}:
            loaded[key] = rows
    if not population["train"][1] <= population["val"][0] or not population["val"][1] <= population["test"][0]:
        raise ValueError("Chronological splits overlap")
    ids = [str(row["request_id"]) for row in loaded[split]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate forecast request IDs")
    for row in loaded[split]:
        if row["split"] != split:
            raise ValueError("Record split disagrees with manifest")
        if row["model_identity"] != manifest["model_identity"]:
            raise ValueError("Forecast model identity disagrees with manifest")
    return manifest, loaded[split], loaded["residual_memory"]


def configuration_digest(config):
    return digest({k:v for k,v in config.items() if k != "device"})


def verify_adapter_provenance(adapter_config, manifest, config):
    training = adapter_config.get("training_manifest", {})
    if training.get("input_manifest_digest") != digest(manifest):
        raise ValueError("Adapter was not fitted against this locked input manifest")
    if training.get("configuration_digest") != configuration_digest(config):
        raise ValueError("Feature/gate configuration differs from the adapter fitting run")
    if adapter_config.get("model_identity") != manifest["model_identity"]:
        raise ValueError("Adapter/backbone provenance mismatch")


def new_output_dir(path):
    """Prevent overwriting existing experiment artifacts."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    return p
