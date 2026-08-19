"""Layered knowledge-base helpers for paper experiments."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


SENSITIVE_KEYS = {"api_key", "openai_api_key", "dashscope_api_key", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"}


def _redact(value):
    if isinstance(value, dict):
        return {k: ("<redacted>" if k in SENSITIVE_KEYS else _redact(v)) for k, v in value.items() if k not in SENSITIVE_KEYS}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str) and value.startswith("sk-"):
        return "<redacted>"
    return value


def write_kb_manifest(
    kb_dir: str | Path,
    kb_name: str,
    source_files: Sequence[str],
    split_policy: str,
    date_range: dict,
    doc_count: int,
    uses_qwenplus: bool = False,
    embedding_model: str = "",
    extra: dict | None = None,
) -> Path:
    """Write a redacted manifest for a layered KB directory."""
    root = Path(kb_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "kb_name": kb_name,
        "source_files": list(source_files),
        "split_policy": split_policy,
        "date_range": date_range,
        "doc_count": int(doc_count),
        "uses_qwenplus": bool(uses_qwenplus),
        "embedding_model": embedding_model,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload["extra"] = _redact(extra)
    path = root / "manifest.json"
    path.write_text(json.dumps(_redact(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def build_true_residual_cases(
    prediction_rows: Iterable[dict],
    allowed_splits: set[str] | None = None,
    residual_bound: float = 0.20,
) -> Tuple[List[str], List[dict]]:
    """Build residual-case documents from real LP-MOMENT predictions."""
    allowed_splits = allowed_splits or {"train", "val"}
    groups: dict[tuple, list] = defaultdict(list)
    for row in prediction_rows:
        split = str(row.get("split", "train"))
        if split not in allowed_splits:
            continue
        pred = max(_float(row, "moment_pred"), 1.0)
        actual = _float(row, "actual")
        correction = max(-residual_bound, min(residual_bound, (actual - pred) / pred))
        key = (
            str(row.get("event_type", "unknown") or "unknown"),
            str(row.get("rank_group", "all") or "all"),
            str(row.get("day_type", "unknown") or "unknown"),
            str(row.get("impact_tier", "B") or "B"),
        )
        groups[key].append({**row, "correction": correction})

    docs: List[str] = []
    metas: List[dict] = []
    for (event_type, rank_group, day_type, tier), rows in groups.items():
        corrections = [float(r["correction"]) for r in rows]
        if not corrections:
            continue
        med = statistics.median(corrections)
        p25 = sorted(corrections)[int(0.25 * (len(corrections) - 1))]
        p75 = sorted(corrections)[int(0.75 * (len(corrections) - 1))]
        doc = (
            f"[True LP-MOMENT Residual RAG] Event type: {event_type} | "
            f"Station rank: {rank_group} | Day: {day_type} | Tier: {tier}\n"
            f"Using real LP-MOMENT predictions on train/validation rows, "
            f"actual ridership was median {med:+.1%} relative to LP-MOMENT "
            f"(p25={p25:+.1%}, p75={p75:+.1%}, N={len(corrections)}). "
            "Use this as bounded last-mile correction context, not as ground truth for test windows."
        )
        meta = {
            "event_type": event_type,
            "rank_group": rank_group,
            "day_type": day_type,
            "impact_tier": tier,
            "n_instances": len(corrections),
            "median_correction": round(med, 4),
            "p25_correction": round(p25, 4),
            "p75_correction": round(p75, 4),
            "source": "true_lp_moment_residual",
            "doc_type": "model_correction",
        }
        docs.append(doc)
        metas.append(meta)
    return docs, metas


def write_layered_kb_manifests(root: str | Path, uses_qwenplus: bool = False) -> dict:
    """Create the four paper KB layer directories and minimal manifests."""
    root = Path(root)
    layers = {
        "structured_event_kb": {"doc_count": 0, "split_policy": "all_forecast_time_event_records"},
        "true_residual_case_kb": {"doc_count": 0, "split_policy": "train_val_only"},
        "external_evidence_kb": {"doc_count": 0, "split_policy": "forecast_time_cached_evidence"},
        "skill_memory": {"doc_count": 0, "split_policy": "train_val_learn_test_read_only"},
    }
    out = {}
    for name, cfg in layers.items():
        layer = root / name
        path = write_kb_manifest(
            layer,
            kb_name=name,
            source_files=[],
            split_policy=cfg["split_policy"],
            date_range={},
            doc_count=cfg["doc_count"],
            uses_qwenplus=uses_qwenplus if name == "external_evidence_kb" else False,
        )
        out[name] = str(path)
    return out
