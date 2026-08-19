#!/usr/bin/env python3
"""Select diverse forecast windows for paper explanation case studies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


CASE_TYPES = ["event_intervention", "weak_evidence_failure", "historical_memory_case", "safe_abstention"]
MIN_EVENT_WINDOW_WAPE_GAIN = 0.05
MIN_ADJUSTED_CHANNEL_WAPE_GAIN = 0.10


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _as_bool(row: dict, key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return str(value or "").lower() in {"1", "true", "yes", "y"}


def _anchor(row: dict) -> str:
    return str(row.get("anchor") or row.get("date") or row.get("target_date") or row.get("window_id") or "")


def _row_key(row: dict) -> tuple[str, str]:
    return (_anchor(row), str(row.get("event_type") or row.get("venue") or row.get("location") or "unknown"))


def _classify(row: dict) -> List[str]:
    has_event = _as_bool(row, "has_major_event") or str(row.get("impact_tier", "")).upper() in {"A", "B"}
    delta = _as_float(row, "wape_delta_raw_minus_adjusted", _as_float(row, "raw_wape") - _as_float(row, "adjusted_wape"))
    accepted = int(_as_float(row, "accepted_evidence_count", 0.0))
    residual_cases = int(_as_float(row, "residual_memory_cases", _as_float(row, "historical_event_cases", 0.0)))
    abstain = _as_bool(row, "abstain") or str(row.get("adjusted_channels") or "").strip() in {"", "[]", "(none)"}
    labels: List[str] = []
    if has_event and delta > 0 and not abstain and _passes_event_intervention_threshold(row):
        labels.append("event_intervention")
    if has_event and (delta < 0 or accepted == 0):
        labels.append("weak_evidence_failure")
    if (not has_event) or abstain:
        labels.append("safe_abstention")
    if residual_cases > 0:
        labels.append("historical_memory_case")
    return labels


def _passes_event_intervention_threshold(row: dict) -> bool:
    """Require visible localized improvement before using a case as the main intervention example."""
    local_keys = ("event_window_wape_gain", "adjusted_channel_wape_gain", "included_unit_wape_gain")
    has_local_gain = any(str(row.get(key, "")).strip() != "" for key in local_keys)
    event_gain = _as_float(row, "event_window_wape_gain", 0.0)
    adjusted_gain = _as_float(row, "adjusted_channel_wape_gain", 0.0)
    included_gain = _as_float(row, "included_unit_wape_gain", 0.0)
    if has_local_gain:
        return (
            event_gain >= MIN_EVENT_WINDOW_WAPE_GAIN
            or adjusted_gain >= MIN_ADJUSTED_CHANNEL_WAPE_GAIN
            or included_gain >= MIN_ADJUSTED_CHANNEL_WAPE_GAIN
        )
    delta = _as_float(row, "wape_delta_raw_minus_adjusted", _as_float(row, "raw_wape") - _as_float(row, "adjusted_wape"))
    return delta >= MIN_EVENT_WINDOW_WAPE_GAIN


def _intervention_score(row: dict) -> float:
    return max(
        _as_float(row, "event_window_wape_gain", 0.0),
        _as_float(row, "adjusted_channel_wape_gain", 0.0),
        _as_float(row, "included_unit_wape_gain", 0.0),
        _as_float(row, "wape_delta_raw_minus_adjusted", _as_float(row, "raw_wape") - _as_float(row, "adjusted_wape")),
    )


def select_windows(rows: Sequence[dict], max_windows: int = 4, default_anchor: str = "2023-06-19 10:00:00") -> List[dict]:
    """Select one deterministic representative for each paper case type."""
    enriched = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["_idx"] = i
        item["_anchor"] = _anchor(item)
        item["_labels"] = _classify(item)
        item["_delta"] = _as_float(item, "wape_delta_raw_minus_adjusted", _as_float(item, "raw_wape") - _as_float(item, "adjusted_wape"))
        item["_residual_cases"] = int(_as_float(item, "residual_memory_cases", _as_float(item, "historical_event_cases", 0.0)))
        enriched.append(item)

    selected: List[dict] = []
    seen_keys = set()

    def add(label: str, candidates: Iterable[dict]) -> None:
        if len(selected) >= max_windows:
            return
        if label == "event_intervention":
            ranked = sorted(
                candidates,
                key=lambda r: (
                    -_intervention_score(r),
                    0 if r.get("_anchor") == default_anchor else 1,
                    -int(r.get("_residual_cases") or 0),
                    int(r.get("_idx") or 0),
                ),
            )
        else:
            ranked = sorted(
                candidates,
                key=lambda r: (
                    0 if r.get("_anchor") == default_anchor else 1,
                    -abs(float(r.get("_delta") or 0.0)),
                    -int(r.get("_residual_cases") or 0),
                    int(r.get("_idx") or 0),
                ),
            )
        for row in ranked:
            key = _row_key(row)
            if key in seen_keys:
                continue
            out = {k: v for k, v in row.items() if not k.startswith("_")}
            out["case_type"] = label
            out["selection_reason"] = _selection_reason(label, row)
            selected.append(out)
            seen_keys.add(key)
            return

    for label in CASE_TYPES:
        add(label, (row for row in enriched if label in row.get("_labels", [])))

    if len(selected) < max_windows:
        for row in sorted(enriched, key=lambda r: (int(r.get("_idx") or 0))):
            if len(selected) >= max_windows:
                break
            key = _row_key(row)
            if key in seen_keys:
                continue
            out = {k: v for k, v in row.items() if not k.startswith("_")}
            out["case_type"] = "diversity_fill"
            out["selection_reason"] = "Fills the multi-window panel after typed cases are exhausted."
            selected.append(out)
            seen_keys.add(key)
    return selected[:max_windows]


def _selection_reason(label: str, row: dict) -> str:
    if label == "event_intervention":
        return (
            "Major event window with visible localized raw-minus-adjusted WAPE gain "
            f"(event_window={_as_float(row, 'event_window_wape_gain', 0.0):.3f}, "
            f"adjusted_channel={_as_float(row, 'adjusted_channel_wape_gain', 0.0):.3f})."
        )
    if label == "weak_evidence_failure":
        return "Major event window with weak external evidence or harmful correction."
    if label == "safe_abstention":
        return "No major event or no validated correction, illustrating LP-MOMENT preservation."
    if label == "historical_memory_case":
        return "Window has train/validation residual-memory cases for explanation."
    return "Selected for deterministic diversity."


def _rows_from_prediction_json(paths: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        evidence = payload.get("evidence") or {}
        decision = payload.get("decision") or {}
        request = payload.get("request") or {}
        metrics = payload.get("metrics") or {}
        raw_metrics = payload.get("raw_metrics") or {}
        quality = payload.get("explanation_quality") or {}
        rows.append(
            {
                "date": request.get("date"),
                "mode": request.get("mode"),
                "station_scope": request.get("station_scope"),
                "has_major_event": evidence.get("has_major_event"),
                "event_type": (evidence.get("structured_events") or [{}])[0].get("event_type", "unknown"),
                "impact_tier": (evidence.get("structured_events") or [{}])[0].get("impact_tier", "unknown"),
                "raw_wape": raw_metrics.get("wape", metrics.get("raw_wape", "")),
                "adjusted_wape": metrics.get("wape", ""),
                "wape_delta_raw_minus_adjusted": quality.get("wape_delta_raw_minus_adjusted", 0.0),
                "accepted_evidence_count": quality.get("accepted_evidence_count", 0),
                "residual_memory_cases": len(evidence.get("historical_event_cases") or []) + len(evidence.get("local_residual_cases") or []),
                "abstain": decision.get("abstain"),
                "adjusted_channels": decision.get("adjusted_channels"),
                "prediction_json": path,
            }
        )
    return rows


def _rows_from_csv(path: str | None) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if _looks_like_full_metric_rows(rows):
        return _paired_rows_from_full_metrics(rows)
    return rows


def _looks_like_full_metric_rows(rows: Sequence[dict]) -> bool:
    if not rows:
        return False
    required = {"mode", "date", "wape", "event_wape"}
    return required.issubset(set(rows[0].keys())) and any(str(row.get("mode")) == "numerical_only" for row in rows)


def _paired_rows_from_full_metrics(rows: Sequence[dict]) -> List[dict]:
    baseline: Dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if str(row.get("mode")) != "numerical_only":
            continue
        key = (str(row.get("split") or ""), str(row.get("scope") or ""), str(row.get("date") or ""))
        baseline[key] = row

    paired: List[dict] = []
    for row in rows:
        mode = str(row.get("mode") or "")
        if mode not in {"event_adapter_frozen_moment"}:
            continue
        key = (str(row.get("split") or ""), str(row.get("scope") or ""), str(row.get("date") or ""))
        raw = baseline.get(key)
        if not raw:
            continue
        event_n = int(_as_float(row, "event_n", 0.0))
        raw_wape = _as_float(raw, "wape")
        adjusted_wape = _as_float(row, "wape")
        raw_event_wape = _as_float(raw, "event_wape")
        adjusted_event_wape = _as_float(row, "event_wape")
        out = {
            "split": row.get("split"),
            "anchor": row.get("anchor"),
            "date": row.get("date"),
            "mode": mode,
            "station_scope": row.get("scope"),
            "has_major_event": event_n > 0,
            "impact_tier": "A" if event_n > 0 else "none",
            "event_type": "event_window",
            "raw_wape": raw_wape,
            "adjusted_wape": adjusted_wape,
            "wape_delta_raw_minus_adjusted": raw_wape - adjusted_wape,
            "event_window_wape_gain": raw_event_wape - adjusted_event_wape,
            "adjusted_channel_wape_gain": "",
            "included_unit_wape_gain": "",
            "event_n": event_n,
            "accepted_evidence_count": 1 if event_n > 0 else 0,
            "residual_memory_cases": 0,
            "abstain": False,
            "adjusted_channels": "(unknown)",
        }
        paired.append(out)
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_rows_csv", default=None)
    parser.add_argument("--prediction_json", action="append", default=[])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_windows", type=int, default=4)
    parser.add_argument("--default_anchor", default="2023-06-19 10:00:00")
    args = parser.parse_args()

    rows = _rows_from_csv(args.metric_rows_csv) + _rows_from_prediction_json(args.prediction_json)
    selected = select_windows(rows, max_windows=args.max_windows, default_anchor=args.default_anchor)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"selected_windows": selected, "n_input_rows": len(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out), "selected": len(selected), "n_input_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
