#!/usr/bin/env python3
"""Screen noisy NYC permit events into major subway-impact event candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_screening.llm import IsolatedTransformersRunner, RunnerUnavailable, build_llm_prompt, make_runner
from event_screening.screener import deduplicate_physical_events, final_keep_for_modeling, rule_prefilter_event
from event_screening.traffic import compute_traffic_evidence

TRAIN_ROWS = 12 * 30 * 24


def load_events(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("events", payload) if isinstance(payload, dict) else payload
    return list(rows)


def ensure_run_dir(run_dir: Path) -> None:
    for sub in ["figures", "logs", "reports"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def should_write_data_output(runner_used: str, requested: bool, allow_heuristic: bool = False) -> bool:
    if not requested:
        return False
    if runner_used in {"hf_api", "isolated_transformers"}:
        return True
    return bool(allow_heuristic and runner_used == "heuristic")


def affected_channels_for_event(event, traffic_columns: Sequence[str]) -> List[str]:
    channels = [ch for ch in event.affected_channels if ch in traffic_columns]
    return channels or [ch for ch in event.affected_channels if ch]


def rank_events_for_llm(physical, traffic_df: pd.DataFrame, traffic_columns: Sequence[str], max_physical_events: int | None) -> List[dict]:
    enriched = []
    for event in physical:
        channels = affected_channels_for_event(event, traffic_columns)
        evidence = compute_traffic_evidence(traffic_df, channels, event.event_time, horizon_hours=4, train_end_idx=TRAIN_ROWS)
        rule = rule_prefilter_event(event, evidence)
        anomaly = 0.0
        if evidence.get("available") and not evidence.get("low_volume_warning"):
            anomaly = min(1.0, abs(float(evidence.get("z_score", 0.0))) / 5.0)
        combined = max(float(rule.rule_score), anomaly)
        enriched.append({"event": event, "traffic_evidence": evidence, "rule": rule, "combined_score": combined})
    candidates = [row for row in enriched if row["rule"].high_priority or row["combined_score"] >= 0.55]
    candidates.sort(key=lambda row: (row["combined_score"], row["event"].duplicate_count), reverse=True)
    if max_physical_events:
        candidates = candidates[:max_physical_events]
    return candidates


def row_payload(item: dict, decision, runner_name: str) -> dict:
    event = item["event"]
    evidence = item["traffic_evidence"]
    rule = item["rule"]
    keep = final_keep_for_modeling(decision)
    return {
        "event_id": event.event_id,
        "title": event.title,
        "event_time": event.event_time,
        "date": event.date,
        "event_type": event.event_type,
        "location": event.location,
        "duplicate_count": event.duplicate_count,
        "affected_channels": "|".join(event.affected_channels),
        "station_complex_ids": "|".join(event.station_complex_ids),
        "min_distance_m": event.min_distance_m,
        "rule_score": rule.rule_score,
        "rule_category": rule.category,
        "rule_high_priority": rule.high_priority,
        "rule_reason": rule.reason,
        "traffic_available": evidence.get("available", False),
        "traffic_z_score": evidence.get("z_score"),
        "traffic_delta_pct": evidence.get("delta_pct"),
        "traffic_actual_sum": evidence.get("actual_sum"),
        "traffic_low_volume_warning": evidence.get("low_volume_warning"),
        "llm_runner": runner_name,
        "is_major_event": decision.is_major_event,
        "major_event_type": decision.major_event_type,
        "crowd_scale": decision.crowd_scale,
        "transit_impact_likelihood": decision.transit_impact_likelihood,
        "expected_direction": decision.expected_direction,
        "affected_scope": decision.affected_scope,
        "traffic_evidence_used": decision.traffic_evidence_used,
        "keep_for_modeling": keep,
        "needs_review": decision.needs_review,
        "llm_reason": decision.reason,
    }


def filtered_event_json(rows: List[dict], physical_by_id: Dict[str, object]) -> dict:
    physical_events = []
    channel_events = []
    for row in rows:
        if not row["keep_for_modeling"]:
            continue
        event = physical_by_id[row["event_id"]]
        base = {
            "event_id": row["event_id"],
            "title": row["title"],
            "event_time": row["event_time"],
            "location": row["location"],
            "event_type": row["event_type"],
            "major_event_type": row["major_event_type"],
            "event_category": row["major_event_type"],
            "crowd_scale": row["crowd_scale"],
            "transit_impact_likelihood": row["transit_impact_likelihood"],
            "expected_direction": row["expected_direction"],
            "affected_scope": row["affected_scope"],
            "duplicate_count": event.duplicate_count,
            "impact_tier": "A" if row["transit_impact_likelihood"] >= 0.75 else "B",
            "impact_confidence": "high" if row["transit_impact_likelihood"] >= 0.75 else "medium",
            "should_adjust_forecast": True,
            "traffic_evidence": {
                "z_score": row["traffic_z_score"],
                "delta_pct": row["traffic_delta_pct"],
                "actual_sum": row["traffic_actual_sum"],
                "low_volume_warning": row["traffic_low_volume_warning"],
            },
            "reason": row["llm_reason"],
        }
        physical_events.append({**base, "affected_channels": event.affected_channels, "station_complex_ids": event.station_complex_ids})
        channels = event.affected_channels or [None]
        for i, channel in enumerate(channels):
            sid = event.station_complex_ids[i] if i < len(event.station_complex_ids) else (event.station_complex_ids[0] if event.station_complex_ids else None)
            channel_events.append({
                **base,
                "event_id": f"{row['event_id']}::{i}",
                "physical_event_id": row["event_id"],
                "channel_name": channel,
                "station_complex_id": sid,
                "content": f"major_event_type={row['major_event_type']}; crowd_scale={row['crowd_scale']}; likelihood={row['transit_impact_likelihood']}; physical_event_id={row['event_id']}",
            })
    return {"events": channel_events, "physical_events": physical_events, "count": len(channel_events), "physical_count": len(physical_events)}


def make_figures(score_df: pd.DataFrame, traffic_df: pd.DataFrame, run_dir: Path, top_n: int = 5) -> None:
    if score_df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kept = score_df.sort_values(["keep_for_modeling", "transit_impact_likelihood", "rule_score"], ascending=False).head(top_n)
    traffic_df = traffic_df.copy()
    traffic_df["date"] = pd.to_datetime(traffic_df["date"])
    for i, row in enumerate(kept.itertuples(), 1):
        channels = str(row.affected_channels).split("|")[:4]
        channels = [ch for ch in channels if ch in traffic_df.columns]
        if not channels:
            continue
        event_ts = pd.Timestamp(row.event_time)
        start = event_ts - pd.Timedelta(hours=72)
        end = event_ts + pd.Timedelta(hours=72)
        sub = traffic_df[(traffic_df["date"] >= start) & (traffic_df["date"] <= end)]
        if sub.empty:
            continue
        plt.figure(figsize=(10, 4))
        for ch in channels:
            plt.plot(sub["date"], sub[ch], label=ch[:35])
        plt.axvline(event_ts, color="red", linestyle="--", label="event")
        plt.title(f"{row.title[:80]} | likelihood={row.transit_impact_likelihood:.2f}")
        plt.legend(fontsize=7)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / f"top_event_{i}_{row.event_id}.png", dpi=150)
        plt.close()
    # Score histogram
    plt.figure(figsize=(8, 4))
    plt.hist(score_df["transit_impact_likelihood"].dropna(), bins=20)
    plt.xlabel("transit_impact_likelihood")
    plt.ylabel("physical events")
    plt.title("LLM/heuristic major-event likelihood distribution")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "likelihood_histogram.png", dpi=150)
    plt.close()


def _simple_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No candidate rows."
    rows = [columns]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.3f}"
            text = str(val).replace("|", "/").replace("\n", " ")[:120]
            vals.append(text)
        rows.append(vals)
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, sep] + body)


def write_report(run_dir: Path, raw_count: int, physical_count: int, candidates_count: int, score_df: pd.DataFrame, runner_name: str, model_name: str) -> None:
    kept = int(score_df["keep_for_modeling"].sum()) if not score_df.empty else 0
    review = int(score_df["needs_review"].sum()) if not score_df.empty else 0
    type_counts = Counter(score_df["major_event_type"]) if not score_df.empty else Counter()
    top_rows = score_df.sort_values(["keep_for_modeling", "transit_impact_likelihood", "rule_score"], ascending=False).head(20)
    columns = ["title", "event_time", "major_event_type", "crowd_scale", "transit_impact_likelihood", "keep_for_modeling", "rule_reason", "llm_reason"]
    top_table = _simple_markdown_table(top_rows, columns)
    report = f"""# Major Event Screening Report

## Run

- run_dir: `{run_dir}`
- model_name: `{model_name}`
- runner: `{runner_name}`
- raw channel-events: {raw_count}
- deduplicated physical events: {physical_count}
- rule/anomaly candidates sent to LLM runner: {candidates_count}
- keep_for_modeling: {kept}
- needs_review: {review}

## Major event type counts

```json
{json.dumps(dict(type_counts), ensure_ascii=False, indent=2)}
```

## Top 20 retained/candidate events

{top_table}

## Interpretation

The pipeline first removes duplicate station-event copies, then uses rule and traffic evidence to avoid sending all permit records to the model. `Special Event` is not sufficient for retention; ordinary youth sports, farmers markets, filming/setup, park/lawn closures, and private parties are demoted unless strong subway ridership evidence is present.
"""
    (run_dir / "traffic_impact_report.md").write_text(report, encoding="utf-8")
    (run_dir / "reports" / "traffic_impact_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--run_dir", default=None)
    p.add_argument("--max_physical_events", type=int, default=100)
    p.add_argument("--backend", choices=["auto", "hf_api", "isolated_transformers", "transformers", "heuristic"], default="auto")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--write_data_output", action="store_true")
    p.add_argument("--allow_heuristic_data_overwrite", action="store_true")
    p.add_argument("--prepare_isolated_env", action="store_true")
    p.add_argument("--isolated_env", default="/root/autodl-tmp/llm_screening_env")
    p.add_argument("--hf_cache", default="/root/autodl-tmp/hf_cache")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "autotemp" / f"event_screening_{stamp}"
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    ensure_run_dir(run_dir)
    if args.prepare_isolated_env:
        runner = IsolatedTransformersRunner(args.model_name, env_dir=args.isolated_env, hf_cache=args.hf_cache, require_ready=False)
        commands = runner.install_commands()
        install_script = run_dir / "logs" / "prepare_isolated_env.sh"
        install_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")
        print(json.dumps({
            "run_dir": str(run_dir),
            "prepare_isolated_env_script": str(install_script),
            "commands": commands,
            "note": "Run this script manually if you want to install the isolated local model runner.",
        }, ensure_ascii=False, indent=2), flush=True)
        return
    try:
        runner, runner_name = make_runner(args.backend, args.model_name)
    except RunnerUnavailable as exc:
        runner = None
        runner_name = f"{args.backend}_unavailable"
        unavailable_manifest = {
            "run_dir": str(run_dir),
            "backend_requested": args.backend,
            "runner_used": runner_name,
            "model_name": args.model_name,
            "unavailable_reason": str(exc),
            "data_output_written": False,
        }
        (run_dir / "screening_manifest.json").write_text(json.dumps(unavailable_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(unavailable_manifest, ensure_ascii=False, indent=2), flush=True)
        return
    events = load_events(PROJECT_ROOT / args.events_json)
    traffic_df = pd.read_csv(PROJECT_ROOT / args.traffic_csv, parse_dates=["date"])
    traffic_columns = [c for c in traffic_df.columns if c != "date"]
    physical = deduplicate_physical_events(events)
    candidates = rank_events_for_llm(physical, traffic_df, traffic_columns, args.max_physical_events)
    rows = []
    physical_by_id = {event.event_id: event for event in physical}
    for i, item in enumerate(candidates, 1):
        event = item["event"]
        decision = runner.classify(event, item["traffic_evidence"], item["rule"].category)
        rows.append(row_payload(item, decision, runner_name))
        if i % 20 == 0 or i == len(candidates):
            print(f"[screen] {i}/{len(candidates)}", flush=True)
    score_df = pd.DataFrame(rows)
    score_path = run_dir / "event_screening_scores.csv"
    score_df.to_csv(score_path, index=False)
    filtered = filtered_event_json(rows, physical_by_id)
    filtered_path = run_dir / "major_events_llm_filtered.json"
    filtered_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    data_output_path = PROJECT_ROOT / "data" / "major_events_llm_filtered.json"
    data_output_written = should_write_data_output(runner_name, args.write_data_output, args.allow_heuristic_data_overwrite)
    data_output_skip_reason = None
    if data_output_written:
        data_output_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.write_data_output:
        data_output_skip_reason = (
            "data/major_events_llm_filtered.json is only overwritten for hf_api or "
            "isolated_transformers runners unless --allow_heuristic_data_overwrite is set"
        )
    make_figures(score_df, traffic_df, run_dir)
    write_report(run_dir, len(events), len(physical), len(candidates), score_df, runner_name, args.model_name)
    manifest = {
        "run_dir": str(run_dir),
        "raw_channel_events": len(events),
        "physical_events": len(physical),
        "candidates": len(candidates),
        "kept": int(score_df["keep_for_modeling"].sum()) if not score_df.empty else 0,
        "needs_review": int(score_df["needs_review"].sum()) if not score_df.empty else 0,
        "backend_requested": args.backend,
        "runner_used": runner_name,
        "model_name": args.model_name,
        "scores_csv": str(score_path),
        "filtered_json": str(filtered_path),
        "data_output_requested": bool(args.write_data_output),
        "data_output_path": str(data_output_path),
        "data_output_written": data_output_written,
        "data_output_skip_reason": data_output_skip_reason,
    }
    (run_dir / "screening_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
