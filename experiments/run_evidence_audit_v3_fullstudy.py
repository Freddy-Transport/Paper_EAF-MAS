#!/usr/bin/env python3
"""Evidence Audit v3 and large-sample explanation-quality study.

This runner separates numerical evaluation from explanation evaluation. It does
not tune forecasting thresholds on test data and it does not use LLMs as external
fact sources. The local LLM audits supplied evidence and judges explanation text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.event_adapter import load_events_json  # noqa: E402
from agents.explanation_evaluator import evaluate_explanation_quality  # noqa: E402
from agents.forecast_evidence_auditor import ForecastEvidenceAuditor  # noqa: E402
from agents.llm_evidence_auditor import LLMEvidenceAuditScorer  # noqa: E402
from agents.paper_workflow import deduplicate_residual_cases, extract_historical_event_cases  # noqa: E402
from agents.rag_pipeline import RAGPipeline  # noqa: E402
from event_post_training.config import EventPostTrainingConfig  # noqa: E402

MODES = ["structured_event_only", "residual_rag_no_skill", "residual_rag_static_skill", "full_eaf_mas_explanation"]
QUALITY_KEYS = [
    "evidence_coverage",
    "station_event_linking_accuracy",
    "geo_consistency_pass_rate",
    "temporal_admissibility_pass_rate",
    "unsupported_claim_rate",
    "citation_quality_coverage",
    "residual_memory_relevance",
    "decision_consistency",
    "leakage_free_rate",
    "multi_hop_completeness",
    "abstention_appropriateness",
]
AUDIT_KEYS = [
    "llm_source_relevance_score",
    "llm_temporal_admissibility_score",
    "llm_geo_station_consistency_score",
    "llm_event_station_linking_score",
    "llm_residual_memory_support_score",
    "llm_conflict_severity",
    "llm_confidence",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default=f"autotemp/evidence_audit_v3_llm_fullstudy_{time.strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--fusion_channels", default="data/venue37_fusion_channels.json")
    p.add_argument("--knowledge_base_dir", default="agents/knowledge_base_residual")
    p.add_argument("--horizon", type=int, default=192)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_windows", type=int, default=120)
    p.add_argument("--smoke_windows", type=int, default=0)
    p.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--llm_model", default="Qwen/Qwen3-8B")
    p.add_argument("--llm_api_key", default="EMPTY")
    p.add_argument("--llm_timeout_s", type=float, default=120.0)
    p.add_argument("--use_fake_llm", action="store_true", help="Only for unit/smoke fallback when vLLM is unavailable.")
    p.add_argument("--skip_llm_judge", action="store_true", help="Use mechanical explanation metrics only.")
    p.add_argument("--min_score_std", type=float, default=0.03)
    return p.parse_args()


def ensure_dirs(root: Path) -> None:
    for sub in ["reports", "tables", "figures", "predictions", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def parse_dt(value: Any):
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def as_dict(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        return dict(event)
    if hasattr(event, "model_dump"):
        return dict(event.model_dump())
    return dict(getattr(event, "__dict__", {"value": str(event)}))


def load_scope_channels(path: Path, all_channels: Sequence[str]) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = list(payload.get("channel_names", []))
    existing = set(all_channels)
    return [name for name in requested if name in existing]


def choose_anchor_dates(df: pd.DataFrame, cfg: EventPostTrainingConfig, split: str, stride: int) -> List[Dict[str, Any]]:
    if split == "val":
        split_start = cfg.train_rows
        split_end = min(cfg.train_rows + cfg.val_rows, len(df))
    else:
        split_start = min(cfg.train_rows + cfg.val_rows, len(df))
        split_end = len(df)
    start = max(split_start + cfg.seq_len, cfg.seq_len)
    stop = split_end - cfg.horizon
    rows = []
    if stop < start:
        return rows
    for anchor in range(start, stop + 1, max(1, stride)):
        rows.append({"split": split, "anchor": int(anchor), "date": str(df.iloc[anchor]["date"])})
    return rows


def events_in_horizon(events: Sequence[Any], anchor_time: str, horizon: int, scope_channels: set[str]) -> List[Dict[str, Any]]:
    anchor = parse_dt(anchor_time)
    if anchor is None:
        return []
    end = anchor + timedelta(hours=int(horizon))
    out = []
    for event in events:
        row = as_dict(event)
        when = parse_dt(row.get("event_time"))
        if when is None or not (anchor <= when < end):
            continue
        if row.get("channel_name") and str(row.get("channel_name")) not in scope_channels:
            continue
        out.append(row)
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    out.sort(key=lambda r: (tier_order.get(str(r.get("impact_tier") or "C").upper(), 9), str(r.get("event_time") or ""), str(r.get("title") or "")))
    return out


def event_active(rows: Sequence[Dict[str, Any]]) -> bool:
    return any(str(r.get("impact_tier") or "C").upper() in {"A", "B"} for r in rows)


def select_windows(df: pd.DataFrame, cfg: EventPostTrainingConfig, events: Sequence[Any], scope_channels: Sequence[str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    scope_set = set(scope_channels)
    val = choose_anchor_dates(df, cfg, "val", stride=max(1, args.stride * 24))
    test = choose_anchor_dates(df, cfg, "test", stride=max(1, args.stride))
    candidates = []
    for row in val + test:
        horizon_events = events_in_horizon(events, row["date"], args.horizon, scope_set)
        major = event_active(horizon_events)
        candidates.append({
            **row,
            "event_active": bool(horizon_events),
            "major_event_active": bool(major),
            "event_count": len(horizon_events),
            "events": horizon_events[:6],
        })

    def diversity_key(row: Dict[str, Any]) -> tuple:
        ev = (row.get("events") or [{}])[0] if row.get("events") else {}
        title = str(ev.get("title") or "no_event").lower()[:40]
        etype = str(ev.get("event_type") or "unknown").lower()[:25]
        channel = str(ev.get("channel_name") or "no_channel").lower()
        return (title, etype, channel)

    def take_diverse(rows: Sequence[Dict[str, Any]], limit: int, selected: List[Dict[str, Any]], seen_windows: set, seen_events: set) -> None:
        for row in rows:
            wkey = (row["split"], row["anchor"])
            ekey = diversity_key(row)
            if wkey in seen_windows or ekey in seen_events:
                continue
            selected.append(row)
            seen_windows.add(wkey)
            seen_events.add(ekey)
            if len(selected) >= limit:
                return
        for row in rows:
            wkey = (row["split"], row["anchor"])
            if wkey in seen_windows:
                continue
            selected.append(row)
            seen_windows.add(wkey)
            if len(selected) >= limit:
                return

    event_rows = [r for r in candidates if r["event_count"] > 0]
    control_rows = [r for r in candidates if r["event_count"] == 0]
    major_rows = [r for r in event_rows if r.get("major_event_active")]
    selected: List[Dict[str, Any]] = []
    seen_windows: set = set()
    seen_events: set = set()
    target = int(args.smoke_windows or args.max_windows)
    take_diverse([r for r in major_rows if r["split"] == "test"], target, selected, seen_windows, seen_events)
    take_diverse([r for r in event_rows if r["split"] == "test"], target, selected, seen_windows, seen_events)
    take_diverse([r for r in major_rows if r["split"] == "val"], target, selected, seen_windows, seen_events)
    take_diverse([r for r in event_rows if r["split"] == "val"], target, selected, seen_windows, seen_events)
    take_diverse(control_rows, target, selected, seen_windows, seen_events)
    return selected[:target]


def fake_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for i, item in enumerate(payload.get("items_to_score") or []):
        title = str(item.get("event_title") or "").lower()
        location = str(item.get("location") or "").lower()
        tier = str(item.get("impact_tier") or "C").upper()
        source = 0.45 + (0.20 if tier in {"A", "B"} else 0.0) + (0.08 if item.get("source_type") else 0.0)
        temporal = 0.88 if item.get("within_horizon_by_hard_rule") else 0.18
        geo = 0.82 if any(t in location + title for t in ["times square", "herald", "barclays", "yankee", "garden"]) else 0.48 + 0.04 * (i % 3)
        residual = 0.65 if payload.get("historical_event_cases") else 0.20 + 0.05 * (i % 2)
        conflict = 0.10 if temporal > 0.5 and geo > 0.5 else 0.55
        decision = "accept_for_calibration" if min(temporal, geo, residual) > 0.6 else "accept_for_explanation" if temporal > 0.5 else "reject"
        rows.append({
            "item_id": item.get("item_id"),
            "source_relevance_score": round(min(source, 1.0), 3),
            "temporal_admissibility_score": round(temporal, 3),
            "geo_station_consistency_score": round(min(geo, 1.0), 3),
            "event_station_linking_score": round(min(geo + 0.04, 1.0), 3),
            "residual_memory_support_score": round(min(residual, 1.0), 3),
            "conflict_severity": round(conflict, 3),
            "confidence": round(0.70 if conflict < 0.2 else 0.48, 3),
            "decision": decision,
            "rationale": "Score varies with supplied tier, location, timing, and residual memory.",
            "referenced_item_ids": [item.get("item_id")],
        })
    return {"item_audits": rows, "aggregate": {}}


def build_scorer(args: argparse.Namespace) -> LLMEvidenceAuditScorer:
    return LLMEvidenceAuditScorer(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout_s=args.llm_timeout_s,
        completion_fn=fake_completion if args.use_fake_llm else None,
    )


def retrieve_residual_context(rag: RAGPipeline | None, events: Sequence[Dict[str, Any]], cutoff_date: str) -> tuple[List[str], List[Dict[str, Any]]]:
    if rag is None or not events:
        return [], []
    chunks = []
    for event in events[:3]:
        try:
            ctx = rag.retrieve_context(event, cutoff_date=cutoff_date)
        except Exception:
            ctx = ""
        if ctx:
            chunks.append(ctx[:1400])
    residual_cases = deduplicate_residual_cases(chunks, max_cases=4)
    historical = extract_historical_event_cases(chunks, max_cases=4)
    return residual_cases, historical


def markdown_for_mode(mode: str, window: Dict[str, Any], events: Sequence[Dict[str, Any]], audit: Dict[str, Any], residual_cases: Sequence[str], historical_cases: Sequence[Dict[str, Any]]) -> str:
    event_lines = [f"- {e.get('title')} | {e.get('event_time')} | tier={e.get('impact_tier')} | station={e.get('channel_name')}" for e in events[:3]] or ["- No structured future event in horizon."]
    memory_lines = [f"- {c.get('historical_event_title')} | split={c.get('split')} | direction={c.get('residual_direction')} | correction={c.get('median_lp_moment_correction')}" for c in historical_cases[:3]] or ["- No train/validation historical residual case selected."]
    skill_line = "Residual-memory skill guidance was not used."
    if mode in {"residual_rag_static_skill", "full_eaf_mas_explanation"}:
        skill_line = "Residual-memory skill guidance selects train/validation analogues with matching event type, station rank, and residual direction."
    decision = audit.get("llm_final_decision") or "accept_for_explanation"
    correction_text = "No numerical correction is claimed in this explanation study."
    if decision == "accept_for_calibration" and mode == "full_eaf_mas_explanation":
        correction_text = "The evidence supports only bounded calibration on event-station-channel cells; the correction remains capped by the controller."
    sections = [
        "# Event-Aware Forecast Explanation",
        "",
        "## Forecast-time Evidence",
        f"- Anchor time: {window['date']}",
        f"- Forecast horizon: 192 hours",
        f"- Mode: {mode}",
        "### Structured Future Events within Horizon",
        *event_lines,
    ]
    if mode != "structured_event_only":
        sections += [
            "### Historical Event Memory from Train/Validation",
            *memory_lines,
            "### Residual Pattern Evidence",
            *(list(residual_cases[:3]) or ["- No residual pattern evidence selected."]),
        ]
    if mode in {"residual_rag_static_skill", "full_eaf_mas_explanation"}:
        sections += ["### Residual-Memory Skill Guidance", f"- {skill_line}"]
    sections += [
        "## Forecast-Time Model Reasoning",
        "The current event is first linked to a venue and nearby station channel, then aligned with forecast horizon hours. Historical train/validation residual memory is used only when available to support the direction and uncertainty of the residual pattern. The controller either keeps the raw PT-MOMENT forecast or allows a bounded local correction.",
        "## Calibration Decision",
        f"- LLM audit decision: {decision}",
        f"- Risk control: {correction_text}",
        "## Uncertainty / Abstention",
        "If evidence is weak, geographically inconsistent, temporally inadmissible, or unsupported by residual memory, the framework abstains from numerical calibration and keeps the raw forecast.",
    ]
    return "\n".join(str(x) for x in sections) + "\n"


def llm_judge_explanation(args: argparse.Namespace, evidence_payload: Dict[str, Any], markdown: str) -> Dict[str, float]:
    if args.skip_llm_judge or args.use_fake_llm:
        return {}
    try:
        from openai import OpenAI
        client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key, timeout=args.llm_timeout_s)
        prompt = {
            "instruction": "Score this forecast explanation using only supplied evidence. Return JSON scores in [0,1] for evidence_relevance, station_event_linking_accuracy, geo_consistency, temporal_admissibility, residual_memory_relevance, decision_consistency, multi_hop_completeness, unsupported_claim_rate, leakage_free, abstention_appropriateness.",
            "evidence": evidence_payload,
            "explanation": markdown[:5000],
        }
        kwargs = {
            "model": args.llm_model,
            "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            "temperature": 0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
        }
        if "qwen3" in args.llm_model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        content = client.chat.completions.create(**kwargs).choices[0].message.content or "{}"
        return {k: float(v) for k, v in json.loads(content).items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def evaluate_mode(args: argparse.Namespace, mode: str, window: Dict[str, Any], events: Sequence[Dict[str, Any]], audit: Dict[str, Any], residual_cases: Sequence[str], historical_cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    markdown = markdown_for_mode(mode, window, events, audit, residual_cases, historical_cases)
    decision = {"abstain": audit.get("llm_final_decision") not in {"accept_for_calibration"}, "adjusted_channels": [] if audit.get("llm_final_decision") != "accept_for_calibration" else [events[0].get("channel_name", "")] if events else []}
    mechanical = evaluate_explanation_quality(
        markdown=markdown,
        accepted_evidence=[],
        decision=decision,
        residual_cases=[] if mode == "structured_event_only" else residual_cases,
        structured_events=events,
        historical_event_cases=[] if mode == "structured_event_only" else historical_cases,
        selected_residual_memory_skills=[{"skill_category": "residual_memory_skill", "promoted": True}] if mode in {"residual_rag_static_skill", "full_eaf_mas_explanation"} else [],
    )
    judge = llm_judge_explanation(args, {"events": events, "audit": audit, "historical_cases": historical_cases, "residual_cases": residual_cases}, markdown)
    row = {
        "mode": mode,
        "evidence_coverage": mechanical.get("evidence_coverage", 0.0),
        "station_event_linking_accuracy": 1.0 if mechanical.get("station_event_linking") else 0.0,
        "geo_consistency_pass_rate": 1.0 if float(audit.get("geo_consistency_score", 0.0) or 0.0) >= 0.6 else 0.0,
        "temporal_admissibility_pass_rate": 1.0 if float(audit.get("temporal_alignment_score", 0.0) or 0.0) >= 0.7 else 0.0,
        "unsupported_claim_rate": mechanical.get("unsupported_claim_rate", 0.0),
        "citation_quality_coverage": mechanical.get("citation_coverage", 0.0),
        "residual_memory_relevance": mechanical.get("residual_case_relevance", 0.0),
        "decision_consistency": 1.0 if mechanical.get("calibration_decision_consistent") else 0.0,
        "leakage_free_rate": mechanical.get("leakage_free_rate", 0.0),
        "multi_hop_completeness": mechanical.get("multi_hop_completeness", 0.0),
        "abstention_appropriateness": 1.0 if decision.get("abstain") else 0.5,
        "llm_judge": judge,
        "markdown": markdown,
    }
    if judge:
        mapping = {
            "evidence_relevance": "evidence_coverage",
            "station_event_linking_accuracy": "station_event_linking_accuracy",
            "geo_consistency": "geo_consistency_pass_rate",
            "temporal_admissibility": "temporal_admissibility_pass_rate",
            "residual_memory_relevance": "residual_memory_relevance",
            "decision_consistency": "decision_consistency",
            "multi_hop_completeness": "multi_hop_completeness",
            "unsupported_claim_rate": "unsupported_claim_rate",
            "leakage_free": "leakage_free_rate",
            "abstention_appropriateness": "abstention_appropriateness",
        }
        for src, dst in mapping.items():
            if src in judge:
                row[dst] = float(judge[src])
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys() if k not in {"markdown", "llm_judge", "events", "audit"}})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def mean_by(rows: Sequence[Dict[str, Any]], group_key: str, metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for group in sorted({str(r.get(group_key)) for r in rows}):
        subset = [r for r in rows if str(r.get(group_key)) == group]
        out[group] = {m: float(np.mean([float(r.get(m, 0.0) or 0.0) for r in subset])) for m in metrics}
        out[group]["n"] = len(subset)
    return out


def write_tex_table(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Mode & Evidence & Link & Geo & Temporal & Residual & Multi-hop " + "\\\\",
        r"\midrule",
    ]
    for mode in MODES:
        vals = summary.get(mode, {})
        mode_label = mode.replace("_", r"\_")
        lines.append(
            f"{mode_label} & "
            f"{vals.get('evidence_coverage', 0.0):.3f} & "
            f"{vals.get('station_event_linking_accuracy', 0.0):.3f} & "
            f"{vals.get('geo_consistency_pass_rate', 0.0):.3f} & "
            f"{vals.get('temporal_admissibility_pass_rate', 0.0):.3f} & "
            f"{vals.get('residual_memory_relevance', 0.0):.3f} & "
            f"{vals.get('multi_hop_completeness', 0.0):.3f} " + "\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_figures(root: Path, audit_rows: Sequence[Dict[str, Any]], quality_rows: Sequence[Dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix_rows = audit_rows[: min(40, len(audit_rows))]
    fig, ax = plt.subplots(figsize=(11.5, max(4.0, 0.22 * len(matrix_rows) + 2.4)))
    data = np.asarray([[float(row.get(k, 0.0) or 0.0) for k in AUDIT_KEYS] for row in matrix_rows], dtype=float)
    if data.size == 0:
        data = np.zeros((1, len(AUDIT_KEYS)))
        labels = ["no audited units"]
    else:
        labels = [f"{row.get('split')}:{row.get('anchor')}:{str(row.get('event_title',''))[:28]}" for row in matrix_rows]
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks(np.arange(len(AUDIT_KEYS)))
    ax.set_xticklabels([k.replace("llm_", "").replace("_score", "").replace("_", "\n") for k in AUDIT_KEYS], fontsize=8)
    ax.set_title("Evidence Audit v3: item-level LLM audit scores over matched event-station units")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(root / "figures" / "fig8_evidence_audit_v3_matrix.pdf")
    fig.savefig(root / "figures" / "fig8_evidence_audit_v3_matrix.png", dpi=180)
    plt.close(fig)

    summary = mean_by(quality_rows, "mode", QUALITY_KEYS)
    plot_keys = ["evidence_coverage", "station_event_linking_accuracy", "residual_memory_relevance", "decision_consistency", "leakage_free_rate", "multi_hop_completeness"]
    x = np.arange(len(plot_keys))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for i, mode in enumerate(MODES):
        vals = [summary.get(mode, {}).get(k, 0.0) for k in plot_keys]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=mode.replace("_", " "))
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in plot_keys], fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("Large-sample explanation-quality study across matched windows")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "figures" / "explanation_quality_fullstudy.pdf")
    fig.savefig(root / "figures" / "explanation_quality_fullstudy.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = PROJECT_ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    ensure_dirs(root)
    cfg = EventPostTrainingConfig(traffic_csv=PROJECT_ROOT / args.traffic_csv, events_json=PROJECT_ROOT / args.events_json, channel_map_path=PROJECT_ROOT / "data/nyc_top128_channel_map.json", fusion_channels_path=PROJECT_ROOT / args.fusion_channels)
    cfg.horizon = int(args.horizon)
    df = pd.read_csv(PROJECT_ROOT / args.traffic_csv)
    all_channels = [c for c in df.columns if c != "date"]
    scope_channels = load_scope_channels(PROJECT_ROOT / args.fusion_channels, all_channels)
    events = load_events_json(PROJECT_ROOT / args.events_json)
    windows = select_windows(df, cfg, events, scope_channels, args)
    scorer = build_scorer(args)
    auditor = ForecastEvidenceAuditor()
    rag = RAGPipeline(knowledge_base_dir=str(PROJECT_ROOT / args.knowledge_base_dir)) if (PROJECT_ROOT / args.knowledge_base_dir).exists() else None

    audit_rows: List[Dict[str, Any]] = []
    quality_rows: List[Dict[str, Any]] = []
    for w_i, window in enumerate(windows):
        events_for_window = list(window.get("events") or [])[:4]
        candidates = sorted({str(e.get("channel_name")) for e in events_for_window if e.get("channel_name") and str(e.get("channel_name")) in scope_channels})
        residual_cases, historical_cases = retrieve_residual_context(rag, events_for_window, window["date"])
        audit = auditor.audit(
            anchor_time=window["date"],
            horizon_hours=args.horizon,
            structured_events=events_for_window,
            evidence_sources=[],
            model_assisted_summaries=[],
            channel_names=scope_channels,
            adjusted_channel_candidates=candidates,
            historical_event_cases=historical_cases,
            residual_cases=residual_cases,
            llm_scorer=scorer,
        )
        item_rows = audit.get("llm_item_audits") or []
        if not item_rows:
            item_rows = [{"item_id": "none"}]
        for item in item_rows:
            event_title = ""
            try:
                idx = int(str(item.get("item_id", "event_0")).split("_", 1)[1])
                event_title = str(events_for_window[idx].get("title", "")) if idx < len(events_for_window) else ""
            except Exception:
                event_title = ""
            row = {
                "window_index": w_i,
                "split": window["split"],
                "anchor": window["anchor"],
                "date": window["date"],
                "event_active": window["event_active"],
                "event_count": window["event_count"],
                "event_title": event_title,
                "llm_final_decision": audit.get("llm_final_decision", ""),
                "audit_version": audit.get("audit_version", ""),
            }
            for key in AUDIT_KEYS:
                row[key] = float(item.get(key.replace("llm_", ""), audit.get(key, 0.0)) or 0.0)
            audit_rows.append(row)
        for mode in MODES:
            q = evaluate_mode(args, mode, window, events_for_window, audit, residual_cases, historical_cases)
            q.update({"window_index": w_i, "split": window["split"], "anchor": window["anchor"], "date": window["date"], "event_active": window["event_active"]})
            quality_rows.append(q)
            if w_i < 8:
                md_path = root / "predictions" / f"window_{w_i:03d}_{mode}.md"
                md_path.write_text(q.pop("markdown"), encoding="utf-8")
            else:
                q.pop("markdown", None)
    write_csv(root / "predictions" / "llm_evidence_audit_rows.csv", audit_rows)
    write_csv(root / "predictions" / "explanation_quality_rows.csv", quality_rows)
    quality_summary = mean_by(quality_rows, "mode", QUALITY_KEYS)
    audit_stds = {key: float(np.std([float(row.get(key, 0.0) or 0.0) for row in audit_rows])) if audit_rows else 0.0 for key in AUDIT_KEYS}
    variance_warnings = {k: v for k, v in audit_stds.items() if k not in {"llm_conflict_severity"} and len(audit_rows) >= 20 and v < args.min_score_std}
    summary = {
        "run_dir": str(root),
        "window_count": len(windows),
        "audit_row_count": len(audit_rows),
        "quality_row_count": len(quality_rows),
        "modes": MODES,
        "audit_score_stds": audit_stds,
        "variance_warnings": variance_warnings,
        "quality_summary_by_mode": quality_summary,
        "llm_model": args.llm_model,
        "used_fake_llm": bool(args.use_fake_llm),
        "skipped_llm_judge": bool(args.skip_llm_judge),
    }
    (root / "reports" / "explanation_quality_fullstudy.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "reports" / "full_anchor_protocol.json").write_text(json.dumps({"split_policy": {"train_rows": cfg.train_rows, "val_rows": cfg.val_rows, "test_start": cfg.train_rows + cfg.val_rows, "horizon": args.horizon}, "selected_windows": windows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Evidence Audit v3 Summary", "", f"- window_count: `{len(windows)}`", f"- audit_row_count: `{len(audit_rows)}`", f"- llm_model: `{args.llm_model}`", "", "## Audit Score Std", ""]
    for key, value in audit_stds.items():
        lines.append(f"- {key}: `{value:.4f}`")
    if variance_warnings:
        lines += ["", "## Variance Warnings", ""]
        for key, value in variance_warnings.items():
            lines.append(f"- {key}: std `{value:.4f}` below threshold `{args.min_score_std}`")
    (root / "reports" / "evidence_audit_v3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_tex_table(root / "tables" / "explanation_quality_fullstudy.tex", quality_summary)
    write_figures(root, audit_rows, quality_rows)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
