#!/usr/bin/env python3
"""Run the paper-grade event-aware explainable forecasting workflow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.event_adapter import apply_adapter_to_forecast, channel_meta_by_name, load_channel_map, load_events_json  # noqa: E402
from agents.event_agent import EventAnalysisAgent  # noqa: E402
from agents.event_relevance_agent import EventRelevanceAgent  # noqa: E402
from agents.event_retrieval_agent import EventRetrievalAgent  # noqa: E402
from agents.forecast_explanation_agent import ForecastExplanationAgent  # noqa: E402
from agents.numerical_agent import NumericalPredictionAgent  # noqa: E402
from agents.paper_workflow import (  # noqa: E402
    CalibrationDecisionSpec,
    EventEvidenceSpec,
    ForecastRequestSpec,
    ForecastResultSpec,
    NumericalForecastSpec,
    build_explanation_markdown,
    compute_metrics,
    ensure_paper_run_dir,
    write_forecast_artifacts,
)
from agents.prediction_fusion import PredictionFusion  # noqa: E402
from agents.rag_pipeline import RAGPipeline  # noqa: E402
from agents.schemas import FinalPrediction  # noqa: E402
from agents.skill_extractor import SkillExtractor  # noqa: E402
from agents.skill_library import SkillLibrary  # noqa: E402
from event_post_training.config import resolve_lp_model_path  # noqa: E402


MODES = {
    "numerical_only",
    "rag_explain",
    "event_adapter_frozen_moment",
    "event_adapter_peft_moment",
    "full_skill_agent",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=sorted(MODES), default="numerical_only")
    p.add_argument("--target_date", required=True)
    p.add_argument("--station_scope", choices=["top128", "event_venue28"], default="event_venue28")
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    p.add_argument("--fusion_channels", default="data/venue37_fusion_channels.json")
    p.add_argument("--lp_model_path", default="experiments/outputs/lp_top128")
    p.add_argument("--event_adapter_path", default="experiments/outputs/event_adapter_top128")
    p.add_argument("--event_adapter_peft_path", default="experiments/outputs/event_adapter_peft_top128")
    p.add_argument("--knowledge_base_dir", default="agents/knowledge_base_residual")
    p.add_argument("--output_root", default=None)
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--llm_model", default="Qwen/Qwen3-8B")
    p.add_argument("--llm_api_key", default="EMPTY")
    p.add_argument("--device", default="auto")
    p.add_argument("--horizon", type=int, default=192)
    p.add_argument("--enable_online_retrieval", action="store_true")
    p.add_argument("--max_retrieval_events", type=int, default=8)
    p.add_argument("--max_analysis_events", type=int, default=8)
    p.add_argument("--max_explanation_events", type=int, default=12)
    p.add_argument("--allow_missing_adapter_passthrough", action="store_true")
    p.add_argument("--update_skills", action="store_true")
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_channel_names(traffic_csv: Path) -> List[str]:
    cols = pd.read_csv(traffic_csv, nrows=1).columns.tolist()
    return [c for c in cols if c != "date"]


def load_scope_indices(scope: str, channel_names: Sequence[str], fusion_channels: Path) -> Tuple[List[int], dict]:
    if scope == "top128":
        return list(range(len(channel_names))), {"scope": "top128", "n": len(channel_names)}
    payload = json.loads(fusion_channels.read_text(encoding="utf-8"))
    requested = list(payload.get("channel_names", []))
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    names = [name for name in requested if name in name_to_idx]
    return [name_to_idx[name] for name in names], {
        "scope": "event_venue28",
        "requested_venue_total": int(payload.get("venue37_total", len(requested))),
        "n": len(names),
        "missing_from_top128": payload.get("missing_from_top_n", []),
    }


def subset_prediction(pred, indices: Sequence[int]):
    pred.forecast = [pred.forecast[i] for i in indices]
    pred.channel_names = [pred.channel_names[i] for i in indices]
    if pred.ground_truth:
        pred.ground_truth = [pred.ground_truth[i] for i in indices]
    return pred


def filter_events(events, prediction, channel_map_path: Path):
    relevance = EventRelevanceAgent(channel_map_path=channel_map_path, relevance_threshold=0.45, max_events=80)
    filtered, rows = relevance.filter_events(
        events,
        forecast_timestamps=prediction.forecast_timestamps,
        target_channels=prediction.channel_names,
    )
    has_major = any(getattr(ev, "impact_tier", "C") in ("A", "B") for ev in filtered)
    return filtered, rows, has_major


def _confidence_score(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return {"high": 1.0, "medium": 0.6, "low": 0.25}.get(str(value).lower(), 0.0)


def rank_events_for_reasoning(events) -> List:
    tier_score = {"A": 4, "B": 3, "C": 2, "D": 1}

    def key(ev):
        tier = str(getattr(ev, "impact_tier", "C") or "C").upper()
        confidence = _confidence_score(getattr(ev, "confidence", None))
        title = str(getattr(ev, "title", "") or "")
        location = str(getattr(ev, "location", "") or "")
        return (
            tier_score.get(tier, 0),
            confidence,
            1 if title else 0,
            1 if location else 0,
            title,
        )

    return sorted(events, key=key, reverse=True)


def analyze_events(events, args, rag_cutoff_date: str):
    rag = None
    kb_dir = resolve_path(args.knowledge_base_dir)
    if kb_dir.exists():
        rag = RAGPipeline(knowledge_base_dir=str(kb_dir))
    agent = EventAnalysisAgent(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        rag_pipeline=rag,
    )
    return agent.analyze(events, _progress=True, rag_cutoff_date=rag_cutoff_date)


def build_rag_residual_context(events, args, cutoff_date: str) -> str:
    kb_dir = resolve_path(args.knowledge_base_dir)
    if not kb_dir.exists() or not events:
        return ""
    try:
        rag = RAGPipeline(knowledge_base_dir=str(kb_dir))
        chunks = []
        for event in events[:4]:
            ctx = rag.retrieve_context(event, cutoff_date=cutoff_date)
            if ctx:
                chunks.append(ctx[:1200])
        return "\n\n".join(chunks)
    except Exception as exc:
        return f"RAG residual context unavailable: {type(exc).__name__}: {exc}"


def adapter_adjust(mode: str, args, prediction, events, device: str):
    adapter_dir = resolve_path(args.event_adapter_path if mode == "event_adapter_frozen_moment" else args.event_adapter_peft_path)
    if not (adapter_dir / "event_adapter.pt").is_file():
        if args.allow_missing_adapter_passthrough:
            return prediction.forecast, [], f"Adapter checkpoint missing at {adapter_dir}; passthrough enabled."
        raise FileNotFoundError(f"Adapter checkpoint missing: {adapter_dir / 'event_adapter.pt'}")
    meta = channel_meta_by_name(load_channel_map(resolve_path(args.channel_map)))
    adjusted, correction = apply_adapter_to_forecast(
        prediction.forecast,
        prediction.forecast_timestamps,
        prediction.channel_names,
        events,
        meta,
        adapter_dir,
        device=device,
    )
    changed = np.abs(correction).max(axis=1) > 1e-6
    channels = [prediction.channel_names[i] for i, yes in enumerate(changed) if yes]
    reason = f"Applied residual adapter from {adapter_dir}; max correction={float(np.abs(correction).max()):.4f}."
    return adjusted.tolist(), channels, reason


def write_preference_record(run_dir: Path, mode: str, metrics_raw: dict, metrics_adjusted: dict, explanation: str) -> None:
    if not metrics_raw or not metrics_adjusted:
        return
    delta = metrics_raw.get("wape", 0.0) - metrics_adjusted.get("wape", 0.0)
    row = {
        "mode": mode,
        "metric": "wape",
        "metric_delta_raw_minus_adjusted": delta,
        "auto_preference": "adjusted" if delta > 0 else "raw",
        "explanation_preview": explanation[:500],
    }
    path = run_dir / "preferences" / "preference_records.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_update_skill_library(run_dir: Path, final_prediction: FinalPrediction, channel_map_path: Path) -> None:
    library = SkillLibrary(str(run_dir / "models" / "skill_library.jsonl"))
    extractor = SkillExtractor(library, channel_map_path=str(channel_map_path))
    trace_id = f"paper_{int(time.time())}"
    skills = extractor.extract(final_prediction, trace_id=trace_id)
    (run_dir / "reports" / "skill_updates.json").write_text(
        json.dumps([s.model_dump() for s in skills], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = "cuda:0" if args.device == "auto" else args.device
    if args.device == "auto":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    run_dir = ensure_paper_run_dir(
        args.output_root
        or PROJECT_ROOT / "autotemp" / f"paper_event_forecasting_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    traffic_csv = resolve_path(args.traffic_csv)
    events_json = resolve_path(args.events_json)
    channel_map = resolve_path(args.channel_map)
    fusion_channels = resolve_path(args.fusion_channels)
    channel_names = load_channel_names(traffic_csv)
    scope_indices, scope_meta = load_scope_indices(args.station_scope, channel_names, fusion_channels)
    resolved_lp, lp_source = resolve_lp_model_path(resolve_path(args.lp_model_path), PROJECT_ROOT)

    manifest = {
        "mode": args.mode,
        "target_date": args.target_date,
        "station_scope": args.station_scope,
        "scope_meta": scope_meta,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "resolved_lp_model_path": str(resolved_lp),
        "lp_source": lp_source,
        "device": device,
        "max_retrieval_events": args.max_retrieval_events,
        "max_analysis_events": args.max_analysis_events,
        "max_explanation_events": args.max_explanation_events,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    agent = NumericalPredictionAgent(
        model_path=str(PROJECT_ROOT / "__paper_no_gca__"),
        device=device,
        forecast_horizon=args.horizon,
        n_channels=len(channel_names),
        channel_names=channel_names,
        lp_model_path=str(resolved_lp),
        lp_data_path=str(traffic_csv),
        allow_lp_training_fallback=False,
    )
    prediction = agent.predict_for_date(str(traffic_csv), args.target_date, forecast_horizon=args.horizon)
    prediction = subset_prediction(prediction, scope_indices)
    events = load_events_json(events_json) if events_json.is_file() else []
    filtered_events, relevance_rows, has_major = filter_events(events, prediction, channel_map)
    ranked_events = rank_events_for_reasoning(filtered_events)
    retrieval_events = ranked_events[: max(0, args.max_retrieval_events)]
    analysis_events = ranked_events[: max(0, args.max_analysis_events)]
    explanation_events = ranked_events[: max(0, args.max_explanation_events)]

    retrieval = EventRetrievalAgent(
        cache_dir=run_dir / "retrieval_cache",
        enabled=args.enable_online_retrieval,
    )
    evidence_rows = retrieval.retrieve_for_events(
        retrieval_events,
        args.target_date,
        prediction.channel_names,
        max_results_per_event=4,
    )
    impacts = []
    adjusted = prediction.forecast
    adjusted_channels: List[str] = []
    reason = "Numerical baseline used directly."
    abstain = True

    if args.mode == "numerical_only":
        pass
    elif args.mode == "rag_explain":
        if analysis_events:
            impacts = analyze_events(analysis_events, args, prediction.forecast_timestamps[0][:10])
            reason = "RAG/LLM explanation generated; numerical forecast left unchanged for ablation."
        abstain = True
    elif args.mode in ("event_adapter_frozen_moment", "event_adapter_peft_moment"):
        if has_major:
            adjusted, adjusted_channels, reason = adapter_adjust(args.mode, args, prediction, filtered_events, device)
            abstain = len(adjusted_channels) == 0
        else:
            reason = "No Tier A/B event after relevance filtering; adapter abstained."
            abstain = True
    elif args.mode == "full_skill_agent":
        if has_major and analysis_events:
            impacts = analyze_events(analysis_events, args, prediction.forecast_timestamps[0][:10])
            fused = PredictionFusion(fusion_channel_names=set(prediction.channel_names)).fuse(prediction, impacts)
            adjusted = fused.adjusted_forecast
            adjusted_channels = fused.channels_adjusted
            reason = "Full event RAG plus skill-compatible fusion applied."
            abstain = len(adjusted_channels) == 0
        else:
            reason = "No Tier A/B event after relevance filtering; full agent abstained."
            abstain = True

    raw_metrics = compute_metrics(prediction.ground_truth, prediction.forecast)
    adjusted_metrics = compute_metrics(prediction.ground_truth, adjusted)
    request = ForecastRequestSpec(
        date=args.target_date,
        horizon=args.horizon,
        station_scope=args.station_scope,
        mode=args.mode,
        retrieval_policy="online_cached" if args.enable_online_retrieval else "cache_record_only",
    )
    evidence = EventEvidenceSpec(
        sources=[row.to_dict() for row in evidence_rows],
        structured_events=[ev.model_dump() for ev in explanation_events],
        has_major_event=has_major,
    )
    decision = CalibrationDecisionSpec(
        mode=args.mode,
        adjusted_channels=adjusted_channels,
        abstain=abstain,
        confidence=0.8 if adjusted_channels else 0.0,
        reason=reason,
        correction_bound=0.05 if "adapter" in args.mode else 0.10,
    )
    rag_residual_context = build_rag_residual_context(explanation_events, args, prediction.forecast_timestamps[0][:10])
    explanation_agent = ForecastExplanationAgent(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
    )
    explanation_result = explanation_agent.explain(
        forecast_request=request.__dict__,
        structured_events=[ev.model_dump() for ev in explanation_events],
        retrieved_sources=[row.to_dict() for row in evidence_rows],
        station_scope=scope_meta,
        raw_forecast=prediction.forecast,
        channel_names=prediction.channel_names,
        calibration_decision=decision.__dict__,
        rag_residual_context=rag_residual_context,
        log_path=run_dir / "logs" / f"llm_explanation_{args.mode}.json",
    )
    model_explanation = (
        PredictionFusion().fuse(prediction, impacts).explanation
        if impacts and args.mode == "rag_explain"
        else reason
    )
    explanation = build_explanation_markdown(
        request,
        evidence,
        decision,
        adjusted_metrics,
        model_explanation,
        llm_markdown=explanation_result.markdown,
    )
    result = ForecastResultSpec(
        request=request,
        numerical=NumericalForecastSpec(
            raw_forecast=prediction.forecast,
            channel_names=prediction.channel_names,
            timestamps=prediction.forecast_timestamps or [],
            ground_truth=prediction.ground_truth,
            model_type=agent.model_type,
        ),
        evidence=evidence,
        decision=decision,
        adjusted_forecast=adjusted,
        explanation_markdown=explanation,
        metrics=adjusted_metrics,
    )
    stem = f"{args.target_date.replace(':', '').replace(' ', '_')}_{args.station_scope}_{args.mode}"
    write_forecast_artifacts(result, run_dir, stem=stem)
    write_preference_record(run_dir, args.mode, raw_metrics, adjusted_metrics, explanation)

    if args.update_skills and args.mode == "full_skill_agent" and impacts:
        final_prediction = FinalPrediction(
            raw_forecast=prediction.forecast,
            adjusted_forecast=adjusted,
            events_considered=impacts,
            explanation=explanation,
            channel_names=prediction.channel_names,
            ground_truth=prediction.ground_truth,
            forecast_timestamps=prediction.forecast_timestamps,
            channels_adjusted=adjusted_channels,
            channels_passthrough=[ch for ch in prediction.channel_names if ch not in adjusted_channels],
        )
        maybe_update_skill_library(run_dir, final_prediction, channel_map)

    summary = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "raw_metrics": raw_metrics,
        "adjusted_metrics": adjusted_metrics,
        "has_major_event": has_major,
        "adjusted_channels": adjusted_channels,
        "llm_explanation_parsed": explanation_result.parsed,
        "llm_explanation_fallback_reason": explanation_result.fallback_reason,
        "artifacts": result.artifacts,
    }
    (run_dir / "reports" / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
