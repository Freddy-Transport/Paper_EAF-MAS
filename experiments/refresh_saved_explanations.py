#!/usr/bin/env python3
"""Refresh saved case explanation Markdown after paper-workflow formatting changes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from agents.paper_workflow import (
    CalibrationDecisionSpec,
    EventEvidenceSpec,
    ForecastRequestSpec,
    build_explanation_markdown,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _extract_reasoning(markdown: str) -> str:
    match = re.search(r"## Forecast-Time Model Reasoning\s*(.*?)(?=\n## |\Z)", markdown or "", flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _citation_summaries_by_case(run_root: Path) -> Dict[str, List[dict]]:
    status = _read_json(run_root / "reports" / "citation_evidence_status.json")
    out: Dict[str, List[dict]] = {}
    for case in status.get("cases", []) or []:
        prediction_path = str(case.get("case_prediction") or "")
        if not prediction_path:
            continue
        out[prediction_path] = list(case.get("model_assisted_summaries") or [])
    return out


def refresh_saved_explanations(run_root: str | Path) -> dict:
    run_root = Path(run_root)
    citation_summaries = _citation_summaries_by_case(run_root)
    refreshed = []
    for pred_path in sorted((run_root / "cases_vllm").glob("*/predictions/*.json")):
        payload = _read_json(pred_path)
        request_data = payload.get("request") or {}
        evidence_data = payload.get("evidence") or {}
        decision_data = payload.get("decision") or {}
        metrics = payload.get("metrics") or {}
        existing_markdown = str(payload.get("explanation_markdown") or "")
        prediction_key = str(pred_path)
        merged_summaries = list(evidence_data.get("model_assisted_summaries") or [])
        for summary in citation_summaries.get(prediction_key, []):
            if summary not in merged_summaries:
                merged_summaries.append(summary)
        evidence_data["model_assisted_summaries"] = merged_summaries
        markdown = build_explanation_markdown(
            ForecastRequestSpec(
                date=str(request_data.get("date", "")),
                horizon=int(request_data.get("horizon", 0) or 0),
                station_scope=str(request_data.get("station_scope", "")),
                mode=str(request_data.get("mode", "")),
                retrieval_policy=str(request_data.get("retrieval_policy", "online_cached")),
            ),
            EventEvidenceSpec(
                sources=list(evidence_data.get("sources") or []),
                structured_events=list(evidence_data.get("structured_events") or []),
                local_residual_cases=list(evidence_data.get("local_residual_cases") or []),
                model_assisted_summaries=merged_summaries,
                has_major_event=bool(evidence_data.get("has_major_event")),
            ),
            CalibrationDecisionSpec(
                mode=str(decision_data.get("mode", "")),
                adjusted_channels=list(decision_data.get("adjusted_channels") or []),
                abstain=bool(decision_data.get("abstain")),
                confidence=float(decision_data.get("confidence", 0.0) or 0.0),
                reason=str(decision_data.get("reason", "")),
                correction_bound=float(decision_data.get("correction_bound", 0.0) or 0.0),
            ),
            metrics=metrics,
            model_explanation=_extract_reasoning(existing_markdown) or str(decision_data.get("reason", "")),
            llm_markdown=_extract_reasoning(existing_markdown),
        )
        payload["evidence"] = evidence_data
        payload["explanation_markdown"] = markdown
        pred_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = pred_path.parents[1] / "explanations" / (pred_path.stem + ".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        refreshed.append({"prediction": str(pred_path), "markdown": str(md_path), "model_assisted_summaries": len(merged_summaries)})
    summary = {"run_root": str(run_root), "refreshed_count": len(refreshed), "refreshed": refreshed}
    (run_root / "reports" / "refresh_saved_explanations.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="autotemp/ccfa_formal_rolling_20260607_115014")
    args = parser.parse_args()
    print(json.dumps(refresh_saved_explanations(args.run_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
