"""LLM-based forecast-time explanation for paper artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


SYSTEM_PROMPT = """You are a NYC subway demand analyst.
Explain an event-aware hourly ridership forecast using only forecast-time evidence.
Do not claim to know ground truth, future realized ridership, or post-hoc metrics.
External facts must come from retrieved_sources. Your own reasoning may connect
evidence, event timing, venues, station proximity, and calibration decisions.
Use historical_event_cases only as train/validation memory, never as test evidence.
If calibration_decision.abstain is true or adjusted_channels is empty, state
that no numeric calibration was applied; residual RAG is context only.
Return exactly one valid JSON object. Do not include chain-of-thought, XML tags,
code fences, or prose outside the JSON object."""


USER_PROMPT_TEMPLATE = """/no_think
Build a concise paper-ready forecast-time explanation for an audited event-aware
subway forecast. The explanation should justify evidence, historical residual
analogues, bounded calibration, uncertainty, and excluded station/channel units.

Required JSON fields:
- event_summary: string
- evidence_used: list of strings, each naming structured evidence or Qwen-Plus model-assisted summaries used
- station_event_linking: string explaining venue/event -> nearby station link
- multi_hop_reasoning: string explaining event -> crowd pattern -> station-hour demand -> calibration
- calibration_rationale: string explaining why the system adjusted or abstained
- uncertainty_and_abstention: string explaining evidence limits and abstention policy
- markdown: markdown text with sections: Factual evidence, Residual analogues,
  Calibration rationale, Uncertainty, Risk control

Decision consistency rule:
- If calibration_decision.abstain is true or calibration_decision.adjusted_channels is empty,
  write that the final forecast remained the numerical LP-MOMENT forecast.
- In that case, do not say that the system adjusted, increased, decreased, or applied
  the residual correction to the forecast. Historical residual cases may be cited
  only as context for why the agent abstained or why a small correction would be plausible.
- Do not use rejected retrieval diagnostics as forecast-time evidence. In the formal
  summary-only workflow, retrieved_sources is expected to be empty; use
  model_assisted_summaries, structured events, and train/validation historical_event_cases.
- Multi-hop reasoning must explicitly connect: current event -> venue/station match ->
  selected residual-memory skill -> similar train/validation cases -> observed residual pattern ->
  bounded calibration or abstention.
- The explanation must answer four concrete questions:
  1) why this event is related to each included station/channel;
  2) why historical residual memory supports the correction direction;
  3) why the correction magnitude is restricted by the bound;
  4) why excluded station/channel units are excluded.
- Do not produce an operator-answer section or a local ridership outlook.
  focused_forecast_context may exist for machine-readable artifacts, but the
  Markdown should follow the audited explanation structure and should not
  summarize local start/peak/end/total ridership.

forecast_request:
{forecast_request}

structured_events:
{structured_events}

retrieved_sources:
{retrieved_sources}

model_assisted_summaries:
{model_assisted_summaries}

historical_event_cases:
{historical_event_cases}

selected_residual_memory_skills:
{selected_residual_memory_skills}

station_scope:
{station_scope}

numerical_forecast_summary:
{numerical_forecast_summary}

calibration_decision:
{calibration_decision}

evidence_audit:
{evidence_audit}

calibration_controller:
{calibration_controller}

focused_forecast_context:
{focused_forecast_context}

rag_residual_context:
{rag_residual_context}
"""


@dataclass
class ExplanationResult:
    event_summary: str = ""
    evidence_used: List[str] = field(default_factory=list)
    station_event_linking: str = ""
    multi_hop_reasoning: str = ""
    calibration_rationale: str = ""
    uncertainty_and_abstention: str = ""
    markdown: str = ""
    raw_response: str = ""
    parsed: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ForecastExplanationAgent:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-14B",
        api_key: str = "EMPTY",
        timeout_s: float = 120.0,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout_s = float(timeout_s)

    def explain(
        self,
        forecast_request: dict,
        structured_events: Sequence[dict],
        retrieved_sources: Sequence[dict],
        station_scope: dict,
        raw_forecast: Sequence[Sequence[float]],
        channel_names: Sequence[str],
        calibration_decision: dict,
        rag_residual_context: str = "",
        historical_event_cases: Sequence[dict] | None = None,
        selected_residual_memory_skills: Sequence[dict] | None = None,
        model_assisted_summaries: Sequence[dict] | None = None,
        evidence_audit: Optional[dict] = None,
        calibration_controller: Optional[dict] = None,
        focused_forecast_context: Optional[dict] = None,
        log_path: Optional[str | Path] = None,
    ) -> ExplanationResult:
        payload = {
            "forecast_request": forecast_request,
            "structured_events": self._compact_events(list(structured_events)[:4]),
            "retrieved_sources": self._compact_sources(list(retrieved_sources)[:8]),
            "model_assisted_summaries": self._compact_model_summaries(list(model_assisted_summaries or [])[:4]),
            "station_scope": station_scope,
            "numerical_forecast_summary": self._forecast_summary(raw_forecast, channel_names),
            "calibration_decision": self._compact_decision(dict(calibration_decision or {})),
            "historical_event_cases": self._compact_historical_cases(list(historical_event_cases or [])[:3]),
            "selected_residual_memory_skills": self._compact_skills(list(selected_residual_memory_skills or [])[:3]),
            "evidence_audit": self._compact_audit(dict(evidence_audit or {})),
            "calibration_controller": self._compact_controller(dict(calibration_controller or {})),
            "focused_forecast_context": dict(focused_forecast_context or {}),
            "rag_residual_context": rag_residual_context[:1200],
        }
        prompt = USER_PROMPT_TEMPLATE.format(
            forecast_request=json.dumps(payload["forecast_request"], ensure_ascii=False, indent=2),
            structured_events=json.dumps(payload["structured_events"], ensure_ascii=False, indent=2),
            retrieved_sources=json.dumps(payload["retrieved_sources"], ensure_ascii=False, indent=2),
            model_assisted_summaries=json.dumps(payload["model_assisted_summaries"], ensure_ascii=False, indent=2),
            historical_event_cases=json.dumps(payload["historical_event_cases"], ensure_ascii=False, indent=2),
            selected_residual_memory_skills=json.dumps(payload["selected_residual_memory_skills"], ensure_ascii=False, indent=2),
            station_scope=json.dumps(payload["station_scope"], ensure_ascii=False, indent=2),
            numerical_forecast_summary=json.dumps(payload["numerical_forecast_summary"], ensure_ascii=False, indent=2),
            calibration_decision=json.dumps(payload["calibration_decision"], ensure_ascii=False, indent=2),
            evidence_audit=json.dumps(payload["evidence_audit"], ensure_ascii=False, indent=2),
            calibration_controller=json.dumps(payload["calibration_controller"], ensure_ascii=False, indent=2),
            focused_forecast_context=json.dumps(payload["focused_forecast_context"], ensure_ascii=False, indent=2),
            rag_residual_context=payload["rag_residual_context"] or "(none)",
        )
        raw = ""
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": self._completion_max_tokens(),
                "response_format": {"type": "json_object"},
            }
            extra_body = self._completion_extra_body()
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**request_kwargs)
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            result = self._fallback(payload, fallback_reason=f"{type(exc).__name__}: {exc}")
        else:
            try:
                result = self._parse(raw)
            except Exception as exc:
                result = self._fallback(payload, fallback_reason=f"{type(exc).__name__}: {exc}")
        result = self._enforce_decision_consistency(result, payload["calibration_decision"])
        result.raw_response = raw or result.raw_response
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(
                json.dumps({"request_payload": payload, "result": result.to_dict()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return result

    def _completion_extra_body(self) -> Optional[dict]:
        if "qwen3" in str(self.model).lower():
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    def _completion_max_tokens(self) -> int:
        if "qwen3" in str(self.model).lower():
            return 2000
        return 1400

    @staticmethod
    def _enforce_decision_consistency(result: ExplanationResult, decision: Dict[str, Any]) -> ExplanationResult:
        adjusted_channels = decision.get("adjusted_channels") or []
        abstain = bool(decision.get("abstain")) or not bool(adjusted_channels)
        if not abstain:
            return result
        reason = str(decision.get("reason", "") or "").strip()
        consistency_note = (
            "The final forecast remains the raw PT-MOMENT numerical forecast. "
            "No numerical event calibration was applied because the decision was abstain "
            "or no station channel was selected for adjustment."
        )
        if reason:
            consistency_note = f"{consistency_note} Decision reason: {reason}"
        result.calibration_rationale = consistency_note
        result.uncertainty_and_abstention = (
            "Residual RAG cases and model-assisted event summaries are treated as forecast-time "
            "context only; they are not evidence that a correction was applied."
        )
        replacement = (
            "### Calibration rationale\n"
            f"{consistency_note}\n\n"
            "Historical residual cases may suggest a possible direction, but this run "
            "abstained from changing the numerical forecast."
        )
        markdown = result.markdown or ForecastExplanationAgent._markdown_from_fields(result)
        if "### Calibration rationale" in markdown:
            markdown = re.sub(
                r"### Calibration rationale\s*.*?(?=\n### |\Z)",
                replacement,
                markdown,
                flags=re.DOTALL,
            ).strip()
        elif "## Calibration Decision" in markdown:
            markdown = re.sub(
                r"## Calibration Decision\s*.*?(?=\n## |\n### |\Z)",
                replacement,
                markdown,
                flags=re.DOTALL,
            ).strip()
        else:
            markdown = f"{markdown.rstrip()}\n\n{replacement}"
        result.markdown = markdown
        return result

    @staticmethod
    def _parse(raw: str) -> ExplanationResult:
        text = raw.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= start:
                text = text[start : end + 1]
        data = json.loads(text)
        result = ExplanationResult(
            event_summary=str(data.get("event_summary", "")),
            evidence_used=[str(x) for x in data.get("evidence_used", [])],
            station_event_linking=str(data.get("station_event_linking", "")),
            multi_hop_reasoning=str(data.get("multi_hop_reasoning", "")),
            calibration_rationale=str(data.get("calibration_rationale", "")),
            uncertainty_and_abstention=str(data.get("uncertainty_and_abstention", "")),
            markdown=str(data.get("markdown", "")),
            parsed=True,
        )
        if not result.markdown:
            result.markdown = ForecastExplanationAgent._markdown_from_fields(result)
        return result

    @staticmethod
    def _fallback(payload: Dict[str, Any], fallback_reason: str) -> ExplanationResult:
        sources = payload.get("retrieved_sources", []) or payload.get("model_assisted_summaries", [])
        event_titles = [str(ev.get("title", "")) for ev in payload.get("structured_events", []) if ev.get("title")]
        decision = payload.get("calibration_decision", {})
        result = ExplanationResult(
            event_summary="; ".join(event_titles[:3]) or "No structured major event summary available.",
            evidence_used=[str(src.get("title") or src.get("status") or "cached evidence") for src in sources[:6]],
            station_event_linking="Station-event linking is based on precomputed venue/station matching and event relevance filtering.",
            multi_hop_reasoning="The system links event timing and nearby station matches to the hourly forecast window, then applies bounded calibration only when a major event is active.",
            calibration_rationale=str(decision.get("reason", "")),
            uncertainty_and_abstention=f"LLM explanation fallback used because {fallback_reason}. External evidence is limited to cached retrieval records.",
            parsed=False,
            fallback_reason=fallback_reason,
        )
        result.markdown = ForecastExplanationAgent._markdown_from_fields(result)
        result.raw_response = fallback_reason
        return result

    @staticmethod
    def _markdown_from_fields(result: ExplanationResult) -> str:
        evidence = "\n".join(f"- {item}" for item in result.evidence_used) or "- No external evidence available."
        return f"""### Factual evidence
{evidence}

### Residual analogues
{result.station_event_linking}

{result.multi_hop_reasoning}

### Calibration rationale
{result.calibration_rationale}

### Uncertainty
{result.uncertainty_and_abstention}

### Risk control
Bounded correction and event-station-channel masks are used to prevent unsupported changes outside the audited units.
"""


    @staticmethod
    def _compact_audit(audit: dict) -> dict:
        keep = [
            "source_validity_score",
            "geo_consistency_score",
            "temporal_alignment_score",
            "semantic_consistency_score",
            "residual_support_score",
            "evidence_validity_score",
            "conflict_flags",
            "severe_conflict_flags",
            "included_units",
            "excluded_units",
            "residual_support_summary",
        ]
        out = {k: audit.get(k) for k in keep if k in audit}
        out["included_units"] = ForecastExplanationAgent._compact_units(list(out.get("included_units") or [])[:4])
        out["excluded_units"] = ForecastExplanationAgent._compact_units(list(out.get("excluded_units") or [])[:4])
        out["conflict_flags"] = list(out.get("conflict_flags") or [])[:4]
        out["severe_conflict_flags"] = list(out.get("severe_conflict_flags") or [])[:4]
        return out

    @staticmethod
    def _compact_controller(controller: dict) -> dict:
        keep = [
            "controller_allowed",
            "abstain",
            "abstention_reason",
            "correction_stats",
            "audit_scores",
            "included_units",
            "excluded_units",
        ]
        out = {k: controller.get(k) for k in keep if k in controller}
        out["included_units"] = ForecastExplanationAgent._compact_units(list(out.get("included_units") or [])[:4])
        out["excluded_units"] = ForecastExplanationAgent._compact_units(list(out.get("excluded_units") or [])[:4])
        return out

    @staticmethod
    def _compact_model_summaries(rows: List[dict]) -> List[dict]:
        out = []
        for row in rows[:4]:
            out.append(
                {
                    "event_key": row.get("event_key", ""),
                    "summary": str(row.get("summary") or row.get("event_summary") or "")[:700],
                    "event_relevance_to_station": str(row.get("event_relevance_to_station") or "")[:400],
                    "expected_ridership_effect": str(row.get("expected_ridership_effect") or "")[:400],
                    "uncertainty": str(row.get("uncertainty") or "")[:400],
                    "source_agent": row.get("source_agent", "qwen_plus"),
                    "summary_used_for_explanation": bool(row.get("summary_used_for_explanation", True)),
                }
            )
        return out

    @staticmethod
    def _compact_decision(decision: dict) -> dict:
        keep = [
            "mode",
            "abstain",
            "reason",
            "controller_allowed",
            "confidence",
            "residual_bound",
            "max_correction",
            "correction_stats",
            "audit_scores",
            "adjusted_channels",
            "included_units",
            "excluded_units",
        ]
        out = {k: decision.get(k) for k in keep if k in decision}
        out["adjusted_channels"] = [str(x)[:80] for x in list(out.get("adjusted_channels") or [])[:4]]
        out["included_units"] = ForecastExplanationAgent._compact_units(list(out.get("included_units") or [])[:4])
        out["excluded_units"] = ForecastExplanationAgent._compact_units(list(out.get("excluded_units") or [])[:4])
        return out

    @staticmethod
    def _compact_units(units: Sequence[dict]) -> List[dict]:
        compact = []
        for unit in units:
            compact.append(
                {
                    "station_channel": str(unit.get("station_channel") or "")[:80],
                    "relation": str(unit.get("relation") or "")[:80],
                    "gate_score": unit.get("gate_score"),
                    "reason": str(unit.get("reason") or unit.get("exclusion_reason") or "")[:120],
                }
            )
        return compact

    @staticmethod
    def _forecast_summary(raw_forecast: Sequence[Sequence[float]], channel_names: Sequence[str]) -> dict:
        raw = np.asarray(raw_forecast, dtype=np.float32)
        if raw.ndim != 2 or raw.size == 0:
            return {"n_channels": 0, "horizon": 0}
        totals = raw.sum(axis=1)
        ranked = np.argsort(-totals)[:5]
        return {
            "n_channels": int(raw.shape[0]),
            "horizon": int(raw.shape[1]),
            "mean_hourly_flow": float(raw.mean()),
            "max_hourly_flow": float(raw.max()),
            "top_channels_by_predicted_volume": [
                {
                    "channel": channel_names[int(idx)] if int(idx) < len(channel_names) else f"channel_{int(idx)}",
                    "total_predicted_flow": float(totals[int(idx)]),
                    "peak_predicted_flow": float(raw[int(idx)].max()),
                }
                for idx in ranked
            ],
        }

    @staticmethod
    def _compact_events(events: Sequence[dict]) -> List[dict]:
        keep = [
            "title",
            "event_time",
            "location",
            "event_type",
            "event_category",
            "impact_tier",
            "confidence",
            "channel_name",
            "expected_direction",
        ]
        compact = []
        for event in events:
            row = {k: event.get(k) for k in keep if event.get(k) not in (None, "")}
            text = str(event.get("content") or event.get("enriched_description") or "")
            if text:
                row["description"] = text[:120]
            compact.append(row)
        return compact

    @staticmethod
    def _compact_historical_cases(cases: Sequence[dict]) -> List[dict]:
        keep = [
            "split",
            "historical_event_title",
            "event_time",
            "matched_station",
            "event_type",
            "rank_group",
            "day_type",
            "residual_direction",
            "historical_baseline_residual",
            "median_lp_moment_correction",
            "similarity_reason",
        ]
        compact = []
        for case in cases:
            compact.append({k: case.get(k) for k in keep if case.get(k) not in (None, "")})
        return compact

    @staticmethod
    def _compact_skills(skills: Sequence[dict]) -> List[dict]:
        compact = []
        for skill in skills:
            compact.append(
                {
                    "skill_id": skill.get("skill_id"),
                    "skill_category": skill.get("skill_category"),
                    "trigger_condition": skill.get("trigger_condition"),
                    "evidence_pattern": skill.get("evidence_pattern"),
                    "explanation_action": skill.get("explanation_action") or skill.get("calibration_action"),
                    "reason": skill.get("reason"),
                }
            )
        return compact

    @staticmethod
    def _compact_sources(sources: Sequence[dict]) -> List[dict]:
        keep = ["event_key", "query", "url", "title", "snippet", "retrieved_at", "status"]
        compact = []
        for source in sources:
            row = {k: source.get(k) for k in keep if source.get(k) not in (None, "")}
            if "snippet" in row:
                row["snippet"] = str(row["snippet"])[:240]
            if "query" in row:
                row["query"] = str(row["query"])[:180]
            compact.append(row)
        return compact
