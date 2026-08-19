"""Forecast-time evidence auditing for event-aware calibration decisions."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence


ALLOWED_SOURCE_TYPES = {"official", "venue", "permit", "archived", "model-assisted", "source-assisted"}
POST_EVENT_PATTERNS = ("recap", "after", "review", "photos", "coverage", "post-event")


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", text or "") if len(t) >= 3}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    try:
        return float(text) / (100.0 if abs(float(text)) > 1.0 else 1.0)
    except Exception:
        return default


class ForecastEvidenceAuditor:
    def __init__(
        self,
        source_threshold: float = 0.60,
        geo_threshold: float = 0.60,
        temporal_threshold: float = 0.70,
        residual_threshold: float = 0.50,
    ):
        self.thresholds = {
            "source_validity_score": float(source_threshold),
            "geo_consistency_score": float(geo_threshold),
            "temporal_alignment_score": float(temporal_threshold),
            "residual_support_score": float(residual_threshold),
        }

    def audit(
        self,
        anchor_time: str,
        horizon_hours: int,
        structured_events: Sequence[dict] | None = None,
        evidence_sources: Sequence[dict] | None = None,
        model_assisted_summaries: Sequence[dict] | None = None,
        channel_names: Sequence[str] | None = None,
        adjusted_channel_candidates: Sequence[str] | None = None,
        historical_event_cases: Sequence[dict] | None = None,
        residual_cases: Sequence[str] | None = None,
        llm_scorer: Any | None = None,
    ) -> Dict[str, Any]:
        """Audit forecast-time evidence.

        The rule-based branch is the hard safety gate. When ``llm_scorer`` is
        provided, its item-level scores are merged into the same score keys used
        by the controller, but severe hard-gate conflicts remain non-overridable.
        """
        anchor = _parse_dt(anchor_time) or datetime.fromisoformat(str(anchor_time)[:19])
        horizon_end = anchor + timedelta(hours=int(horizon_hours))
        events = [dict(x) for x in structured_events or []]
        sources = [dict(x) for x in evidence_sources or []]
        summaries = [dict(x) for x in model_assisted_summaries or []]
        channel_names = list(channel_names or [])
        candidates = list(adjusted_channel_candidates or [])
        conflicts: List[str] = []
        severe: List[str] = []

        source_score = self._source_score(sources, summaries, anchor, conflicts, severe)
        temporal_score = self._temporal_score(events, anchor, horizon_end, conflicts, severe)
        semantic_score = self._semantic_score(events)
        included_units, excluded_units, geo_score = self._units_and_geo(events, channel_names, candidates)
        residual_score, residual_summary = self._residual_score(historical_event_cases or [], residual_cases or [], events)

        if candidates and not included_units:
            conflicts.append("no_candidate_channel_has_event_match")
        evidence_validity = round((source_score + geo_score + temporal_score + semantic_score + residual_score) / 5.0, 4)
        base: Dict[str, Any] = {
            "source_validity_score": round(source_score, 4),
            "geo_consistency_score": round(geo_score, 4),
            "temporal_alignment_score": round(temporal_score, 4),
            "semantic_consistency_score": round(semantic_score, 4),
            "residual_support_score": round(residual_score, 4),
            "evidence_validity_score": evidence_validity,
            "thresholds": dict(self.thresholds),
            "conflict_flags": conflicts,
            "severe_conflict_flags": severe,
            "included_units": included_units,
            "excluded_units": excluded_units,
            "residual_support_summary": residual_summary,
            "audit_version": "hard_gate_v2",
        }
        if llm_scorer is None:
            return base
        llm_payload = llm_scorer.score(
            anchor_time=anchor_time,
            horizon_hours=horizon_hours,
            structured_events=events,
            evidence_sources=sources,
            model_assisted_summaries=summaries,
            channel_names=channel_names,
            adjusted_channel_candidates=candidates,
            historical_event_cases=historical_event_cases or [],
            residual_cases=residual_cases or [],
            hard_audit=base,
        )
        if llm_payload.get("llm_audit_status") == "parsed":
            for key in [
                "source_validity_score",
                "geo_consistency_score",
                "temporal_alignment_score",
                "semantic_consistency_score",
                "residual_support_score",
                "evidence_validity_score",
            ]:
                if key in llm_payload:
                    base[key] = round(float(llm_payload[key]), 4)
            if severe:
                base["severe_conflict_flags"] = severe
            base["audit_version"] = "hard_gate_plus_llm_v3"
        else:
            base["audit_version"] = "hard_gate_v2_llm_failed"
        for key, value in llm_payload.items():
            if key != "raw_response":
                base[key] = value
        return base

    @staticmethod
    def _source_score(sources: Sequence[dict], summaries: Sequence[dict], anchor: datetime, conflicts: List[str], severe: List[str]) -> float:
        if not sources and not summaries:
            conflicts.append("no_external_or_model_assisted_source")
            return 0.0
        scores = []
        for src in sources:
            score = 0.75 if src.get("accepted", True) else 0.30
            stype = str(src.get("source_type") or src.get("provider") or "source-assisted").lower()
            if stype not in ALLOWED_SOURCE_TYPES:
                score -= 0.20
                conflicts.append(f"unsupported_source_type:{stype}")
            title_text = f"{src.get('title','')} {src.get('snippet','')}".lower()
            if any(p in title_text for p in POST_EVENT_PATTERNS):
                score -= 0.25
                conflicts.append("possible_post_event_report")
            source_time = _parse_dt(src.get("source_time") or src.get("published_at") or src.get("publish_time") or src.get("date_published"))
            if source_time is None:
                conflicts.append("source_time_unknown")
                score -= 0.10
            elif source_time > anchor:
                conflicts.append("source_time_after_anchor")
                severe.append("source_time_after_anchor")
                score = min(score, 0.30)
            scores.append(max(0.0, min(1.0, score)))
        for item in summaries:
            if item.get("summary"):
                scores.append(0.65)
        return max(scores) if scores else 0.0

    @staticmethod
    def _temporal_score(events: Sequence[dict], anchor: datetime, horizon_end: datetime, conflicts: List[str], severe: List[str]) -> float:
        if not events:
            conflicts.append("no_structured_event_for_temporal_check")
            return 0.0
        scores = []
        for event in events:
            ev_time = _parse_dt(event.get("event_time"))
            if ev_time is None:
                conflicts.append("event_time_missing")
                scores.append(0.2)
            elif anchor <= ev_time < horizon_end:
                scores.append(0.95)
            elif anchor - timedelta(hours=6) <= ev_time < anchor:
                scores.append(0.55)
                conflicts.append("event_before_anchor_arrival_departure_tail")
            else:
                scores.append(0.15)
                severe.append("event_time_outside_horizon")
        return max(scores) if scores else 0.0

    @staticmethod
    def _semantic_score(events: Sequence[dict]) -> float:
        if not events:
            return 0.0
        vals = []
        for event in events:
            tier = str(event.get("impact_tier") or "").upper()
            has_type = bool(event.get("event_type") or event.get("event_category"))
            vals.append((0.65 if tier in {"A", "B"} else 0.35) + (0.25 if has_type else 0.0))
        return min(1.0, max(vals))

    @staticmethod
    def _units_and_geo(events: Sequence[dict], channel_names: Sequence[str], candidates: Sequence[str]):
        event_text = " ".join(str(ev.get(k, "") or "") for ev in events for k in ["title", "location", "venue_name", "channel_name", "station_name"])
        event_tokens = _tokens(event_text)
        explicit_channels = {str(ev.get("channel_name")) for ev in events if ev.get("channel_name")}
        included = []
        excluded = []
        candidate_set = set(candidates)
        for ch in channel_names:
            ch_tokens = _tokens(ch.replace("_", " "))
            token_overlap = bool(event_tokens & ch_tokens)
            explicit = ch in explicit_channels or any(ch.startswith(x.split("__", 1)[0]) for x in explicit_channels if "__" in x)
            selected = (not candidate_set) or ch in candidate_set
            if selected and (token_overlap or explicit):
                included.append({
                    "station_channel": ch,
                    "relation": "event location/station token match" if token_overlap else "explicit event-channel match",
                    "gate_score": 0.90 if explicit else 0.75,
                    "reason": "event venue/location is consistent with station/channel label",
                })
            else:
                reason = "not selected by adapter" if candidate_set and ch not in candidate_set else "no event-station-channel evidence"
                excluded.append({"station_channel": ch, "exclusion_reason": reason})
        if included:
            return included, excluded[:20], min(1.0, sum(x["gate_score"] for x in included) / len(included))
        return included, excluded[:20], 0.25 if events else 0.0

    @staticmethod
    def _residual_score(historical_cases: Sequence[dict], residual_cases: Sequence[str], events: Sequence[dict]):
        event_type_tokens = _tokens(" ".join(str(ev.get("event_type") or ev.get("event_category") or ev.get("title") or "") for ev in events))
        scores = []
        medians = []
        directions = []
        n_eff = 0
        for case in historical_cases:
            case_tokens = _tokens(str(case.get("event_type") or case.get("historical_event_title") or ""))
            median = _as_float(case.get("median_lp_moment_correction"), 0.0)
            iqr = abs(_as_float(case.get("iqr"), 0.0))
            n = int(case.get("n_eff") or case.get("sample_size") or 1)
            match = bool(event_type_tokens & case_tokens) if event_type_tokens and case_tokens else True
            score = 0.25 + (0.30 if match else 0.0) + (0.25 if abs(median) > 0.001 else 0.0) + (0.20 if n >= 5 else 0.0)
            if iqr > 0.08:
                score -= 0.10
            scores.append(max(0.0, min(1.0, score)))
            medians.append(median)
            directions.append("increase" if median > 0 else "decrease" if median < 0 else "neutral")
            n_eff += n
        for text in residual_cases:
            m = re.search(r"correction=([+-]?[0-9.]+)%", text)
            n = re.search(r"N=([0-9]+)", text)
            median = float(m.group(1)) / 100.0 if m else 0.0
            count = int(n.group(1)) if n else 1
            scores.append(0.50 + (0.25 if abs(median) > 0.001 else 0.0) + (0.20 if count >= 5 else 0.0))
            medians.append(median)
            directions.append("increase" if median > 0 else "decrease" if median < 0 else "neutral")
            n_eff += count
        if not scores:
            return 0.0, {"direction_agreement": "none", "median_correction": 0.0, "n_eff": 0, "confidence_cap": 0.0}
        median_correction = sorted(medians)[len(medians)//2] if medians else 0.0
        dominant = max(set(directions), key=directions.count) if directions else "neutral"
        return min(1.0, max(scores)), {
            "direction_agreement": dominant,
            "median_correction": round(float(median_correction), 4),
            "n_eff": int(n_eff),
            "confidence_cap": round(min(0.90, 0.45 + 0.05 * min(n_eff, 8)), 4),
        }
