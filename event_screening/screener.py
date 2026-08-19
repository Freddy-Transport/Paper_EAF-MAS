from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

from event_screening.models import PhysicalEvent, RulePrefilterResult

NOISE_KEYWORDS = [
    "youth", "little league", "soccer - non", "soccer -regulation", "farmers market", "greenmarket",
    "filming", "shooting permit", "setup", "breakdown", "parking", "walk-through", "walkthrough",
    "lawn closure", "lawn closed", "great lawn", "winter closure", "picnic", "party", "miscellaneous",
]
MAJOR_VENUE_KEYWORDS = [
    "yankee stadium", "madison square garden", "msg", "barclays", "citi field", "usta", "javits",
    "times square", "radio city", "lincoln center", "forest hills stadium", "apollo",
]
MAJOR_EVENT_KEYWORDS = [
    "parade", "marathon", "concert", "tour", "playoff", "championship", "world series", "new year's eve",
    "new years eve", "nyrr", "festival", "street festival", "rally", "march",
]
SPORT_KEYWORDS = ["yankees", "mets", "knicks", "nets", "rangers", "islanders", "liberty", "game", " vs ", "match"]
DISRUPTION_KEYWORDS = ["service disruption", "station closure", "closed", "closure", "outage", "maintenance"]


def _as_dict(row) -> dict:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    return dict(row)


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    text = re.sub(r"[^a-z0-9\s:/_-]", "", text)
    return text[:80]


def _extract_content_value(content: str, key: str) -> Optional[str]:
    match = re.search(rf"{re.escape(key)}=([^;|]+)", content or "", flags=re.I)
    return match.group(1).strip() if match else None


def _extract_distance(row: dict) -> Optional[float]:
    for key in ["distance_m", "distance_to_station"]:
        val = row.get(key)
        if val not in (None, ""):
            try:
                return float(str(val).replace("m", ""))
            except Exception:
                pass
    match = re.search(r"distance(?:_to_station)?=([0-9.]+)m?", str(row.get("content", "")), flags=re.I)
    if match:
        return float(match.group(1))
    return None


def _physical_key(row: dict) -> str:
    title = _norm(row.get("title") or _extract_content_value(row.get("content", ""), "event_name") or "")
    date = str(row.get("event_time") or row.get("start_datetime") or "")[:10]
    venue = _norm(_extract_content_value(row.get("content", ""), "matched_venue") or row.get("location") or "")[:50]
    raw = f"{title}|{date}|{venue}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def deduplicate_physical_events(events: Iterable[dict]) -> List[PhysicalEvent]:
    grouped: "OrderedDict[str, PhysicalEvent]" = OrderedDict()
    for raw in events:
        row = _as_dict(raw)
        key = _physical_key(row)
        title = row.get("title") or _extract_content_value(row.get("content", ""), "event_name") or ""
        event_time = str(row.get("event_time") or _extract_content_value(row.get("content", ""), "start_datetime") or "")
        location = str(row.get("location") or _extract_content_value(row.get("content", ""), "event_location") or "")
        event_type = str(row.get("event_type") or _extract_content_value(row.get("content", ""), "event_type") or "")
        if key not in grouped:
            grouped[key] = PhysicalEvent(
                event_id=key,
                title=title,
                event_time=event_time,
                date=event_time[:10],
                location=location,
                event_type=event_type,
                content=str(row.get("content", "")),
                source=str(row.get("source", "unknown")),
            )
        item = grouped[key]
        item.duplicate_count += 1
        item.raw_events.append(row)
        ch = row.get("channel_name")
        sid = row.get("station_complex_id")
        if ch and ch not in item.affected_channels:
            item.affected_channels.append(str(ch))
        if sid and sid not in item.station_complex_ids:
            item.station_complex_ids.append(str(sid))
        try:
            rank = int(row.get("station_rank"))
            if rank not in item.station_ranks:
                item.station_ranks.append(rank)
        except Exception:
            pass
        dist = _extract_distance(row)
        if dist is not None:
            item.min_distance_m = dist if item.min_distance_m is None else min(item.min_distance_m, dist)
    return list(grouped.values())


def event_text(event: PhysicalEvent) -> str:
    return " ".join([event.title, event.event_type, event.location, event.content]).lower()


def rule_prefilter_event(event: PhysicalEvent, traffic_evidence: Optional[Dict] = None) -> RulePrefilterResult:
    text = event_text(event)
    score = 0.0
    reasons: List[str] = []
    category = "other"

    if any(k in text for k in NOISE_KEYWORDS):
        score -= 0.45
        reasons.append("noise_keyword")
        category = "noise"
    if any(k in text for k in DISRUPTION_KEYWORDS) and not any(k in text for k in ["lawn", "park drive", "grass"]):
        score += 0.85
        reasons.append("service_disruption")
        category = "service_disruption"
    if any(k in text for k in MAJOR_VENUE_KEYWORDS):
        score += 0.55
        reasons.append("major_venue")
    if any(k in text for k in SPORT_KEYWORDS):
        score += 0.25
        reasons.append("sports_keyword")
        category = "sports"
    if any(k in text for k in MAJOR_EVENT_KEYWORDS):
        score += 0.45
        reasons.append("major_event_keyword")
        if category == "other":
            category = "large_gathering"
    if event.duplicate_count >= 5:
        score += 0.10
        reasons.append("multi_station_duplicates")
    if event.min_distance_m is not None and event.min_distance_m <= 500:
        score += 0.10
        reasons.append("near_station")
    if traffic_evidence:
        if abs(float(traffic_evidence.get("z_score", 0.0))) >= 2.5 and not traffic_evidence.get("low_volume_warning", False):
            score += 0.25
            reasons.append("traffic_anomaly")
    high_priority = score >= 0.65
    if not reasons:
        reasons.append("weak_or_no_major_event_signal")
    if category == "noise" and high_priority:
        high_priority = score >= 0.85
    return RulePrefilterResult(high_priority=bool(high_priority), rule_score=max(0.0, min(1.0, score)), category=category, reason=", ".join(reasons))


def final_keep_for_modeling(decision, threshold: float = 0.65) -> bool:
    if getattr(decision, "needs_review", False):
        return False
    if getattr(decision, "major_event_type", "") == "service_disruption" and decision.transit_impact_likelihood >= threshold:
        return True
    return bool(decision.keep_for_modeling and decision.transit_impact_likelihood >= threshold and decision.crowd_scale in {"large", "mega"})
