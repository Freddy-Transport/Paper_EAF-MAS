"""事件语义 enrichment — 规则优先，可选 LLM 增强."""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from agents import config as cfg
from agents.event_tier import annotate_event_tier, infer_event_category, match_venue
from agents.schemas import EventInfo

logger = logging.getLogger(__name__)

ENRICHMENT_PROMPT = """You are a NYC subway ridership analyst. Enrich this permit/event record for forecast adjustment.

Raw event:
- title: {title}
- type: {event_type}
- time: {event_time}
- location: {location}
- content: {content}
- channel: {channel_name}
- distance_m: {distance_m}

Output ONLY valid JSON:
{{
  "normalized_title": "English title <=80 chars",
  "event_category": "sports|concert|parade|festival|disruption|admin|other",
  "impact_tier": "A|B|C|D",
  "venue_or_route": "specific venue or street",
  "likely_crowd_size": "small|medium|large|mega",
  "subway_impothesis": "2-3 sentences on HOW/WHEN subway ridership at nearby stations changes",
  "peak_hours_relative": "[-2, 4] style list for hours relative to start",
  "should_adjust_forecast": true or false
}}
"""


def enrich_event_rule(event: EventInfo, target_channels: Optional[List[str]] = None) -> EventInfo:
    """规则 enrichment，不调用 LLM."""
    ev = annotate_event_tier(event, target_channels)
    tier = ev.impact_tier
    category = ev.event_category or infer_event_category(ev, tier)
    venue = match_venue(ev)
    venue_name = venue["id"] if venue else (ev.venue_name or ev.location or "unknown location")

    crowd = "small"
    if tier == "A":
        crowd = "large" if category in ("sports", "concert", "parade") else "medium"
    elif tier == "B":
        crowd = "medium"

    hypothesis_parts = []
    if tier in ("A", "B"):
        if category == "sports":
            hypothesis_parts.append(
                f"Pre-event arrivals and post-game dispersal likely increase entries at stations near {venue_name}."
            )
        elif category in ("parade", "festival"):
            hypothesis_parts.append(
                f"Crowd gathering along the route may elevate nearby station ridership during event hours."
            )
        elif category == "disruption":
            hypothesis_parts.append(
                "Service disruption may reroute passengers or reduce entries at affected stations."
            )
        else:
            hypothesis_parts.append(
                f"Localized activity near {venue_name} may cause a modest residual ridership change."
            )
    else:
        hypothesis_parts.append(
            "Administrative or hyper-local permit; unlikely to measurably shift subway ridership."
        )

    enriched = (
        f"[Tier {tier}] {ev.title or 'Event'} | category={category} | venue={venue_name} | "
        f"crowd={crowd}. {' '.join(hypothesis_parts)}"
    )

    return ev.model_copy(
        update={
            "event_category": category,
            "enriched_description": enriched,
        }
    )


def enrich_events(
    events: List[EventInfo],
    target_channels: Optional[List[str]] = None,
    use_llm: bool = False,
) -> List[EventInfo]:
    """批量 enrichment."""
    if not events:
        return []
    use_llm = use_llm or os.environ.get("EVENT_ENRICHMENT_LLM", "").lower() in ("1", "true", "yes")
    out: List[EventInfo] = []
    llm = _init_llm() if use_llm else None

    for ev in events:
        base = enrich_event_rule(ev, target_channels)
        if llm is not None:
            try:
                base = _enrich_with_llm(llm, base)
            except Exception as e:
                logger.warning("LLM enrichment failed for '%s': %s", (ev.title or "")[:30], e)
        out.append(base)
    return out


def _init_llm():
    if not cfg.LLM_API_KEY:
        return None
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=cfg.LLM_API_KEY,
            base_url=cfg.LLM_BASE_URL,
            model=cfg.LLM_MODEL,
            temperature=0.1,
            max_tokens=512,
        )
    except ImportError:
        return None


def _enrich_with_llm(llm, event: EventInfo) -> EventInfo:
    prompt = ENRICHMENT_PROMPT.format(
        title=event.title,
        event_type=event.event_type or "",
        event_time=event.event_time,
        location=event.location or "",
        content=(event.content or "")[:500],
        channel_name=event.channel_name or "",
        distance_m=event.distance_m if event.distance_m is not None else "unknown",
    )
    resp = llm.invoke(prompt)
    text = resp.content.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    data = json.loads(text)
    tier = str(data.get("impact_tier", event.impact_tier)).upper()
    if tier not in ("A", "B", "C", "D"):
        tier = event.impact_tier
    adjust = bool(data.get("should_adjust_forecast", tier in ("A", "B")))
    enriched = (
        f"[Tier {tier}] {data.get('normalized_title', event.title)} | "
        f"category={data.get('event_category', event.event_category)} | "
        f"crowd={data.get('likely_crowd_size', '?')}. "
        f"{data.get('subway_hypothesis', '')}"
    )
    return event.model_copy(
        update={
            "impact_tier": tier,
            "event_category": data.get("event_category", event.event_category),
            "enriched_description": enriched,
            "should_adjust_forecast": adjust and tier in ("A", "B"),
            "impact_confidence": "high" if tier == "A" else "medium",
        }
    )
