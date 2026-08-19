from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


@dataclass
class PhysicalEvent:
    event_id: str
    title: str
    event_time: str
    date: str
    location: str = ""
    event_type: str = ""
    content: str = ""
    source: str = "unknown"
    duplicate_count: int = 0
    affected_channels: List[str] = field(default_factory=list)
    station_complex_ids: List[str] = field(default_factory=list)
    station_ranks: List[int] = field(default_factory=list)
    min_distance_m: Optional[float] = None
    raw_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RulePrefilterResult:
    high_priority: bool
    rule_score: float
    category: str
    reason: str


class MajorEventDecision(BaseModel):
    is_major_event: bool = False
    major_event_type: Literal[
        "sports", "concert", "festival", "parade", "service_disruption", "conference", "ceremony", "street_event", "other", "none"
    ] = "none"
    crowd_scale: Literal["none", "small", "medium", "large", "mega"] = "none"
    transit_impact_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_direction: Literal["increase", "decrease", "neutral", "unknown"] = "unknown"
    affected_scope: Literal["none", "station", "multi_station", "borough", "citywide", "unknown"] = "unknown"
    traffic_evidence_used: bool = False
    keep_for_modeling: bool = False
    reason: str = ""
    needs_review: bool = False
    raw_response: Optional[str] = None

    def modeling_eligible(self, threshold: float = 0.65) -> bool:
        if self.major_event_type == "service_disruption" and self.transit_impact_likelihood >= threshold:
            return True
        return self.transit_impact_likelihood >= threshold and self.crowd_scale in {"large", "mega"}
