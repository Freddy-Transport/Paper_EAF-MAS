"""事件影响分级（Tier A/B/C/D）与客流验真 — 在线/离线共用.

Tier A: 确定性高影响（大型体育、游行、满场演唱会、重大封站）
Tier B: 可能高影响（中型 festival、大型 street event）
Tier C: 低影响行政/许可（filming、草坪关闭、小促销）
Tier D: 与目标站点无关或远场
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from agents import config as cfg
from agents.schemas import EventInfo

logger = logging.getLogger(__name__)

# ── Tier A：事件类型白名单 ────────────────────────────────────────────────────
TIER_A_EVENT_TYPES = {
    "parade", "marathon", "concert", "sport - adult", "sport - youth",
    "athletic race / tour", "athletic race", "special event",
    "street festival", "single block festival",
    "service disruption",
}

TIER_B_EVENT_TYPES = {
    "plaza partner event", "plaza event", "street event",
    "theater load in and load outs", "production event",
}

# ── Tier C/D：黑名单关键词（小写匹配）────────────────────────────────────────
TIER_C_KEYWORDS = [
    "filming", "walk-through", "walkthrough", "inspection", "setup", "breakdown",
    "parking permit", "parking only", "coffee", "giveaway", "cart giveaway",
    "health outreach", "covid testing", "little league", "softball (little",
    "baseball - 12", "baseball - 13", "soccer - non", "soccer -regulation",
    "farmers market", "greenmarket", "miscellaneous", "wedding", "private party",
    "lawn closure", "lawn closed", "great lawn", "sheep meadow", "in the park",
    "filming permit", "walk through",
]

TIER_A_KEYWORDS = [
    "nba", "nhl", "nfl", "mlb", "playoff", "championship", "sold out", "sold-out",
    "marathon", "parade", "nyrr", "new year's eve", "new years eve", "nye",
    "concert", "tour date", "stanley cup", "world series",
]

TIER_B_KEYWORDS = [
    "street festival", "block party", "festival", "fair", "rally", "march",
    "broadway", "theater district", "tony awards",
]

# 仅当 event_type 含 sport 且命中大型场馆时才升 A
MAJOR_VENUE_KEYWORDS = [
    "barclays", "madison square garden", "yankee stadium", "citi field", "usta",
    "javits", "times square", "msg", "forest hills stadium",
]

MAX_DISTANCE_TIER_B_M = 1200
MAX_DISTANCE_TIER_A_M = 800

KB_ELIGIBLE_TIERS = frozenset({"A", "B"})
ADJUSTABLE_TIERS = frozenset({"A", "B"})

TIER_CAP = {"A": 0.08, "B": 0.05, "C": 0.0, "D": 0.0}


@lru_cache(maxsize=1)
def load_venue_station_map(path: Optional[str] = None) -> Dict:
    p = Path(path or cfg.DATA_DIR) / "venue_station_map.json"
    if not p.is_file():
        return {"venues": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _event_text_blob(event: Union[EventInfo, dict]) -> str:
    if isinstance(event, EventInfo):
        parts = [event.title, event.content or "", event.location or "", event.event_type or ""]
    else:
        parts = [
            event.get("title", ""),
            event.get("content", ""),
            event.get("location", ""),
            event.get("event_type", ""),
        ]
    return " ".join(str(p) for p in parts if p).lower()


def _get_field(event: Union[EventInfo, dict], name: str, default=None):
    if isinstance(event, EventInfo):
        return getattr(event, name, default)
    return event.get(name, default)


def match_venue(event: Union[EventInfo, dict]) -> Optional[dict]:
    """若文本命中场馆表，返回场馆记录."""
    text = _event_text_blob(event)
    for venue in load_venue_station_map().get("venues", []):
        if any(alias in text for alias in venue.get("aliases", [])):
            return venue
    return None


def classify_impact_tier(
    event: Union[EventInfo, dict],
    target_channels: Optional[List[str]] = None,
) -> str:
    """返回 A / B / C / D."""
    text = _event_text_blob(event)
    etype = (_get_field(event, "event_type") or "").lower().strip()
    distance = _get_field(event, "distance_m")
    station_id = str(_get_field(event, "station_complex_id") or "")
    channel = str(_get_field(event, "channel_name") or "")

    # 目标站地理不匹配（优先于黑名单，避免 Central Park 误留 C）
    if target_channels and any("N060" in t for t in target_channels):
        park_kw = ["central park", "great lawn", "lawn closure", "museum of natural history", "81 st"]
        if any(kw in text for kw in park_kw):
            return "D"

    # 黑名单 → C
    if any(kw in text for kw in TIER_C_KEYWORDS):
        # 大型游行/马拉松标题里可能含 "park"，排除误杀
        if not any(kw in text for kw in ("parade", "marathon", "nyrr")):
            return "C"

    disruption_terms = ("service disruption", "station closure", "street closure", "road closure", "closed", "closure", "maintenance", "outage")
    if (etype == "construction" or "routine construction" in text) and not any(kw in text for kw in disruption_terms):
        return "C"

    venue = match_venue(event)
    if venue:
        default = venue.get("default_tier", "B")
        # 场馆活动但距离过远 → D
        if distance is not None and float(distance) > MAX_DISTANCE_TIER_B_M:
            return "D"
        v_stations = venue.get("station_complex_ids", [])
        if target_channels and v_stations:
            if not _targets_overlap_stations(target_channels, v_stations, channel, station_id):
                return "D"
        return default

    # 类型白名单
    if etype in TIER_A_EVENT_TYPES or any(kw in text for kw in TIER_A_KEYWORDS):
        if distance is not None and float(distance) > MAX_DISTANCE_TIER_A_M:
            return "B"
        return "A"

    if etype in TIER_B_EVENT_TYPES or any(kw in text for kw in TIER_B_KEYWORDS):
        return "B"

    # 大型场馆关键词 + 体育/演唱会
    if any(v in text for v in MAJOR_VENUE_KEYWORDS):
        if any(kw in text for kw in ("game", "match", "concert", "show", "vs ", " vs")):
            return "A"
        return "B"

    # 明确服务中断（非草坪类）。Generic/routine construction 只作为低影响噪声，
    # 除非文本明确出现 closure/service disruption/maintenance/closed 等运营影响证据。
    disruption_terms = ("service disruption", "station closure", "street closure", "road closure", "closed", "closure", "maintenance", "outage")
    if any(kw in text for kw in disruption_terms):
        if not any(kw in text for kw in ("lawn", "park drive", "grass")):
            return "A"

    if distance is not None and float(distance) > 1500:
        return "D"

    return "C"


def _targets_overlap_stations(
    target_channels: List[str],
    venue_station_ids: List[str],
    event_channel: str,
    event_station_id: str,
) -> bool:
    for tid in venue_station_ids:
        if event_station_id and event_station_id.startswith(tid):
            return True
        if event_channel and tid in event_channel:
            return True
        for tch in target_channels:
            if tid in tch:
                return True
    return False


def infer_event_category(event: Union[EventInfo, dict], tier: str) -> str:
    text = _event_text_blob(event)
    etype = (_get_field(event, "event_type") or "").lower()
    if "parade" in text or "marathon" in etype:
        return "parade"
    if any(k in text for k in ("nba", "yankee", "mets", "knicks", "nets", "game", "sport")):
        return "sports"
    if "concert" in text or "concert" in etype:
        return "concert"
    if "festival" in text or "festival" in etype:
        return "festival"
    if any(k in text for k in ("construction", "closure", "disruption")):
        return "disruption"
    if tier in ("C", "D"):
        return "admin"
    return "other"


def annotate_event_tier(
    event: EventInfo,
    target_channels: Optional[List[str]] = None,
) -> EventInfo:
    """写入 impact_tier / should_adjust_forecast / event_category."""
    tier = classify_impact_tier(event, target_channels)
    category = infer_event_category(event, tier)
    confidence = "high" if tier == "A" else ("medium" if tier == "B" else "low")
    adjust = tier in ADJUSTABLE_TIERS
    return event.model_copy(
        update={
            "impact_tier": tier,
            "event_category": category,
            "impact_confidence": confidence,
            "should_adjust_forecast": adjust,
        }
    )


def filter_adjustable_events(
    events: List[EventInfo],
    target_channels: Optional[List[str]] = None,
) -> Tuple[List[EventInfo], int]:
    """仅保留 Tier A/B 且 should_adjust_forecast=True 的事件."""
    out: List[EventInfo] = []
    dropped = 0
    for ev in events:
        annotated = annotate_event_tier(ev, target_channels)
        if annotated.should_adjust_forecast and annotated.impact_tier in ADJUSTABLE_TIERS:
            out.append(annotated)
        else:
            dropped += 1
    return out, dropped


def tier_is_kb_eligible(tier: str) -> bool:
    return tier in KB_ELIGIBLE_TIERS


def _compute_mean_baseline_residual(
    actual_vals: np.ndarray,
    timestamps: pd.DatetimeIndex,
    baseline: Dict[int, float],
) -> float:
    residuals = []
    for val, ts in zip(actual_vals, timestamps):
        if np.isnan(val):
            continue
        h = int(ts.dayofweek * 24 + ts.hour)
        bl_val = baseline.get(h, 0.0)
        if bl_val > 50:
            residuals.append((val - bl_val) / bl_val)
    return float(np.mean(residuals)) if residuals else 0.0


def passes_ridership_validation(
    event: dict,
    baselines: Dict[str, Dict[int, float]],
    gt_full: pd.DataFrame,
    window_hours: int = 8,
    min_abs_baseline_residual: float = 0.05,
) -> bool:
    """离线：事件窗口内相对周期基线的平均残差是否达到阈值."""
    ev_time_str = event.get("event_time", "")
    ch_name = event.get("channel_name", "")
    if not ev_time_str or not ch_name or ch_name not in baselines:
        return False
    if ch_name not in gt_full.columns:
        return False
    try:
        ev_dt = pd.to_datetime(ev_time_str)
    except Exception:
        return False

    baseline = baselines[ch_name]
    ts_window = pd.date_range(ev_dt, periods=window_hours, freq="h")
    actual_vals = gt_full[ch_name].reindex(ts_window).values.astype(float)
    if np.all(np.isnan(actual_vals)):
        return False

    bl_resid = _compute_mean_baseline_residual(actual_vals, ts_window, baseline)
    return abs(bl_resid) >= min_abs_baseline_residual


def filter_events_for_knowledge_base(
    events: List[dict],
    baselines: Optional[Dict[str, Dict[int, float]]] = None,
    gt_full: Optional[pd.DataFrame] = None,
    require_traffic_validation: bool = True,
) -> List[dict]:
    """离线入库筛选：Tier A/B + 可选客流验真."""
    kept: List[dict] = []
    for ev in events:
        tier = classify_impact_tier(ev)
        if not tier_is_kb_eligible(tier):
            continue
        if require_traffic_validation and baselines is not None and gt_full is not None:
            if not passes_ridership_validation(ev, baselines, gt_full):
                continue
        ev = dict(ev)
        ev["impact_tier"] = tier
        kept.append(ev)
    logger.info(
        "KB filter: %d / %d events kept (Tier A/B%s)",
        len(kept),
        len(events),
        " + traffic validated" if require_traffic_validation else "",
    )
    return kept
