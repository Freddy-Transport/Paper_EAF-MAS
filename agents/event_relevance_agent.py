"""事件相关性筛选智能体。

用于 Top128 站点通道场景：先用规则和结构化字段筛出可能显著影响站点
客流的事件，再把少量高价值事件交给 LLM/RAG 分析，避免对所有 permit
记录做语义调用。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from agents.event_tier import annotate_event_tier
from agents.schemas import EventInfo, EventRelevance

logger = logging.getLogger(__name__)

# ── LLM 批量重打分 Prompt ─────────────────────────────────────────────────────
_LLM_RESCORE_PROMPT = """\
You are a NYC subway ridership analyst. Rate the likely impact of each event on nearby subway station ridership.

Scoring guide:
- 0.9-1.0: Major sporting events / sold-out concerts at large venues (Yankee Stadium, MSG, Barclays)
- 0.7-0.9: Street festivals, parades, large cultural events with crowd gathering
- 0.5-0.7: Medium events, some crowd gathering but limited transit impact
- 0.3-0.5: Small or localized events, minor indirect impact
- 0.0-0.3: Administrative, maintenance, filming permits, or tiny gatherings

Events:
{events_block}

Respond ONLY as a valid JSON array (no extra text):
[{{"id": 1, "relevance_score": 0.95, "semantic_category": "sports", "brief_reason": "..."}}]

semantic_category must be one of: sports / concert / festival / ceremony / street_event / admin / other
"""

HIGH_IMPACT_KEYWORDS = {
    "sports": ["nba", "wnba", "nfl", "mlb", "mets", "yankees", "knicks", "nets", "game", "match", "stadium", "sports"],
    "concert": ["concert", "tour", "music", "live", "festival", "performance", "show"],
    "festival": ["festival", "parade", "fair", "carnival", "celebration", "block party"],
    "conference": ["conference", "expo", "convention", "job fair", "career fair", "recruiting", "招聘"],
    "ceremony": ["ceremony", "graduation", "commencement", "award", "memorial"],
    "street_event": ["street fair", "open street", "plaza", "avenue", "march", "rally"],
}

# 真正低影响事件（纯行政/拍摄/停车许可等，不影响客流）
ADMIN_LOW_IMPACT_KEYWORDS = [
    "filming", "walk-through", "inspection", "setup", "breakdown", "parking", "permit",
]

# 服务中断类事件（封站/施工/维护会造成下降型客流冲击，不应被低影响惩罚过滤，
# 而应标记为 semantic_category="disruption"，引导 LLM 输出 impact_direction=decrease）
NEGATIVE_DISRUPTION_KEYWORDS = [
    "service disruption", "station closure", "street closure", "road closure",
    "lane closure", "closure", "closed", "maintenance", "outage",
]

MAJOR_VENUE_KEYWORDS = [
    "barclays", "madison square garden", "yankee stadium", "citi field", "usta",
    "javits", "times square", "rockefeller", "radio city", "lincoln center",
    "apollo", "forest hills stadium", "brooklyn mirage", "avant gardner",
]

# 各典型站点的地理关键词白名单（只有这些关键词的事件才被认为直接影响该站）
# key 是 station_complex_id 前缀，value 是事件文本中必须包含的地理词
STATION_GEO_WHITELIST: Dict[str, List[str]] = {
    "N060": ["times sq", "times square", "broadway", "midtown", "42nd", "seventh ave",
             "eighth ave", "seventh av", "eighth av", "herald sq", "42 st",
             "port authority", "penn station", "34 st", "hells kitchen",
             "theater district", "theatre district"],
    "N044": ["central park", "81 st", "museum", "upper west", "columbus", "amsterdam"],
    "R022": ["lexington", "51 st", "midtown east"],
}

# 这些关键词说明事件主要在远离 Times Sq 的区域（Central Park / 博物馆区）
# 若目标站是 N060，这类事件需要降权
CENTRAL_PARK_KEYWORDS = [
    "central park", "the park", "in the park", "lawn", "great lawn", "sheep meadow",
    "bandshell", "bethesda", "park drive", "park road", "museum of natural history",
    "81 st", "86 st", "columbus circle",
]


def load_channel_map(path: str | Path | None) -> Dict[str, dict]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    channels = payload.get("channels", payload if isinstance(payload, list) else [])
    return {str(row["channel_name"]): row for row in channels if row.get("channel_name")}


class EventRelevanceAgent:
    """规则优先的事件相关性筛选器。"""

    def __init__(
        self,
        channel_map: Optional[Dict[str, dict]] = None,
        channel_map_path: str | Path | None = None,
        relevance_threshold: float = 0.45,
        max_events: Optional[int] = 80,
    ):
        self.channel_map = channel_map or load_channel_map(channel_map_path)
        self.relevance_threshold = relevance_threshold
        self.max_events = max_events

    def filter_events(
        self,
        events: List[EventInfo],
        forecast_timestamps: Optional[List[str]] = None,
        target_channels: Optional[List[str]] = None,
    ) -> Tuple[List[EventInfo], List[EventRelevance]]:
        windowed = self._filter_window(events, forecast_timestamps)
        scored_all = [self.score_event(event, target_channels=target_channels) for event in windowed]
        # Tier 预筛：仅 A/B 进入相关性打分（C/D 不送 LLM / 知识库）
        tiered: List[EventInfo] = []
        dropped_tier = 0
        for ev in windowed:
            ann = annotate_event_tier(ev, target_channels)
            if ann.impact_tier in ("A", "B"):
                tiered.append(ann)
            else:
                dropped_tier += 1
        if dropped_tier:
            logger.info(
                "EventRelevanceAgent: tier filter dropped %d C/D events",
                dropped_tier,
            )
        windowed = tiered
        scored = [self.score_event(event, target_channels=target_channels) for event in windowed]
        keep_ids = {
            relevance.event_key
            for relevance in scored
            if relevance.is_high_impact_candidate and relevance.relevance_score >= self.relevance_threshold
        }
        filtered = [event for event in windowed if self._event_key(event) in keep_ids]

        # ── 同一物理事件的多通道副本去重计数 ────────────────────────────────
        # 问题：同一事件（如 "Netflix熨斗区促销"）被数据管线匹配到6个附近站点，
        # 形成6个副本，全部送 LLM 会浪费 30% 的分析配额。
        # 策略：按 (标题前30字, 日期) 识别物理事件组；max_events 限制的是
        # 不同物理事件的数量（而非 channel-event 对的数量），保证每个物理事件
        # 的所有通道副本都会被分析，LLM 依然能对每个通道给出精确影响幅度。
        if self.max_events is not None and len(filtered) > self.max_events:
            score_by_key = {r.event_key: r.relevance_score for r in scored}

            # 为每个事件计算"物理事件组ID"
            def _phys_id(ev: EventInfo) -> str:
                title_norm = (ev.title or "")[:30].strip().lower()
                date_part  = (ev.event_time or "")[:10]
                return f"{title_norm}|{date_part}"

            # 按物理事件组汇聚最高分
            phys_best_score: dict = {}
            for ev in filtered:
                pid  = _phys_id(ev)
                key  = self._event_key(ev)
                s    = score_by_key.get(key, 0.0)
                if pid not in phys_best_score or s > phys_best_score[pid][0]:
                    phys_best_score[pid] = (s, key)

            # 取分数最高的前 max_events 个物理事件
            top_phys = sorted(phys_best_score.values(), key=lambda x: -x[0])
            top_phys_ids = set()
            for _, representative_key in top_phys[: self.max_events]:
                # 找到代表键对应的物理事件ID
                for ev in filtered:
                    if self._event_key(ev) == representative_key:
                        top_phys_ids.add(_phys_id(ev))
                        break

            # 保留所有属于入选物理事件的通道副本
            filtered = [ev for ev in filtered if _phys_id(ev) in top_phys_ids]

            n_phys   = len(top_phys_ids)
            n_total  = len(filtered)
            logger.info(
                "EventRelevanceAgent: dedup → %d distinct physical events "
                "→ %d channel-event pairs sent to LLM",
                n_phys, n_total,
            )
        # ──────────────────────────────────────────────────────────────────────

        logger.info("EventRelevanceAgent: kept %d / %d events", len(filtered), len(events))
        return filtered, scored_all

    def score_event(
        self,
        event: EventInfo,
        target_channels: Optional[List[str]] = None,
    ) -> EventRelevance:
        text = self._event_text(event)
        category, category_score = self._classify(text, event.event_type)
        score = category_score
        evidence: List[str] = []
        if category != "unknown":
            evidence.append(f"semantic_category={category}")

        if any(keyword in text for keyword in MAJOR_VENUE_KEYWORDS):
            score += 0.15
            evidence.append("major_venue_keyword")

        affected_channels = self._extract_affected_channels(event)
        if affected_channels:
            score += 0.1
            evidence.append("station_channel_matched")

        distance = self._extract_distance(event)
        if distance is not None:
            if distance <= 300:
                score += 0.1
                evidence.append("distance<=300m")
            elif distance <= 800:
                score += 0.05
                evidence.append("distance<=800m")

        # 纯行政/拍摄类：降低评分（不影响客流）
        if any(keyword in text for keyword in ADMIN_LOW_IMPACT_KEYWORDS):
            score -= 0.3
            evidence.append("admin_low_impact_keyword")

        if "construction" in text and not any(keyword in text for keyword in NEGATIVE_DISRUPTION_KEYWORDS):
            score -= 0.3
            evidence.append("generic_construction_low_impact")

        # 服务中断类：不惩罚评分，但覆盖 category 为 disruption（方向=decrease）
        expected_direction: Optional[str] = None
        # 仅当是真正的站点/交通服务中断时才触发（排除 "lawn closure" 类公园封闭）
        _is_transit_disruption = any(kw in text for kw in NEGATIVE_DISRUPTION_KEYWORDS) and not any(
            park_kw in text for park_kw in ["lawn", "field", "grass", "park drive", "park road"]
        )
        if _is_transit_disruption:
            category = "disruption"
            expected_direction = "decrease"
            evidence.append("negative_disruption_keyword")
            # 若原始分过低导致事件被过滤，给一个最低保底分保证进入候选集
            score = max(score, self.relevance_threshold)

        # ── 地理相关性修正（按目标站点过滤区域不匹配事件）──────────────────────
        # 放在 disruption 保底之后，确保地理惩罚可以覆盖非目标区域的 disruption 保底
        if target_channels:
            # 找到与目标通道匹配的地理白名单
            geo_whitelist: List[str] = []
            for ch in target_channels:
                for prefix, kws in STATION_GEO_WHITELIST.items():
                    if prefix in ch:
                        geo_whitelist.extend(kws)
                        break

            # 检测是否命中目标站的地理白名单（地理增益）
            if geo_whitelist and any(kw in text for kw in geo_whitelist):
                score += 0.10
                evidence.append("geo_whitelist_match")

            # 检测是否命中 Central Park 区域词（对 N060 目标站降权）
            is_target_midtown = any("N060" in ch for ch in target_channels)
            if is_target_midtown and any(kw in text for kw in CENTRAL_PARK_KEYWORDS):
                score -= 0.25
                evidence.append("central_park_penalty_for_N060")

            # 如果事件的 affected_channels 与目标通道完全不重叠，则轻微降权
            # 使用前缀匹配：N060__Times_Sq_42_St 是 N060__Times_Sq_42_St_N_Q_R_W_... 的前缀
            def _channels_overlap(ev_ch: str, tgt_chs: List[str]) -> bool:
                for t in tgt_chs:
                    if ev_ch in t or t in ev_ch:
                        return True
                return False

            if affected_channels and not any(
                _channels_overlap(ch, target_channels) for ch in affected_channels
            ):
                score -= 0.10
                evidence.append("channel_mismatch_penalty")
        # ────────────────────────────────────────────────────────────────────────

        if self._looks_like_all_day_admin_event(event, text):
            # 仅对非 disruption 类事件施加全天行政惩罚
            if expected_direction is None:
                score -= 0.2
                evidence.append("all_day_admin_like")

        score = max(0.0, min(1.0, score))
        return EventRelevance(
            event_key=self._event_key(event),
            title=event.title,
            event_time=event.event_time,
            station_complex_id=self._extract_station_id(event),
            affected_channels=affected_channels,
            semantic_category=category,
            relevance_score=round(score, 4),
            is_high_impact_candidate=score >= self.relevance_threshold,
            evidence=evidence,
            expected_direction=expected_direction,
        )

    @staticmethod
    def _filter_window(events: List[EventInfo], forecast_timestamps: Optional[List[str]]) -> List[EventInfo]:
        if not forecast_timestamps:
            return events
        try:
            start = pd.to_datetime(forecast_timestamps[0])
            end = pd.to_datetime(forecast_timestamps[-1])
        except Exception:
            return events
        filtered = []
        for event in events:
            try:
                event_time = pd.to_datetime(event.event_time)
            except Exception:
                filtered.append(event)
                continue
            if start.date() <= event_time.date() <= end.date():
                filtered.append(event)
        return filtered

    @staticmethod
    def _event_text(event: EventInfo) -> str:
        return " ".join(
            str(part or "") for part in [event.title, event.content, event.location, event.event_type]
        ).lower()

    @staticmethod
    def _event_key(event: EventInfo) -> str:
        return "|".join([event.title or "", event.event_time or "", event.location or ""])

    @staticmethod
    def _classify(text: str, explicit_type: Optional[str]) -> Tuple[str, float]:
        etype = (explicit_type or "").lower()
        combined = f"{etype} {text}"
        for category, keywords in HIGH_IMPACT_KEYWORDS.items():
            if any(keyword in combined for keyword in keywords):
                base = 0.85 if category in {"sports", "concert", "festival"} else 0.65
                return category, base
        if "special event" in combined:
            return "special_event", 0.45
        # disruption 类由 score_event 在关键词检测后覆盖 category；此处返回 unknown
        # 以便保留原始 category_score 逻辑，disruption 保底分在 score_event 中处理
        return "unknown", 0.25

    @staticmethod
    def _extract_value(event: EventInfo, name: str) -> str:
        text = f"{event.content or ''}; {event.location or ''}"
        match = re.search(rf"{re.escape(name)}=([^;|]+)", text)
        return match.group(1).strip() if match else ""

    def _extract_station_id(self, event: EventInfo) -> str:
        return event.station_complex_id or self._extract_value(event, "station_complex_id")

    def _extract_affected_channels(self, event: EventInfo) -> List[str]:
        explicit = event.channel_name or self._extract_value(event, "channel_name")
        if explicit:
            return [explicit]
        station_id = self._extract_station_id(event)
        if station_id:
            return [name for name, meta in self.channel_map.items() if str(meta.get("station_complex_id")) == station_id]
        return []

    @staticmethod
    def _extract_distance(event: EventInfo) -> Optional[float]:
        text = f"{event.content or ''}; {event.location or ''}"
        match = re.search(r"distance(?:_to_station)?=([0-9.]+)m?", text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _looks_like_all_day_admin_event(event: EventInfo, text: str) -> bool:
        if not event.event_time:
            return False
        try:
            parsed = pd.to_datetime(event.event_time)
        except Exception:
            return False
        return parsed.hour == 0 and any(word in text for word in ["closure", "closed", "maintenance", "service disruption"])

# ── LLM 批量重打分入口 ─────────────────────────────────────────────────────────

def llm_rescore_candidates(
    events: List[EventInfo],
    rule_relevance: List[EventRelevance],
    api_key: str,
    base_url: str,
    model_name: str,
    batch_size: int = 10,
    only_high_impact: bool = True,
) -> List[EventRelevance]:
    """用 LLM 对规则筛选出的候选事件做精细重打分，替换 semantic_category 和 relevance_score.

    Args:
        events:          完整事件列表（EventInfo 对象，含 title/content/location）
        rule_relevance:  规则打分结果（EventRelevance 列表）
        api_key:         LLM API Key
        base_url:        LLM API Base URL
        model_name:      LLM 模型名称（如 qwen-plus）
        batch_size:      每次 LLM 调用处理的事件数，默认 10
        only_high_impact:若为 True，只对规则认为是高影响候选的事件做 LLM 重打分
    Returns:
        更新后的 EventRelevance 列表（所有事件，未被 LLM 处理的保留原始规则分）
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.error("langchain_openai 未安装，跳过 LLM 重打分。")
        return rule_relevance

    if not api_key:
        logger.warning("LLM_API_KEY 未设置，跳过 LLM 重打分。")
        return rule_relevance

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0,
        max_retries=2,
        request_timeout=30,
    )

    # 构建 event_key → EventInfo 的快速查找表
    key_to_event: Dict[str, EventInfo] = {}
    for ev in events:
        key = "|".join([ev.title or "", ev.event_time or "", ev.location or ""])
        key_to_event[key] = ev

    # 构建规则打分 key → EventRelevance 的映射（用于最终更新）
    result_map: Dict[str, EventRelevance] = {r.event_key: r for r in rule_relevance}

    # 筛选待重打分列表
    to_rescore = [
        r for r in rule_relevance
        if (not only_high_impact or r.is_high_impact_candidate)
    ]
    logger.info("LLM rescore: %d / %d events selected for rescoring.", len(to_rescore), len(rule_relevance))

    # 按批次调用 LLM
    for batch_start in range(0, len(to_rescore), batch_size):
        batch = to_rescore[batch_start: batch_start + batch_size]
        lines = []
        for i, rel in enumerate(batch, 1):
            ev = key_to_event.get(rel.event_key)
            if ev:
                desc = (ev.content or "")[:120].replace("\n", " ")
                lines.append(
                    f"{i}. Title: \"{rel.title}\" | Date: {rel.event_time} "
                    f"| Location: {getattr(ev, 'location', '')} "
                    f"| Station: {rel.station_complex_id} "
                    f"| Type: {getattr(ev, 'event_type', '')} "
                    f"| Desc: {desc}"
                )
            else:
                lines.append(
                    f"{i}. Title: \"{rel.title}\" | Date: {rel.event_time} "
                    f"| Station: {rel.station_complex_id}"
                )

        prompt = _LLM_RESCORE_PROMPT.format(events_block="\n".join(lines))

        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            # 去除可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scored_list = json.loads(raw)
        except Exception as e:
            logger.warning("LLM rescore batch %d 失败: %s，跳过该批次。", batch_start, e)
            continue

        for item in scored_list:
            idx = item.get("id", 0) - 1
            if not (0 <= idx < len(batch)):
                continue
            rel = batch[idx]
            new_score = float(item.get("relevance_score", rel.relevance_score))
            new_score = max(0.0, min(1.0, new_score))
            new_cat = str(item.get("semantic_category", rel.semantic_category))
            brief = str(item.get("brief_reason", ""))
            updated = rel.model_copy(update={
                "relevance_score": round(new_score, 4),
                "semantic_category": new_cat,
                "is_high_impact_candidate": new_score >= 0.45,
                "evidence": rel.evidence + ([f"llm_reason: {brief[:80]}"] if brief else ["llm_rescored"]),
            })
            result_map[rel.event_key] = updated
            logger.debug(
                "  [LLM] '%s' %.2f→%.2f %s→%s",
                (rel.title or "")[:30], rel.relevance_score, new_score,
                rel.semantic_category, new_cat,
            )

        logger.info(
            "LLM rescore batch %d-%d done.",
            batch_start + 1, min(batch_start + batch_size, len(to_rescore)),
        )

    return list(result_map.values())
