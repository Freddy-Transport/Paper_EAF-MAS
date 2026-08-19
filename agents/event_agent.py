"""事件分析智能体 — 集成 EventFetcher + RAG + LLM."""

import json
import logging
from typing import List, Optional

from agents import config as cfg
from agents.event_fetcher import EventFetcher
from agents.rag_pipeline import RAGPipeline
from agents.schemas import EventImpact, EventInfo

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT_TEMPLATE = """/no_think
You are a transportation demand analyst. Given an event and reference cases, assess the **marginal correction** needed on top of the MOMENT time-series model prediction.

IMPORTANT CONTEXT: The MOMENT model is fine-tuned on NYC subway data and already captures
most recurring event patterns (typically 80-90% of event effects). Your task is to estimate
the RESIDUAL correction the model is likely to miss, NOT the total absolute impact.

## Event Information
- Title: {title}
- Description: {content}
- Time: {event_time}
- Location: {location}
- Type: {event_type}
{channel_constraint}{direction_hint}
## Reference Cases (from historical training data)
{rag_context}

## How to use the reference cases
- References labeled [correction_rag=...] show EMPIRICAL MODEL RESIDUALS from training data.
  These show what correction (actual - model_prediction) was needed for similar past events.
  *** PRIORITIZE these values when setting impact_magnitude. ***
- References labeled [ref=...] show historical absolute impacts (vs no-event baseline).
  These are background context ONLY. Do NOT copy their magnitude values directly.

## Instructions
Output a JSON object with these fields:
- "event_summary": brief one-sentence description of the event (in English)
- "affected_channels": list of affected station channel names or station IDs. If the event text contains channel_name=..., use that exact value.
- "impact_direction": one of "increase", "decrease", or "neutral"
- "impact_magnitude": float in [0.0, 0.15] representing the MARGINAL correction fraction.
  *** If [correction_rag] references exist, use their median_correction as primary guidance. ***
  Typical range: 0.02-0.08. Scale by event size vs the correction_rag reference event.
  Small/local events near this station: 0.01-0.03.
  Medium events (street festival, sports game): 0.03-0.07.
  Large unique events (parade through Times Sq, major marathon finish, major concert): 0.07-0.12.
  Extraordinary once-a-year events (New Year's Eve, major ceremony): up to 0.15.
  Do NOT directly use magnitudes from [ref=...] absolute-impact docs.
- "impact_duration_hours": how many hours does the impact last (typically 3-8 h)
- "reasoning": brief reasoning in English (<=80 chars), why MOMENT would miss this correction

Output ONLY one valid JSON object, no chain-of-thought, no XML tags, no code fences, and no other text.
"""



class EventAnalysisAgent:
    """事件分析智能体：采集事件 -> RAG 检索历史案例 -> LLM 评估影响."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        fetcher: Optional[EventFetcher] = None,
    ):
        self.api_key = api_key or cfg.LLM_API_KEY
        self.base_url = base_url or cfg.LLM_BASE_URL
        self.model_name = model or cfg.LLM_MODEL
        self.fetcher = fetcher or EventFetcher()
        self.rag = rag_pipeline

        self.llm = self._init_llm()

    def _init_llm(self):
        if not self.api_key:
            logger.warning("LLM_API_KEY not set. Event analysis will use fallback mode.")
            return None

        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                temperature=0.1,
                max_tokens=512,
            )
            logger.info("LLM initialized: %s via %s", self.model_name, self.base_url)
            return llm
        except ImportError:
            logger.warning("langchain_openai not installed. Using fallback mode.")
            return None

    def fetch_events(self, source: str, **kwargs) -> List[EventInfo]:
        """从数据源采集事件."""
        return self.fetcher.fetch(source, **kwargs)

    def analyze(
        self,
        events: List[EventInfo],
        direction_hints: Optional[dict] = None,
        _progress: bool = False,
        rag_cutoff_date: Optional[str] = None,
    ) -> List[EventImpact]:
        """分析事件列表，返回各事件的影响评估.

        Args:
            events:          待分析事件列表
            direction_hints: event_key → expected_direction 映射，由
                             EventRelevanceAgent 的 disruption 分类提供，
                             用于在 prompt 中提示 LLM 预期方向（如 "decrease"）
            _progress:       是否在终端打印逐事件进度（由 orchestrator 控制）
            rag_cutoff_date: RAG 时间截止日（YYYY-MM-DD）。只使用此日期之前的历史文档，
                             防止 LLM 参考测试期的实测事件影响数据，消除 RAG 泄漏。
        """
        hints = direction_hints or {}
        impacts = []
        total = len(events)
        for i, event in enumerate(events, 1):
            if _progress:
                title_short = (event.title or "")[:45]
                source_tag = "LLM" if self.llm else "rule"
                print(
                    f"  [{i:>2}/{total}] [{source_tag}] {title_short}",
                    flush=True,
                )
            try:
                event_key = "|".join([event.title or "", event.event_time or "", event.location or ""])
                hint = hints.get(event_key)
                impact = self._analyze_single(event, direction_hint=hint,
                                              rag_cutoff_date=rag_cutoff_date)
                if _progress:
                    dir_icon = {"increase": "↑", "decrease": "↓", "neutral": "→"}.get(
                        impact.impact_direction, "?"
                    )
                    src = getattr(impact, "analysis_source", "llm")
                    print(
                        f"       → {dir_icon} {impact.impact_direction} "
                        f"{impact.impact_magnitude:.0%} "
                        f"{impact.impact_duration_hours:.0f}h "
                        f"[{src}]",
                        flush=True,
                    )
                impacts.append(impact)
            except Exception as e:
                logger.error("Failed to analyze event '%s': %s", event.title[:30], e)
                fb = self._fallback_impact(event)
                if _progress:
                    print(f"       → ERROR fallback: {e}", flush=True)
                impacts.append(fb)
        return impacts

    def _analyze_single(self, event: EventInfo, direction_hint: Optional[str] = None,
                        rag_cutoff_date: Optional[str] = None) -> EventImpact:
        """对单个事件进行分析.

        Args:
            direction_hint:  若不为 None，则在 prompt 中加入方向引导（用于 disruption 类事件）
            rag_cutoff_date: RAG 时间截止日（YYYY-MM-DD），只使用截止日之前的历史文档
        """
        rag_context = ""
        if self.rag is not None:
            rag_context = self.rag.retrieve_context(event, cutoff_date=rag_cutoff_date)

        if not rag_context:
            rag_context = "No historical reference cases available."

        if self.llm is None:
            return self._fallback_impact(event)

        # 若 EventRelevanceAgent 判定为 disruption 类，在 prompt 中明确引导方向
        if direction_hint:
            direction_hint_text = (
                f"## Direction Hint\n"
                f"This event is classified as a service disruption (construction/closure/maintenance). "
                f"Expected impact direction: {direction_hint}. "
                f"Please reflect this in your impact_direction field unless evidence strongly suggests otherwise.\n"
            )
        else:
            direction_hint_text = ""

        # 若数据管线已识别出精确通道，显式告知 LLM 必须使用该通道，防止幻觉
        known_channel = (
            event.channel_name
            or self._extract_affected_channels(event)[:1]
        )
        if isinstance(known_channel, list):
            known_channel = known_channel[0] if known_channel else None
        if known_channel:
            channel_constraint = (
                f"## Channel Constraint\n"
                f"This event has been pre-matched to station channel: **{known_channel}**. "
                f"You MUST use exactly this value in the affected_channels field. "
                f"Do NOT output other channel names.\n"
            )
        else:
            channel_constraint = ""

        content_for_llm = event.enriched_description or event.content or event.title
        tier_line = (
            f"## Event Tier\nThis event is classified as Tier {event.impact_tier} "
            f"({event.event_category or 'unknown'}). "
            f"Only output non-neutral impact if Tier A/B warrants a marginal correction.\n"
        )

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            title=event.title,
            content=f"{tier_line}\n{content_for_llm}",
            event_time=event.event_time,
            location=event.location or "Unknown",
            event_type=event.event_type or "Unknown",
            direction_hint=direction_hint_text,
            channel_constraint=channel_constraint,
            rag_context=rag_context,
        )

        response = self.llm.invoke(prompt)
        impact = self._parse_response(response.content, event)

        # 后处理：若 LLM 返回空 affected_channels，用已知通道填充
        if not impact.affected_channels and known_channel:
            impact = impact.model_copy(
                update={"affected_channels": [known_channel]}
            )

        # 记录推理链（截断到合理长度，保留论文实验可视化用）
        rag_preview = rag_context[:300].replace("\n", " ") if rag_context else None
        impact = impact.model_copy(update={
            "rag_context_preview": rag_preview,
            "llm_reasoning_brief": impact.reasoning[:120] if impact.reasoning else None,
        })

        return impact

    def _parse_response(self, text: str, event: EventInfo) -> EventImpact:
        """解析 LLM 返回的 JSON 文本."""
        text = text.strip()
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= start:
                text = text[start : end + 1]

        try:
            data = json.loads(text)
            # 将事件时间注入 EventImpact，用于时间感知融合
            data.setdefault("event_time", event.event_time)
            data.setdefault("analysis_source", "llm")
            data.setdefault("impact_tier", event.impact_tier if event.impact_tier in ("A", "B") else "B")
            if event.impact_tier in ("C", "D"):
                data["impact_direction"] = "neutral"
                data["impact_magnitude"] = 0.0
            return EventImpact(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to parse LLM response: %s. Raw: %s", e, text[:200])
            return self._fallback_impact(event)

    @staticmethod
    def _extract_affected_channels(event: EventInfo) -> List[str]:
        import re

        if event.channel_name:
            return [event.channel_name]
        text = f"{event.content or ''}; {event.location or ''}"
        match = re.search(r"channel_name=([^;|]+)", text)
        if match:
            return [match.group(1).strip()]
        if event.station_complex_id:
            return [event.station_complex_id]
        match = re.search(r"station_complex_id=([^;|]+)", text)
        if match:
            return [match.group(1).strip()]
        return []

    @staticmethod
    def _rule_based_impact(event: EventInfo) -> EventImpact:
        """无 LLM 时用规则推断事件影响（基于事件类型关键词）."""
        if event.impact_tier in ("C", "D") or not event.should_adjust_forecast:
            return EventImpact(
                event_summary=event.title,
                affected_channels=EventAnalysisAgent._extract_affected_channels(event),
                impact_direction="neutral",
                impact_magnitude=0.0,
                impact_duration_hours=0,
                reasoning="Rule-based: low-tier event, no forecast adjustment.",
                event_time=event.event_time,
                impact_tier=event.impact_tier,
                analysis_source="rule_fallback",
            )

        title_lower = event.title.lower()
        etype = (event.event_type or "").lower()
        affected = EventAnalysisAgent._extract_affected_channels(event)
        tier = event.impact_tier if event.impact_tier in ("A", "B") else "B"
        cap_mag = 0.08 if tier == "A" else 0.05

        # 演唱会 / 音乐表演
        if any(k in title_lower for k in ["concert", "tour", "music", "show", "live"]):
            return EventImpact(
                event_summary=event.title,
                affected_channels=affected,
                impact_direction="increase",
                impact_magnitude=min(0.07, cap_mag),
                impact_duration_hours=6,
                reasoning="Rule-based: concert marginal correction (tier-capped).",
                event_time=event.event_time,
                impact_tier=tier,
                analysis_source="rule_fallback",
            )
        # NBA / 体育赛事
        if any(k in title_lower for k in ["nets", "nba", "knicks", "liberty", "wnba",
                                           "ufc", "wwe", "boxing", "wrestling"]):
            return EventImpact(
                event_summary=event.title,
                affected_channels=affected,
                impact_direction="increase",
                impact_magnitude=min(0.08, cap_mag),
                impact_duration_hours=5,
                reasoning="Rule-based: sports marginal correction (tier-capped).",
                event_time=event.event_time,
                impact_tier=tier,
                analysis_source="rule_fallback",
            )
        # 颁奖 / 典礼 / 毕业
        if any(k in title_lower for k in ["award", "ceremony", "graduation", "induction",
                                           "hall of fame", "showcase"]):
            return EventImpact(
                event_summary=event.title,
                affected_channels=affected,
                impact_direction="increase",
                impact_magnitude=0.15,
                impact_duration_hours=4,
                reasoning="Rule-based: ceremony events typically increase ridership by ~15%.",
                event_time=event.event_time,
                analysis_source="rule_fallback",
            )
        # 默认：中性
        return EventImpact(
            event_summary=event.title,
            affected_channels=[],
            impact_direction="neutral",
            impact_magnitude=0.0,
            impact_duration_hours=0,
            reasoning="Rule-based: event type not recognized; neutral impact assumed.",
            event_time=event.event_time,
            impact_tier=tier,
            analysis_source="rule_fallback",
        )

    def _fallback_impact(self, event: EventInfo) -> EventImpact:
        """当 LLM 不可用或解析失败时，使用规则推断（有 API Key 则提示失败）."""
        if self.api_key:
            logger.warning("LLM parse failed for '%s', falling back to rule-based.", event.title[:40])
        return self._rule_based_impact(event)
