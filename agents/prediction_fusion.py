"""预测融合模块 — 合并数值预测与事件影响评估（时间感知 + Tier cap + 单步最大修正）."""

import copy
import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from agents.event_tier import TIER_CAP
from agents.schemas import EventImpact, FinalPrediction, NumericalPrediction

logger = logging.getLogger(__name__)


class PredictionFusion:
    """将数值预测与事件影响融合。

    - 仅 Tier A/B 事件生效（由上游筛选）
    - 同一 (channel, step) 多事件时取 **最大绝对边际修正**（非连乘）
    - 按事件 Tier 应用不同 cap（A=8%, B=5%）
  """

    DIRECTION_FACTORS = {
        "increase": 1.0,
        "decrease": -1.0,
        "neutral": 0.0,
    }

    MAX_TOTAL_ADJUSTMENT: float = 0.10
    MAGNITUDE_SCALE_FACTOR: float = 0.6

    def __init__(self, fusion_channel_names: Optional[Set[str]] = None):
        self.fusion_channel_names = fusion_channel_names

    def fuse(
        self,
        prediction: NumericalPrediction,
        impacts: List[EventImpact],
    ) -> FinalPrediction:
        adjusted = copy.deepcopy(prediction.forecast)
        timestamps = prediction.forecast_timestamps
        n_steps = len(prediction.forecast[0]) if prediction.forecast else 0

        active_impacts = [
            imp for imp in impacts
            if imp.impact_magnitude > 0.01
            and getattr(imp, "impact_tier", "B") in ("A", "B")
            and imp.impact_direction != "neutral"
        ]

        # (ch_idx, step) -> (delta_fraction, tier)
        step_adjustments: Dict[Tuple[int, int], Tuple[float, str]] = {}
        matched_impacts: List[EventImpact] = []

        for impact in active_impacts:
            affected_steps = self._find_affected_steps(impact, timestamps, n_steps)
            if not affected_steps:
                continue

            if self._station_channels_need_explicit_match(prediction.channel_names, impact):
                continue

            delta = self._compute_delta_fraction(impact)
            tier = getattr(impact, "impact_tier", "B") or "B"

            adjusted_any = False
            for ch_idx, ch_name in enumerate(prediction.channel_names):
                if self.fusion_channel_names and ch_name not in self.fusion_channel_names:
                    continue
                if not self._channel_affected(ch_name, impact.affected_channels):
                    continue
                for step in affected_steps:
                    key = (ch_idx, step)
                    prev = step_adjustments.get(key)
                    if prev is None or abs(delta) > abs(prev[0]):
                        step_adjustments[key] = (delta, tier)
                    adjusted_any = True

            if adjusted_any:
                matched_impacts.append(impact)
                logger.info(
                    "Event '%s' tier=%s delta=%+.2f%% | steps %s",
                    impact.event_summary[:50],
                    tier,
                    delta * 100,
                    affected_steps[:6],
                )

        channels_adjusted: List[str] = []
        for (ch_idx, step), (delta, tier) in step_adjustments.items():
            ch_name = prediction.channel_names[ch_idx]
            raw_val = prediction.forecast[ch_idx][step]
            cap = min(TIER_CAP.get(tier, 0.05), self.MAX_TOTAL_ADJUSTMENT)
            clamped = max(-cap, min(cap, delta))
            adjusted[ch_idx][step] = raw_val * (1.0 + clamped)
            if ch_name not in channels_adjusted:
                channels_adjusted.append(ch_name)

        explanation = self._generate_explanation(
            prediction, matched_impacts, adjusted, channels_adjusted,
        )

        return FinalPrediction(
            raw_forecast=prediction.forecast,
            adjusted_forecast=adjusted,
            events_considered=matched_impacts,
            explanation=explanation,
            channel_names=prediction.channel_names,
            ground_truth=prediction.ground_truth,
            forecast_timestamps=prediction.forecast_timestamps,
            fusion_scope="venue37" if self.fusion_channel_names else "all",
            channels_adjusted=channels_adjusted,
            channels_passthrough=[
                ch for ch in (prediction.channel_names or [])
                if ch not in channels_adjusted
            ],
        )

    def _compute_delta_fraction(self, impact: EventImpact) -> float:
        direction = self.DIRECTION_FACTORS.get(impact.impact_direction, 0.0)
        raw_mag = (
            impact.calibrated_magnitude
            if impact.calibrated_magnitude is not None
            else impact.impact_magnitude
        )
        magnitude = min(abs(raw_mag), 0.40) * self.MAGNITUDE_SCALE_FACTOR
        return direction * magnitude

    @staticmethod
    def _find_affected_steps(
        impact: EventImpact,
        timestamps: Optional[List[str]],
        total_steps: int,
    ) -> List[int]:
        if not impact.event_time:
            return []
        if not timestamps:
            return list(range(total_steps))

        try:
            event_dt = pd.to_datetime(impact.event_time)
        except Exception:
            return []

        event_date = event_dt.date()
        duration_h = max(impact.impact_duration_hours, 1)

        affected = []
        for i, ts_str in enumerate(timestamps):
            try:
                ts_dt = pd.to_datetime(ts_str)
            except Exception:
                continue
            delta_h = (ts_dt - event_dt).total_seconds() / 3600.0
            if -2 <= delta_h <= duration_h:
                affected.append(i)

        if not affected and event_dt.hour == 0 and event_dt.minute == 0:
            for i, ts_str in enumerate(timestamps):
                try:
                    if pd.to_datetime(ts_str).date() == event_date:
                        affected.append(i)
                except Exception:
                    continue

        return affected

    @staticmethod
    def _station_channels_need_explicit_match(channel_names: List[str], impact: EventImpact) -> bool:
        if impact.affected_channels:
            return False
        if len(channel_names) <= 2:
            return False
        station_like = sum(1 for name in channel_names if "__" in name)
        return station_like >= max(2, len(channel_names) // 2)

    @staticmethod
    def _channel_affected(channel_name: str, affected_channels: List[str]) -> bool:
        if not affected_channels:
            return True
        lowered = [ac.lower() for ac in affected_channels]
        if any(ac in {"all", "all_stations", "citywide", "systemwide"} for ac in lowered):
            return True
        ch = channel_name.lower()
        return any(ac in ch or ch in ac for ac in lowered)

    @staticmethod
    def _generate_explanation(
        prediction: NumericalPrediction,
        impacts: List[EventImpact],
        adjusted: List[List[float]],
        channels_adjusted: Optional[List[str]] = None,
    ) -> str:
        if not impacts:
            msg = "No major events (Tier A/B) in forecast window. Numerical prediction used directly."
            if channels_adjusted is not None and len(channels_adjusted) == 0:
                return msg + "\n\nFusion scope: venue37 only; no channels adjusted."
            return msg

        lines = ["## Prediction Report\n"]
        if channels_adjusted is not None:
            passthrough = [
                ch for ch in (prediction.channel_names or [])
                if ch not in channels_adjusted
            ]
            lines.append(f"Channels adjusted (fusion): {', '.join(channels_adjusted) or '(none)'}")
            if passthrough:
                lines.append(
                    f"Channels passthrough (MOMENT only): {len(passthrough)} stations "
                    "(outside venue37 fusion scope)"
                )
        else:
            lines.append(f"Channels: {', '.join(prediction.channel_names)}")
        lines.append(f"Major events applied: {len(impacts)}")

        llm_count = sum(1 for imp in impacts if getattr(imp, "analysis_source", "llm") == "llm")
        rule_count = len(impacts) - llm_count
        lines.append(f"Analysis source: llm={llm_count}, rule_fallback={rule_count}\n")

        for i, impact in enumerate(impacts, 1):
            tier = getattr(impact, "impact_tier", "?")
            lines.append(f"### Event {i} [{tier}]: {impact.event_summary}")
            lines.append(f"- Time: {impact.event_time or 'N/A'}")
            lines.append(f"- Direction: {impact.impact_direction}")
            if impact.calibrated_magnitude is not None:
                lines.append(
                    f"- Magnitude: {impact.impact_magnitude:.2f} "
                    f"→ {impact.calibrated_magnitude:.2f} (skill-calibrated)"
                )
            else:
                lines.append(f"- Magnitude: {impact.impact_magnitude:.2f}")
            lines.append(f"- Duration: {impact.impact_duration_hours}h")
            lines.append(f"- Affected channels: {', '.join(impact.affected_channels) or 'all'}")
            lines.append(f"- Reasoning: {impact.reasoning}")

        return "\n".join(lines)