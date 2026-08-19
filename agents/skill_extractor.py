"""技能提取器 — 从每次推理的经验轨迹中提炼并更新 AutoSkill 技能库。

工作原理：
  推理结束后，若 FinalPrediction 包含 ground_truth，则对每个已考量的事件：
  1. 计算该事件窗口内各通道的"实际变化量"与"LLM 估计变化量"之比 → CALIBRATION 技能
  2. 若误差较大（相对 RMSE > 阈值）→ ROUTING 技能（下次额外 RAG 检索）
  3. （可选）分析实际误差消失的时间步 → DURATION 技能修正建议

技能 ID 命名规则：
  - CALIBRATION: "calib_{event_type}"
  - ROUTING:     "routing_{event_type}_extra_rag"
  - DURATION:    "duration_{event_type}"
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents import config as cfg
from agents.schemas import EventImpact, FinalPrediction, Skill, SkillType
from agents.skill_library import SkillLibrary

logger = logging.getLogger(__name__)

# 事件类型关键词映射（中英文均支持）
_EVENT_TYPE_MAP: List[Tuple[str, List[str]]] = [
    ("concert",   ["concert", "music", "show", "live", "tour", "演唱会", "音乐会", "音乐", "演出"]),
    ("sports",    ["nba", "wnba", "knicks", "nets", "mlb", "nfl", "ufc", "wwe", "boxing",
                   "sports", "game", "match", "球赛", "赛事", "比赛", "体育"]),
    ("ceremony",  ["ceremony", "graduation", "award", "induction", "showcase",
                   "典礼", "毕业", "颁奖", "仪式"]),
    ("festival",  ["festival", "parade", "fair", "carnival", "节日", "游行", "庙会"]),
    ("holiday",   ["holiday", "national day", "new year", "假日", "国庆", "元旦", "春节"]),
]


class SkillExtractor:
    """从 FinalPrediction 经验轨迹中自动提炼并更新技能库."""

    # 量化校准比率的合理范围（超出范围视为离群值，不学习）
    MIN_CALIB_RATIO: float = 0.1
    MAX_CALIB_RATIO: float = 5.0
    # EMA 权重：旧值占比（越大越保守，新经验影响越小）
    EMA_ALPHA: float = 0.3
    # 背景误差门控：仅当事件窗口相对误差超过背景误差 N 倍时才更新 calibration
    # 设为 2.0 意味着"事件误差至少是背景噪声的 2 倍"才被认为是事件信号
    CALIB_SNR_THRESHOLD: float = 2.0

    def __init__(
        self,
        library: SkillLibrary,
        routing_error_threshold: float = None,
        channel_map_path: Optional[str] = None,
    ):
        self.library = library
        self.routing_error_threshold = (
            routing_error_threshold
            if routing_error_threshold is not None
            else cfg.SKILL_ROUTING_ERROR_THRESHOLD
        )
        # channel_map: channel_name → {rank, borough, station_complex_id, ...}
        self._channel_meta: Dict[str, dict] = {}
        if channel_map_path:
            self._channel_meta = self._load_channel_map(channel_map_path)
            logger.info(
                "SkillExtractor: loaded channel map with %d entries from %s",
                len(self._channel_meta), channel_map_path,
            )

    @staticmethod
    def _load_channel_map(path: str) -> Dict[str, dict]:
        """从 JSON 文件加载 channel_name → metadata 映射."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            channels = payload.get("channels", payload if isinstance(payload, list) else [])
            return {str(row["channel_name"]): row for row in channels if row.get("channel_name")}
        except Exception as e:
            logger.warning("Failed to load channel map from %s: %s", path, e)
            return {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def extract(self, result: FinalPrediction, trace_id: str) -> List[Skill]:
        """从一次完整推理结果中提炼技能并写入库，返回本次新增/更新的技能列表。

        要求 result.ground_truth 和 result.events_considered 非空，否则直接返回空列表。
        仅当整体 adjusted MAE 优于 raw 时才学习（P5：避免错误校准污染技能库）。
        """
        if not result.ground_truth or not result.events_considered:
            logger.debug("SkillExtractor: no ground_truth or events, skipping.")
            return []

        if not self._overall_forecast_improved(result):
            logger.info(
                "SkillExtractor [trace=%s]: adjusted not better than raw, skip skill learn.",
                trace_id,
            )
            return []

        extracted: List[Skill] = []
        for impact in result.events_considered:
            skills = self._extract_from_impact(impact, result, trace_id)
            extracted.extend(skills)

        logger.info(
            "SkillExtractor [trace=%s]: extracted/updated %d skills from %d events.",
            trace_id, len(extracted), len(result.events_considered),
        )
        return extracted

    @staticmethod
    def _overall_forecast_improved(result: FinalPrediction) -> bool:
        """全通道平均 MAE：adjusted 是否优于 raw."""
        if not result.ground_truth or not result.raw_forecast:
            return False
        n_ch = min(len(result.raw_forecast), len(result.ground_truth), len(result.adjusted_forecast))
        if n_ch == 0:
            return False
        raw_maes, adj_maes = [], []
        for ch_idx in range(n_ch):
            raw = result.raw_forecast[ch_idx]
            adj = result.adjusted_forecast[ch_idx]
            gt = result.ground_truth[ch_idx]
            n = min(len(raw), len(adj), len(gt))
            if n == 0:
                continue
            raw_maes.append(sum(abs(raw[s] - gt[s]) for s in range(n)) / n)
            adj_maes.append(sum(abs(adj[s] - gt[s]) for s in range(n)) / n)
        if not raw_maes:
            return False
        return sum(adj_maes) / len(adj_maes) < sum(raw_maes) / len(raw_maes)

    # ------------------------------------------------------------------
    # 核心提取逻辑
    # ------------------------------------------------------------------

    def _extract_from_impact(
        self,
        impact: EventImpact,
        result: FinalPrediction,
        trace_id: str,
    ) -> List[Skill]:
        """针对单个事件提取技能。"""
        # 找到该事件影响的时间步（复用 PredictionFusion 的静态方法）
        from agents.prediction_fusion import PredictionFusion  # 延迟导入避免循环

        n_steps = len(result.raw_forecast[0]) if result.raw_forecast else 0
        affected_steps = PredictionFusion._find_affected_steps(
            impact, result.forecast_timestamps, n_steps
        )
        if not affected_steps:
            return []

        event_type = self.classify_event(impact.event_summary)
        channel_names = result.channel_names or []
        skills: List[Skill] = []

        # 从受影响通道推断 borough 和 rank_group（用于细粒度 skill trigger）
        borough, rank_group = self._infer_channel_context(
            impact.affected_channels, channel_names
        )

        # 预计算非事件窗口的背景相对误差（用于 SNR 门控）
        affected_set = set(affected_steps)
        non_affected_steps = [s for s in range(n_steps) if s not in affected_set]
        background_rel_errors: List[float] = []
        if non_affected_steps:
            for ch_idx, ch_name in enumerate(channel_names):
                if not self._channel_affected(ch_name, impact.affected_channels):
                    continue
                if ch_idx >= len(result.raw_forecast) or ch_idx >= len(result.ground_truth):
                    continue
                gt_ch = result.ground_truth[ch_idx]
                raw_ch = result.raw_forecast[ch_idx]
                safe_steps = [
                    s for s in non_affected_steps
                    if s < len(gt_ch) and s < len(raw_ch)
                ]
                if not safe_steps:
                    continue
                raw_bg = [raw_ch[s] for s in safe_steps]
                gt_bg = [gt_ch[s] for s in safe_steps]
                mean_raw_bg = sum(raw_bg) / len(raw_bg)
                if abs(mean_raw_bg) < 1e-6:
                    continue
                mse_bg = sum((g - r) ** 2 for g, r in zip(gt_bg, raw_bg)) / len(raw_bg)
                background_rel_errors.append(mse_bg ** 0.5 / (abs(mean_raw_bg) + 1e-6))
        baseline_rel_error = (
            sum(background_rel_errors) / len(background_rel_errors)
            if background_rel_errors else 0.0
        )

        # 收集各通道的校准比率
        ratios: List[float] = []
        max_rel_rmse: float = 0.0

        for ch_idx, ch_name in enumerate(channel_names):
            if not self._channel_affected(ch_name, impact.affected_channels):
                continue
            if ch_idx >= len(result.raw_forecast) or ch_idx >= len(result.ground_truth):
                continue

            raw_steps = [result.raw_forecast[ch_idx][s] for s in affected_steps]
            gt_steps  = [result.ground_truth[ch_idx][s]  for s in affected_steps]

            mean_raw = sum(raw_steps) / len(raw_steps)
            mean_gt  = sum(gt_steps)  / len(gt_steps)

            if abs(mean_raw) < 1e-6:
                continue

            # 实际变化量（相对于数值基线预测）
            actual_change = (mean_gt - mean_raw) / mean_raw

            # LLM 估计的变化量
            direction_sign = (
                1.0  if impact.impact_direction == "increase" else
                -1.0 if impact.impact_direction == "decrease" else
                0.0
            )
            estimated_change = direction_sign * impact.impact_magnitude

            if abs(estimated_change) > 1e-4:
                ratio = actual_change / estimated_change
                if self.MIN_CALIB_RATIO <= ratio <= self.MAX_CALIB_RATIO:
                    ratios.append(ratio)

            # 相对 RMSE（用于判断是否触发路由技能）
            mse = sum((g - r) ** 2 for g, r in zip(gt_steps, raw_steps)) / len(raw_steps)
            rel_rmse = mse ** 0.5 / (abs(mean_raw) + 1e-6)
            max_rel_rmse = max(max_rel_rmse, rel_rmse)

        # 生成 CALIBRATION 技能（各通道比率取平均）
        # SNR 门控：事件窗口误差须显著高于背景误差，否则可能只是普通时序噪声
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            # 检查信噪比：max_rel_rmse 是事件窗口最大相对 RMSE，baseline_rel_error 是背景误差
            snr_ok = (
                baseline_rel_error < 1e-6  # 无背景数据时跳过门控，仍允许学习
                or max_rel_rmse >= self.CALIB_SNR_THRESHOLD * baseline_rel_error
            )
            if not snr_ok:
                logger.info(
                    "Calibration skipped for '%s': event error (%.3f) < %.1f× baseline (%.3f); "
                    "likely normal forecast noise, not event signal.",
                    impact.event_summary[:40],
                    max_rel_rmse,
                    self.CALIB_SNR_THRESHOLD,
                    baseline_rel_error,
                )
            else:
                skill = self._make_or_update_calibration_skill(
                    event_type, avg_ratio, trace_id, borough=borough, rank_group=rank_group
                )
                skills.append(skill)

                # 同时更新已有技能的置信度（ratio 接近 1.0 表示 LLM 已经很准，不需要校准太多）
                self.library.update_confidence(
                    skill.skill_id,
                    supported=(0.8 <= avg_ratio <= 1.2),  # 误差 <20% 视为"支持"
                )

        # 生成 ROUTING 技能（相对误差超过阈值时）
        if max_rel_rmse > self.routing_error_threshold:
            skill = self._make_or_update_routing_skill(
                event_type, trace_id, borough=borough, rank_group=rank_group
            )
            skills.append(skill)

        # 生成 DURATION 技能（通过误差分布推断实际影响时长）
        duration_skill = self._try_extract_duration_skill(
            impact, result, affected_steps, event_type, trace_id
        )
        if duration_skill:
            skills.append(duration_skill)

        return skills

    # ------------------------------------------------------------------
    # 技能构造 / 更新
    # ------------------------------------------------------------------

    def _make_or_update_calibration_skill(
        self,
        event_type: str,
        new_ratio: float,
        trace_id: str,
        borough: Optional[str] = None,
        rank_group: Optional[str] = None,
    ) -> Skill:
        """创建或更新 CALIBRATION 技能.

        若提供 borough / rank_group，skill_id 和 trigger 会更细粒度，
        减少跨地区、跨客流量级的负迁移。
        """
        suffix = self._build_skill_suffix(borough, rank_group)
        skill_id = f"calib_{event_type}{suffix}"
        trigger: dict = {"event_type": event_type}
        if borough:
            trigger["borough"] = borough
        if rank_group:
            trigger["rank_group"] = rank_group

        existing = self.library.get(skill_id)
        if existing:
            old_ratio = existing.action.get("calibration_ratio", 1.0)
            # EMA 更新：保守地混合新旧比率
            updated_ratio = round(
                (1 - self.EMA_ALPHA) * old_ratio + self.EMA_ALPHA * new_ratio, 4
            )
            existing.action["calibration_ratio"] = updated_ratio
            if trace_id not in existing.source_trace_ids:
                existing.source_trace_ids.append(trace_id)
            self.library.save(existing)
            logger.info(
                "CALIBRATION skill '%s' updated: ratio %.3f → %.3f",
                skill_id, old_ratio, updated_ratio,
            )
            return existing

        skill = Skill(
            skill_id=skill_id,
            skill_type=SkillType.CALIBRATION,
            trigger=trigger,
            action={"calibration_ratio": round(new_ratio, 4)},
            confidence=0.4,
            source_trace_ids=[trace_id],
        )
        self.library.save(skill)
        logger.info(
            "CALIBRATION skill '%s' created: ratio=%.3f (borough=%s, rank_group=%s)",
            skill_id, new_ratio, borough or "any", rank_group or "any",
        )
        return skill

    def _make_or_update_routing_skill(
        self,
        event_type: str,
        trace_id: str,
        borough: Optional[str] = None,
        rank_group: Optional[str] = None,
    ) -> Skill:
        suffix = self._build_skill_suffix(borough, rank_group)
        skill_id = f"routing_{event_type}_extra_rag{suffix}"
        trigger: dict = {"event_type": event_type}
        if borough:
            trigger["borough"] = borough
        if rank_group:
            trigger["rank_group"] = rank_group

        existing = self.library.get(skill_id)
        if existing:
            if trace_id not in existing.source_trace_ids:
                existing.source_trace_ids.append(trace_id)
            self.library.update_confidence(skill_id, supported=True)
            logger.info("ROUTING skill '%s' reinforced.", skill_id)
            return existing

        skill = Skill(
            skill_id=skill_id,
            skill_type=SkillType.ROUTING,
            trigger=trigger,
            action={"add_step": "extra_rag_retrieve"},
            confidence=0.4,
            source_trace_ids=[trace_id],
        )
        self.library.save(skill)
        logger.info("ROUTING skill '%s' created (extra_rag triggered by high error).", skill_id)
        return skill

    def _try_extract_duration_skill(
        self,
        impact: EventImpact,
        result: FinalPrediction,
        affected_steps: List[int],
        event_type: str,
        trace_id: str,
    ) -> Optional[Skill]:
        """通过分析误差在时间步上的分布来推断实际影响时长。

        若受影响步长后半段的误差已经很小（占前半段 <30%），则
        实际影响时长约为 LLM 估计值的 50%，生成 DURATION 技能。
        若前半段正常而后半段持续有误差（>150%），则实际时长更长，生成 DURATION 技能。
        """
        if not result.ground_truth or len(affected_steps) < 4:
            return None

        channel_names = result.channel_names or []
        first_half = affected_steps[: len(affected_steps) // 2]
        second_half = affected_steps[len(affected_steps) // 2 :]

        errors_first, errors_second = [], []
        for ch_idx, ch_name in enumerate(channel_names):
            # 仅统计真正受事件影响的通道，与 CALIBRATION 技能逻辑保持一致
            if not self._channel_affected(ch_name, impact.affected_channels):
                continue
            if ch_idx >= len(result.raw_forecast) or ch_idx >= len(result.ground_truth):
                continue
            for s in first_half:
                errors_first.append(
                    abs(result.ground_truth[ch_idx][s] - result.raw_forecast[ch_idx][s])
                )
            for s in second_half:
                errors_second.append(
                    abs(result.ground_truth[ch_idx][s] - result.raw_forecast[ch_idx][s])
                )

        if not errors_first or not errors_second:
            return None

        mean_first  = sum(errors_first)  / len(errors_first)
        mean_second = sum(errors_second) / len(errors_second)

        if mean_first < 1e-6:
            return None

        ratio = mean_second / mean_first
        estimated_h = float(impact.impact_duration_hours)
        skill_id = f"duration_{event_type}"

        if ratio < 0.30:
            # 后半段误差远小于前半段：实际时长更短
            actual_h = round(estimated_h * 0.6, 1)
        elif ratio > 1.50:
            # 后半段误差仍很大：实际时长可能更长
            actual_h = round(estimated_h * 1.4, 1)
        else:
            return None  # 误差分布均匀，无需修正时长

        existing = self.library.get(skill_id)
        if existing:
            old_h = existing.action.get("duration_override", estimated_h)
            updated_h = round(0.7 * old_h + 0.3 * actual_h, 1)
            existing.action["duration_override"] = updated_h
            if trace_id not in existing.source_trace_ids:
                existing.source_trace_ids.append(trace_id)
            self.library.save(existing)
            logger.info("DURATION skill '%s' updated: %.1fh → %.1fh", skill_id, old_h, updated_h)
            return existing

        skill = Skill(
            skill_id=skill_id,
            skill_type=SkillType.DURATION,
            trigger={"event_type": event_type},
            action={"duration_override": actual_h},
            confidence=0.35,
            source_trace_ids=[trace_id],
        )
        self.library.save(skill)
        logger.info(
            "DURATION skill '%s' created: override %.1fh (LLM estimated %.1fh)",
            skill_id, actual_h, estimated_h,
        )
        return skill

    # ------------------------------------------------------------------
    # 工具方法（静态，供 orchestrator 的 skill_calibrate 步骤调用）
    # ------------------------------------------------------------------

    @staticmethod
    def classify_event(event_summary: str) -> str:
        """将事件描述归类为预定义的事件类型关键词."""
        text = event_summary.lower()
        for event_type, keywords in _EVENT_TYPE_MAP:
            if any(kw in text for kw in keywords):
                return event_type
        return "general"

    @staticmethod
    def _channel_affected(channel_name: str, affected_channels: List[str]) -> bool:
        if not affected_channels:
            return True
        ch = channel_name.lower()
        return any(ac.lower() in ch or ch in ac.lower() for ac in affected_channels)

    def _infer_channel_context(
        self,
        affected_channels: List[str],
        all_channel_names: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """从受影响通道推断主要 borough 和 rank_group.

        borough: 若所有受影响通道都在同一 borough，则返回该 borough；否则返回 None
        rank_group: 按通道在 Top128 中的排名分为 top32 / top64 / top128
        """
        if not self._channel_meta:
            return None, None

        boroughs: List[str] = []
        ranks: List[int] = []

        # 若无明确 affected_channels，用所有通道中匹配的
        sources = affected_channels if affected_channels else all_channel_names
        for ch_name in sources:
            meta = self._channel_meta.get(ch_name)
            if meta is None:
                # 尝试子串匹配（channel_name 可能经过截断）
                for key, val in self._channel_meta.items():
                    if ch_name.lower() in key.lower() or key.lower() in ch_name.lower():
                        meta = val
                        break
            if meta:
                b = meta.get("borough")
                r = meta.get("rank")
                if b:
                    boroughs.append(str(b).lower().replace(" ", "_"))
                if r is not None:
                    try:
                        ranks.append(int(r))
                    except (ValueError, TypeError):
                        pass

        borough = boroughs[0] if boroughs and len(set(boroughs)) == 1 else None

        rank_group: Optional[str] = None
        if ranks:
            avg_rank = sum(ranks) / len(ranks)
            if avg_rank <= 32:
                rank_group = "top32"
            elif avg_rank <= 64:
                rank_group = "top64"
            else:
                rank_group = "top128"

        return borough, rank_group

    @staticmethod
    def _build_skill_suffix(borough: Optional[str], rank_group: Optional[str]) -> str:
        """构建 skill_id 的细粒度后缀，如 '_manhattan_top32'."""
        parts = []
        if borough:
            parts.append(borough)
        if rank_group:
            parts.append(rank_group)
        return ("_" + "_".join(parts)) if parts else ""
