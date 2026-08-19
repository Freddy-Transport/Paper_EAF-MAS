"""动态协调器 — 技能驱动的自适应管线编排器。

执行流程：
  Phase-1  numerical_predict + fetch_events + filter_events
  Phase-2  SkillDispatcher.plan_event_steps(context) 动态决定子步骤
           （含可选的 extra_rag_retrieve、llm_analyze、skill_calibrate）
  Phase-3  fuse + 可选 skill_learn

与旧版 AgentOrchestrator 的对比：
  - 旧版：固定三步顺序 DAG，路径不可变
  - 新版：步骤列表由 SkillDispatcher 在运行时根据技能库动态组装；
          每次推理后 SkillExtractor 自动提炼/更新技能，实现跨轮自进化
"""

import logging
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from agents import config as cfg
from agents.event_agent import EventAnalysisAgent
from agents.event_relevance_agent import EventRelevanceAgent
from agents.numerical_agent import NumericalPredictionAgent
from agents.prediction_fusion import PredictionFusion
from agents.rag_pipeline import RAGPipeline
from agents.schemas import (
    EventImpact,
    FinalPrediction,
    PipelineContext,
    PredictionRequest,
    SkillType,
)
from agents.skill_dispatcher import SkillDispatcher
from agents.skill_extractor import SkillExtractor
from agents.skill_library import SkillLibrary

logger = logging.getLogger(__name__)

# 步骤执行状态字典的类型别名
PipelineState = Dict


class DynamicOrchestrator:
    """技能驱动的动态多智能体协调器。

    step_registry 将步骤名映射到对应的执行函数；SkillDispatcher 在运行时
    根据上下文和技能库决定执行哪些步骤及顺序。
    """

    def __init__(
        self,
        numerical_agent: Optional[NumericalPredictionAgent] = None,
        event_agent: Optional[EventAnalysisAgent] = None,
        fusion: Optional[PredictionFusion] = None,
        skill_dispatcher: Optional[SkillDispatcher] = None,
        skill_extractor: Optional[SkillExtractor] = None,
        event_relevance_agent: Optional[EventRelevanceAgent] = None,
        event_adapter_dir: str = None,
        prediction_mode: str = None,
    ):
        self.numerical_agent = numerical_agent
        self.event_agent = event_agent
        self.event_relevance_agent = event_relevance_agent or EventRelevanceAgent()
        self.fusion = fusion or PredictionFusion()
        self.skill_dispatcher = skill_dispatcher
        self.skill_extractor = skill_extractor
        self.event_adapter_dir = event_adapter_dir
        self.prediction_mode = prediction_mode or cfg.PREDICTION_MODE
        self._freeze_skills: bool = False   # 由 from_config 覆盖

        self.step_registry: Dict[str, Callable] = {
            "numerical_predict":  self._step_numerical_predict,
            "fetch_events":       self._step_fetch_events,
            "filter_events":      self._step_filter_events,
            "event_relevance_filter": self._step_event_relevance_filter,
            "extra_rag_retrieve": self._step_extra_rag_retrieve,
            "llm_analyze":        self._step_llm_analyze,
            "skill_calibrate":    self._step_skill_calibrate,
            "event_adapter":      self._step_event_adapter,
            "fuse":               self._step_fuse,
            "skill_learn":        self._step_skill_learn,
        }

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_path: str = None,
        device: str = None,
        channel_names: Optional[List[str]] = None,
        n_channels: int = None,
        forecast_horizon: int = None,
        llm_api_key: str = None,
        knowledge_base_dir: str = None,
        skill_library_path: str = None,
        channel_map_path: str = None,
        event_relevance_threshold: float = 0.45,
        max_llm_events_per_window: int = 80,
        fusion_channel_names: Optional[List[str]] = None,
        event_station_whitelist: Optional[List[str]] = None,
        # LP 回退参数（model_path 下无权重时生效）
        lp_data_path: str = None,
        lp_model_path: str = None,
        lp_max_epoch: int = None,
        lp_lr: float = None,
        lp_batch_size: int = None,
        # 数据泄漏防护开关（默认 False = 训练模式；评估/测试时设为 True）
        freeze_skills: bool = False,
        prediction_mode: str = None,
        event_adapter_path: str = None,
    ) -> "DynamicOrchestrator":
        """从配置参数初始化所有组件（含技能库、调度器、提取器）."""

        mode = prediction_mode or cfg.PREDICTION_MODE
        event_adapter_dir = None
        if mode in ("event_adapter_frozen_moment", "event_adapter_peft_moment"):
            adapter_root = Path(event_adapter_path or cfg.EVENT_ADAPTER_PATH)
            if (adapter_root / "event_adapter.pt").is_file():
                event_adapter_dir = str(adapter_root)
            elif adapter_root.is_file():
                event_adapter_dir = str(adapter_root.parent)
            else:
                logger.warning("Event adapter not found: %s", adapter_root)

        num_agent = NumericalPredictionAgent(
            model_path=model_path or cfg.MODEL_PATH,
            device=device or cfg.DEVICE,
            forecast_horizon=forecast_horizon or cfg.FORECAST_HORIZON,
            n_channels=n_channels or cfg.N_CHANNELS,
            channel_names=channel_names,
            lp_data_path=lp_data_path,
            lp_model_path=lp_model_path,
            lp_max_epoch=lp_max_epoch,
            lp_lr=lp_lr,
            lp_batch_size=lp_batch_size,
        )

        rag = RAGPipeline(knowledge_base_dir=knowledge_base_dir or cfg.KNOWLEDGE_BASE_DIR)

        event_agent = EventAnalysisAgent(
            api_key=llm_api_key or cfg.LLM_API_KEY,
            rag_pipeline=rag,
        )

        library = SkillLibrary(skill_library_path or cfg.SKILL_LIBRARY_PATH)
        dispatcher = SkillDispatcher(library)
        extractor = SkillExtractor(library, channel_map_path=channel_map_path or cfg.CHANNEL_MAP_PATH)

        relevance_agent = EventRelevanceAgent(
            channel_map_path=channel_map_path,
            relevance_threshold=event_relevance_threshold,
            max_events=max_llm_events_per_window,
        )

        fusion_channels = set(fusion_channel_names) if fusion_channel_names else None
        inst = cls(
            numerical_agent=num_agent,
            event_agent=event_agent,
            fusion=PredictionFusion(fusion_channel_names=fusion_channels),
            skill_dispatcher=dispatcher,
            skill_extractor=extractor,
            event_relevance_agent=relevance_agent,
            event_adapter_dir=event_adapter_dir,
            prediction_mode=mode,
        )
        inst._event_station_whitelist = set(event_station_whitelist or [])
        inst._fusion_channel_names = list(fusion_channel_names or [])
        # 记录训练/测试模式开关（freeze_skills=True → 测试评估，不更新技能库）
        inst._freeze_skills = freeze_skills
        return inst

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        data_path: str,
        event_source: str = None,
        forecast_horizon: int = None,
        n_channels: int = None,
        channel_map_path: str = None,
        event_relevance_threshold: float = 0.45,
        max_llm_events_per_window: int = 80,
        event_station_whitelist: Optional[List[str]] = None,
        fusion_channel_names: Optional[List[str]] = None,
        # 指定日期预测（论文实验接口）：若设置，以该日期为预测窗口起始
        target_date: Optional[str] = None,
        # Skill 学习开关：True=运行后更新 Skill 库；False=纯查询/评估，不写 Skill
        update_skills: bool = True,
    ) -> FinalPrediction:
        """执行完整预测流程（动态管线）.

        Args:
            data_path:   CSV 数据路径。
            event_source: 事件 JSON 路径（可选）。
            target_date: 指定预测起始日（YYYY-MM-DD）。若为 None，使用数据集最后一个测试窗口。
            update_skills: 是否在预测后根据 ground_truth 更新 Skill 库。
                           评估/查询时建议设为 False，自适应学习时设为 True。
        """
        logger.info("=" * 60)
        logger.info("DynamicOrchestrator: starting adaptive pipeline")
        logger.info("=" * 60)

        context = PipelineContext(
            data_path=data_path,
            event_source=event_source,
            forecast_horizon=forecast_horizon or cfg.FORECAST_HORIZON,
            n_channels=n_channels or cfg.N_CHANNELS,
            has_events=bool(event_source),
            channel_map_path=channel_map_path,
            event_relevance_threshold=event_relevance_threshold,
            max_llm_events_per_window=max_llm_events_per_window,
            update_skills=update_skills,
            event_station_whitelist=event_station_whitelist or list(getattr(self, "_event_station_whitelist", [])),
            fusion_channel_names=fusion_channel_names or list(getattr(self, "_fusion_channel_names", [])),
        )
        # 若调用方或 from_config 设置了 freeze_skills，也确保 update_skills=False
        if getattr(self, "_freeze_skills", False):
            context.update_skills = False

        # 指定日期：写入 context 供 _step_numerical_predict 读取
        context._target_date = target_date  # 动态属性，不在 schema 内
        context._prediction_mode = getattr(self, "prediction_mode", cfg.PREDICTION_MODE)

        state: PipelineState = {}

        # ── Phase 1: 固定前置步骤 ──────────────────────────────────────
        for step in ["numerical_predict"]:
            state = self._run_step(step, state, context)

        mode = context._prediction_mode
        if mode == "numerical_only":
            state = self._step_numerical_only(state, context)
        elif mode in ("event_adapter_frozen_moment", "event_adapter_peft_moment"):
            state = self._run_step("event_adapter", state, context)
        else:
            # legacy_fusion: LLM + RAG + PredictionFusion
            if event_source:
                for step in ["fetch_events", "filter_events", "event_relevance_filter"]:
                    state = self._run_step(step, state, context)

                if self.skill_dispatcher:
                    event_steps = self.skill_dispatcher.plan_event_steps(context)
                else:
                    event_steps = ["llm_analyze", "skill_calibrate"]

                logger.info("[Phase 2] Dynamic event steps: %s", " → ".join(event_steps) or "(none)")
                for step in event_steps:
                    state = self._run_step(step, state, context)

            state = self._run_step("fuse", state, context)

        if self.skill_dispatcher:
            postprocess_steps = self.skill_dispatcher.plan_postprocess(context)
        else:
            postprocess_steps = ["skill_learn"] if context.has_ground_truth else []

        for step in postprocess_steps:
            state = self._run_step(step, state, context)

        result: FinalPrediction = state.get("result")
        logger.info(
            "Pipeline complete. Events considered: %d | Skills in library: %d",
            len(result.events_considered) if result else 0,
            len(self.skill_dispatcher.library) if self.skill_dispatcher else 0,
        )
        logger.info("=" * 60)
        return result

    def _run_step(self, step_name: str, state: PipelineState, context: PipelineContext) -> PipelineState:
        import time as _time
        fn = self.step_registry.get(step_name)
        if fn is None:
            logger.warning("Unknown step '%s', skipping.", step_name)
            return state
        _STEP_LABELS = {
            "numerical_predict":      "数值预测（MOMENT-LP 推理）",
            "fetch_events":           "事件加载",
            "filter_events":          "时间窗口过滤",
            "event_relevance_filter": "语义相关性筛选",
            "extra_rag_retrieve":     "额外 RAG 检索（技能触发）",
            "llm_analyze":            "LLM 事件影响分析",
            "skill_calibrate":        "Skill 校准（幅度/时长修正）",
            "fuse":                   "预测融合",
            "skill_learn":            "Skill 提取与更新",
        }
        label = _STEP_LABELS.get(step_name, step_name)
        print(f"\n▶ [{step_name}] {label} ...", flush=True)
        logger.info("[Step] %s", step_name)
        _t0 = _time.time()
        result = fn(state, context)
        print(f"  ✓ 完成 ({_time.time() - _t0:.1f}s)", flush=True)
        return result

    # ------------------------------------------------------------------
    # 步骤实现
    # ------------------------------------------------------------------

    def _step_numerical_predict(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        target_date = getattr(context, "_target_date", None)
        request = PredictionRequest(
            history_data_path=context.data_path,
            forecast_horizon=context.forecast_horizon,
            n_channels=context.n_channels,
            target_date=target_date,
        )
        if target_date:
            print(f"  ℹ 指定日期模式: 预测窗口起始 = {target_date}", flush=True)
        prediction = self.numerical_agent.predict(request)
        state["prediction"] = prediction
        context.has_ground_truth = prediction.ground_truth is not None
        context.numerical_confidence = prediction.confidence
        n_ch = len(prediction.forecast)
        n_st = len(prediction.forecast[0]) if prediction.forecast else 0
        ts_range = ""
        if prediction.forecast_timestamps:
            ts_range = f" | {prediction.forecast_timestamps[0][:10]} ~ {prediction.forecast_timestamps[-1][:10]}"
            # 自动将预测窗口起始日写入 context，供 RAG 截止过滤使用
            # 这样 _step_llm_analyze 时 RAG 只会检索此日期之前的历史案例
            context.rag_cutoff_date = prediction.forecast_timestamps[0][:10]
        print(
            f"  ℹ {n_ch} 通道 × {n_st} 步{ts_range} | ground_truth={context.has_ground_truth}",
            flush=True,
        )
        logger.info(
            "  -> %d channels × %d steps | ground_truth=%s",
            n_ch, n_st, context.has_ground_truth,
        )
        return state

    def _step_fetch_events(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        events = []
        if context.event_source and self.event_agent:
            events = self.event_agent.fetch_events(context.event_source)
        state["events"] = events
        context.event_types = list({e.event_type for e in events if e.event_type})
        print(f"  ℹ 共加载 {len(events)} 个事件", flush=True)
        logger.info("  -> Fetched %d events | types=%s", len(events), context.event_types)
        return state

    def _step_filter_events(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        events = state.get("events", [])
        prediction = state.get("prediction")
        if events and prediction and prediction.forecast_timestamps:
            events = self._filter_events_in_window(events, prediction.forecast_timestamps)
        whitelist = set(context.event_station_whitelist or getattr(self, "_event_station_whitelist", set()))
        if whitelist:
            before = len(events)
            events = [
                ev for ev in events
                if str(getattr(ev, "station_complex_id", "") or "") in whitelist
                or (
                    getattr(ev, "channel_name", None)
                    and any(ch in (context.fusion_channel_names or []) for ch in [ev.channel_name])
                )
            ]
            print(f"  ℹ venue37 站点白名单: {before} → {len(events)} 个事件", flush=True)
        state["events"] = events
        context.has_events = len(events) > 0
        print(f"  ℹ 预测窗口内事件: {len(events)} 个", flush=True)
        logger.info("  -> %d events in forecast window", len(events))
        return state

    def _step_event_relevance_filter(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        events = state.get("events", [])
        prediction = state.get("prediction")
        if not events or not self.event_relevance_agent:
            context.has_events = bool(events)
            context.has_major_events = False
            return state

        self.event_relevance_agent.relevance_threshold = context.event_relevance_threshold
        self.event_relevance_agent.max_events = context.max_llm_events_per_window
        target_chs = list(getattr(prediction, "channel_names", None) or [])

        from agents.event_enrichment import enrich_events

        events = enrich_events(events, target_channels=target_chs or None)

        filtered, relevance = self.event_relevance_agent.filter_events(
            events,
            forecast_timestamps=getattr(prediction, "forecast_timestamps", None),
            target_channels=target_chs if target_chs else None,
        )
        state["events"] = filtered
        state["event_relevance"] = relevance
        context.has_events = len(filtered) > 0
        context.has_major_events = any(
            getattr(ev, "impact_tier", "C") in ("A", "B") for ev in filtered
        )
        context.event_types = list({r.semantic_category for r in relevance if r.is_high_impact_candidate})

        disruption_n = sum(1 for r in relevance
                           if getattr(r, "expected_direction", None) == "decrease")
        print(
            f"  ℹ 高影响候选: {len(filtered)} 个 channel-event 对"
            + (f" (含 {disruption_n} 个 disruption↓)" if disruption_n else "")
            + f" | 重大活动(Tier A/B): {'是' if context.has_major_events else '否'}",
            flush=True,
        )
        # 统计去重后的物理事件数
        phys_ids = set()
        for ev in filtered:
            title_norm = (ev.title or "")[:30].strip().lower()
            date_part  = (ev.event_time or "")[:10]
            phys_ids.add(f"{title_norm}|{date_part}")
        if len(phys_ids) < len(filtered):
            print(
                f"  ℹ 去重后物理事件: {len(phys_ids)} 个（{len(filtered)-len(phys_ids)} 个多通道副本保留用于各站点融合）",
                flush=True,
            )
        logger.info("  -> %d high-relevance events retained for LLM/RAG", len(filtered))
        return state

    def _step_extra_rag_retrieve(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        """标记本次使用双倍 top_k 进行 RAG 检索（由 llm_analyze 步骤读取该标记）."""
        state["extra_rag"] = True
        logger.info("  -> Extra RAG flag set: llm_analyze will use top_k × 2")
        return state

    @staticmethod
    def _phys_event_id(event) -> str:
        """物理事件去重键：标题前30字 + 日期（忽略通道差异）."""
        title_norm = (event.title or "")[:30].strip().lower()
        date_part  = (event.event_time or "")[:10]
        return f"{title_norm}|{date_part}"

    def _step_llm_analyze(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        events = state.get("events", [])
        extra_rag = state.get("extra_rag", False)
        impacts: List[EventImpact] = []

        if events and self.event_agent:
            # 从 event_relevance 提取 disruption 方向提示（event_key → expected_direction）
            relevance_list = state.get("event_relevance", [])
            direction_hints = {
                r.event_key: r.expected_direction
                for r in relevance_list
                if getattr(r, "expected_direction", None) is not None
            }
            if direction_hints:
                print(
                    f"  ℹ {len(direction_hints)} 个 disruption 事件将引导方向=decrease",
                    flush=True,
                )

            # ── 物理事件去重：同一物理事件只调用一次 LLM ─────────────────────
            # 问题：同一活动匹配到 N 个站点时会产生 N 条事件记录，
            #       不去重会导致 N 倍 LLM 调用浪费 API 配额。
            # 做法：按 (标题前30字, 日期) 分组；每组只分析「代表事件」一次；
            #       分析结果的 affected_channels 设为整组所有通道，
            #       PredictionFusion 会将其应用到所有关联站点。
            from collections import defaultdict
            phys_groups: dict = defaultdict(list)
            for ev in events:
                phys_groups[self._phys_event_id(ev)].append(ev)

            # 每组取第一个作为代表（已按相关性排序，第一个最优）
            representative_events = [group[0] for group in phys_groups.values()]
            # 为每个代表事件收集全组的 channel_name，供 LLM/Fusion 使用
            group_all_channels: dict = {}   # phys_id → List[str]
            for pid, group in phys_groups.items():
                chs = []
                for ev in group:
                    if ev.channel_name and ev.channel_name not in chs:
                        chs.append(ev.channel_name)
                group_all_channels[pid] = chs

            n_phys = len(representative_events)
            n_total = len(events)
            print(
                f"  ℹ {n_total} 个 channel-event 对 → 去重为 {n_phys} 个物理事件，"
                f"每个物理事件调用 LLM 1 次",
                flush=True,
            )
            if n_total > n_phys:
                print(
                    f"  ℹ 节省 LLM 调用 {n_total - n_phys} 次"
                    f"（{(n_total - n_phys)/n_total*100:.0f}% 配额释放）",
                    flush=True,
                )

            top_k_extra = extra_rag and self.event_agent.rag is not None
            if top_k_extra:
                old_top_k = self.event_agent.rag.top_k
                self.event_agent.rag.top_k = old_top_k * 2

            raw_impacts = self.event_agent.analyze(
                representative_events,
                direction_hints=direction_hints,
                _progress=True,
                rag_cutoff_date=context.rag_cutoff_date,
            )

            if top_k_extra:
                self.event_agent.rag.top_k = old_top_k

            # 将每个分析结果扩展：affected_channels 设为整组所有通道
            for imp, rep_ev in zip(raw_impacts, representative_events):
                pid = self._phys_event_id(rep_ev)
                all_chs = group_all_channels.get(pid, imp.affected_channels)
                updates = {"impact_tier": getattr(rep_ev, "impact_tier", "B")}
                if all_chs and set(all_chs) != set(imp.affected_channels or []):
                    updates["affected_channels"] = all_chs
                if updates:
                    imp = imp.model_copy(update=updates)
                impacts.append(imp)

            logger.info(
                "  -> LLM analyzed %d physical events (covering %d channel-event pairs)",
                n_phys, n_total,
            )

        if context.rag_cutoff_date:
            print(f"  ℹ RAG 截止日: {context.rag_cutoff_date}（已过滤测试期文档）", flush=True)

        state["impacts"] = impacts
        return state

    def _step_skill_calibrate(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        """应用技能库中的 CALIBRATION / CHANNEL / DURATION 技能，校准 LLM 的估计。"""
        impacts: List[EventImpact] = state.get("impacts", [])
        if not impacts or self.skill_dispatcher is None:
            return state

        calibrated: List[EventImpact] = []
        prediction = state.get("prediction")
        all_channel_names = list(getattr(prediction, "channel_names", None) or [])

        for impact in impacts:
            event_type = SkillExtractor.classify_event(impact.event_summary)
            ctx: dict = {"event_type": event_type}

            # 若 SkillExtractor 持有 channel_map，则推断 borough/rank_group 加入查询上下文，
            # 使细粒度技能（如 calib_concert_manhattan_top32）能够被优先匹配
            if self.skill_extractor is not None and self.skill_extractor._channel_meta:
                borough, rank_group = self.skill_extractor._infer_channel_context(
                    impact.affected_channels, all_channel_names
                )
                if borough:
                    ctx["borough"] = borough
                if rank_group:
                    ctx["rank_group"] = rank_group

            matched = self.skill_dispatcher.library.query(
                ctx, min_confidence=cfg.SKILL_MIN_CONFIDENCE
            )

            updates: dict = {}

            # CALIBRATION：校准影响幅度
            calib_skills = [s for s in matched if s.skill_type == SkillType.CALIBRATION]
            if calib_skills:
                ratio = calib_skills[0].action.get("calibration_ratio", 1.0)
                updates["calibrated_magnitude"] = round(impact.impact_magnitude * ratio, 4)
                logger.info(
                    "  CALIBRATION '%s': %.3f × %.3f = %.3f",
                    impact.event_summary[:30],
                    impact.impact_magnitude,
                    ratio,
                    updates["calibrated_magnitude"],
                )

            # CHANNEL：修正受影响通道
            channel_skills = [s for s in matched if s.skill_type == SkillType.CHANNEL]
            if channel_skills:
                new_channels = channel_skills[0].action.get(
                    "affected_channels", impact.affected_channels
                )
                updates["affected_channels"] = new_channels
                logger.info(
                    "  CHANNEL '%s': %s → %s",
                    impact.event_summary[:30],
                    impact.affected_channels,
                    new_channels,
                )

            # DURATION：修正影响持续时长
            duration_skills = [s for s in matched if s.skill_type == SkillType.DURATION]
            if duration_skills:
                new_duration = float(
                    duration_skills[0].action.get("duration_override", impact.impact_duration_hours)
                )
                updates["impact_duration_hours"] = new_duration
                logger.info(
                    "  DURATION '%s': %.1fh → %.1fh",
                    impact.event_summary[:30],
                    impact.impact_duration_hours,
                    new_duration,
                )

            if updates:
                impact = impact.model_copy(update=updates)

            calibrated.append(impact)

        n_calib = sum(1 for imp in calibrated if imp.calibrated_magnitude is not None)
        print(
            f"  ℹ {len(calibrated)} 个事件完成校准，其中 {n_calib} 个命中 Skill 幅度修正",
            flush=True,
        )
        state["impacts"] = calibrated
        return state

    def _step_numerical_only(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        """Passthrough: adjusted_forecast == raw_forecast (LP-MOMENT baseline)."""
        prediction = state.get("prediction")
        result = FinalPrediction(
            raw_forecast=prediction.forecast,
            adjusted_forecast=prediction.forecast,
            events_considered=[],
            explanation="numerical_only (LP-MOMENT baseline, no event correction)",
            channel_names=prediction.channel_names,
            ground_truth=prediction.ground_truth,
            forecast_timestamps=prediction.forecast_timestamps,
            fusion_scope="all",
            channels_adjusted=[],
            channels_passthrough=list(prediction.channel_names or []),
        )
        state["result"] = result
        print("  ℹ numerical_only: adjusted == raw", flush=True)
        return state

    def _step_event_adapter(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        """Apply Codex-style EventResidualAdapter (frozen MOMENT, no head update)."""
        from agents.event_adapter import (
            apply_adapter_to_forecast,
            channel_meta_by_name,
            load_channel_map,
            load_events_json,
        )
        from event_post_training.config import EventPostTrainingConfig

        prediction = state.get("prediction")
        if not self.event_adapter_dir:
            logger.warning("No event adapter loaded; falling back to numerical_only")
            return self._step_numerical_only(state, context)

        ept_cfg = EventPostTrainingConfig()
        channel_names = list(prediction.channel_names or [])
        events = load_events_json(context.event_source) if context.event_source else []
        channel_meta = channel_meta_by_name(load_channel_map(ept_cfg.channel_map_path))

        raw = np.array(prediction.forecast, dtype=np.float32)
        timestamps = prediction.forecast_timestamps or []
        fusion_names = list(context.fusion_channel_names or getattr(self, "_fusion_channel_names", []) or [])
        if fusion_names:
            name_to_idx = {name: i for i, name in enumerate(channel_names)}
            channel_indices = [name_to_idx[name] for name in fusion_names if name in name_to_idx]
            channels_adjusted = [name for name in fusion_names if name in name_to_idx]
            fusion_scope = "venue37"
        else:
            channel_indices = None
            channels_adjusted = list(channel_names)
            fusion_scope = "all"

        adjusted, _ = apply_adapter_to_forecast(
            raw,
            timestamps,
            channel_names,
            events,
            channel_meta,
            self.event_adapter_dir,
            device=self.numerical_agent.device if self.numerical_agent else cfg.DEVICE,
            channel_indices=channel_indices,
        )

        result = FinalPrediction(
            raw_forecast=prediction.forecast,
            adjusted_forecast=adjusted.tolist(),
            events_considered=[],
            explanation=f"EventResidualAdapter frozen_moment ({context._prediction_mode})",
            channel_names=channel_names,
            ground_truth=prediction.ground_truth,
            forecast_timestamps=timestamps,
            fusion_scope=fusion_scope,
            channels_adjusted=channels_adjusted,
            channels_passthrough=[ch for ch in channel_names if ch not in channels_adjusted],
        )
        state["result"] = result
        print(f"  ℹ event_adapter: adjusted {len(channels_adjusted)} channels", flush=True)
        return state

    def _load_venue_history_matrix(self, context: PipelineContext, prediction) -> Optional["np.ndarray"]:
        """Load (n_venue, 512) denormalized history for adapter features."""
        try:
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from event_post_training.config import EventPostTrainingConfig, N_TRAIN_ROWS, SEQ_LEN

            ept_cfg = EventPostTrainingConfig()
            df = pd.read_csv(context.data_path, parse_dates=["date"])
            values = df.drop(columns=["date"]).infer_objects(copy=False).interpolate(method="cubic").values
            scaler = StandardScaler()
            scaler.fit(values[:N_TRAIN_ROWS])
            data_norm = scaler.transform(values)

            target_date = getattr(context, "_target_date", None)
            if target_date and prediction.forecast_timestamps:
                mask = df["date"] >= pd.Timestamp(target_date)
                if mask.any():
                    forecast_start_idx = int(mask.idxmax())
                else:
                    return None
            else:
                forecast_start_idx = len(df) - prediction.forecast_horizon

            hist_start = forecast_start_idx - SEQ_LEN
            if hist_start < 0:
                return None
            block = data_norm[hist_start:forecast_start_idx]
            hist_full = block * scaler.scale_ + scaler.mean_
            return hist_full[:, ept_cfg.venue_top128_indices].T.astype(np.float32)
        except Exception as exc:
            logger.warning("Could not load history matrix for adapter: %s", exc)
            return None

    def _step_fuse(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        prediction = state.get("prediction")
        impacts = state.get("impacts", [])
        result = self.fusion.fuse(prediction, impacts)
        state["result"] = result

        n_matched = len(result.events_considered)
        # 来源统计
        llm_n = sum(1 for imp in result.events_considered
                    if getattr(imp, "analysis_source", "llm") == "llm")
        rule_n = n_matched - llm_n
        scope = getattr(result, "fusion_scope", "all")
        n_adj = len(getattr(result, "channels_adjusted", []) or [])
        print(
            f"  ℹ 事件命中窗口: {n_matched} 个 "
            f"(LLM分析={llm_n}, rule_fallback={rule_n}) | fusion_scope={scope} adjusted_channels={n_adj}",
            flush=True,
        )
        logger.info(
            "  -> Fused. Events matched: %d | channels: %d",
            n_matched, len(result.channel_names or []),
        )
        return state

    def _step_skill_learn(self, state: PipelineState, context: PipelineContext) -> PipelineState:
        """从本次推理结果中提炼技能，写入技能库（需 ground_truth）."""
        result: FinalPrediction = state.get("result")
        if result and self.skill_extractor and result.ground_truth:
            trace_id = uuid.uuid4().hex[:8]
            skills = self.skill_extractor.extract(result, trace_id)
            print(
                f"  ℹ Skill 更新: {len(skills)} 条 (trace={trace_id})",
                flush=True,
            )
            logger.info(
                "  -> SkillExtractor [trace=%s]: %d skills updated/created",
                trace_id, len(skills),
            )
        elif result and not result.ground_truth:
            print("  ℹ 无 ground_truth，跳过 skill_learn", flush=True)
        return state

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_events_in_window(events, forecast_timestamps):
        """只保留 event_time 落在预测窗口时间范围内的事件."""
        try:
            window_start = pd.to_datetime(forecast_timestamps[0]).date()
            window_end   = pd.to_datetime(forecast_timestamps[-1]).date()
        except Exception:
            return events

        filtered = []
        for ev in events:
            if not ev.event_time:
                continue
            try:
                ev_date = pd.to_datetime(ev.event_time).date()
                if window_start <= ev_date <= window_end:
                    filtered.append(ev)
            except Exception:
                filtered.append(ev)
        return filtered


# 向后兼容别名（run.py 中使用 AgentOrchestrator）
AgentOrchestrator = DynamicOrchestrator
