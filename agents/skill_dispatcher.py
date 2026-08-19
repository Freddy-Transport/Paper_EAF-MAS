"""技能调度器 — 根据当前上下文检索技能库，动态组装管线步骤列表。

调度分两阶段：
  Phase-1  plan_initial(context)  → 在事件采集前确定初始步骤
  Phase-2  plan_event_steps(context) → 在 filter_events 之后、event_types 已知时
           决定事件分析子步骤（含可选的 extra_rag_retrieve）

两阶段分离的原因：只有采集到事件后才能知道事件类型，才能用于 ROUTING 技能的触发匹配。
"""

import logging
from typing import List

from agents import config as cfg
from agents.schemas import PipelineContext, SkillType
from agents.skill_library import SkillLibrary

logger = logging.getLogger(__name__)


class SkillDispatcher:
    """技能驱动的管线规划器。

    AVAILABLE_STEPS 定义了所有可用步骤的合法名称集合，
    DynamicOrchestrator 的 step_registry 须为每个步骤注册对应函数。
    """

    AVAILABLE_STEPS: List[str] = [
        "numerical_predict",    # 必须：数值基础预测
        "fetch_events",         # 可选：采集事件列表
        "filter_events",        # 可选：过滤窗口外事件
        "extra_rag_retrieve",   # 可选：增强 RAG 检索（top_k × 2）
        "llm_analyze",          # 可选：LLM 事件影响分析
        "skill_calibrate",      # 可选：技能校准（CALIBRATION/CHANNEL/DURATION）
        "fuse",                 # 必须：融合数值预测与事件影响
        "skill_learn",          # 可选：从本次结果中学习技能（需 ground_truth）
    ]

    def __init__(
        self,
        library: SkillLibrary,
        min_confidence: float = None,
    ):
        self.library = library
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else cfg.SKILL_MIN_CONFIDENCE
        )

    # ------------------------------------------------------------------
    # 两阶段规划接口
    # ------------------------------------------------------------------

    def plan_initial(self, context: PipelineContext) -> List[str]:
        """Phase-1：在事件采集前决定初始步骤序列。

        此时 event_types 为空，只能基于 has_events / event_source 做粗粒度决策。
        返回的步骤列表包含一个 "__REPLAN__" 占位符，
        DynamicOrchestrator 在执行到它时会调用 plan_event_steps 来替换后续步骤。
        """
        steps = ["numerical_predict"]

        if context.event_source or context.has_events:
            steps += ["fetch_events", "filter_events", "__REPLAN__"]

        steps.append("fuse")
        return steps

    def plan_event_steps(self, context: PipelineContext) -> List[str]:
        """Phase-2：事件类型已知后，决定事件分析子步骤。

        匹配 ROUTING 技能：若有匹配的技能要求 add_step=extra_rag_retrieve，
        则在 llm_analyze 前插入该步骤。
        """
        steps: List[str] = []

        # 无重大活动（Tier A/B）时跳过 LLM/RAG 校准，预测保持数值模型输出
        if not getattr(context, "has_major_events", context.has_events):
            logger.info(
                "SkillDispatcher: no Tier A/B major events, skipping event analysis steps."
            )
            return steps

        # 查询 ROUTING 技能（以 event_types 为匹配键）
        ctx_dict = context.to_dict()
        routing_skills = self.library.query(
            ctx_dict,
            skill_type=SkillType.ROUTING,
            min_confidence=self.min_confidence,
        )

        need_extra_rag = any(
            s.action.get("add_step") == "extra_rag_retrieve"
            for s in routing_skills
        )

        if need_extra_rag:
            steps.append("extra_rag_retrieve")
            logger.info(
                "SkillDispatcher: ROUTING skill triggered extra_rag_retrieve "
                "(matched skills: %s)",
                [s.skill_id for s in routing_skills if s.action.get("add_step") == "extra_rag_retrieve"],
            )

        steps.append("llm_analyze")
        steps.append("skill_calibrate")

        return steps

    def plan_postprocess(self, context: PipelineContext) -> List[str]:
        """Phase-3：融合后的后处理步骤（如技能学习）.

        当 context.update_skills=False 时（测试/评估模式），跳过 skill_learn，
        避免用测试集 ground_truth 污染技能库。
        """
        steps: List[str] = []
        if context.has_ground_truth and context.update_skills:
            steps.append("skill_learn")
        elif context.has_ground_truth and not context.update_skills:
            pass  # 冻结模式：有 GT 但不更新技能
        return steps

    # ------------------------------------------------------------------
    # 便捷接口：一次性生成完整计划（用于不需要动态更新的简单场景）
    # ------------------------------------------------------------------

    def plan_full(self, context: PipelineContext) -> List[str]:
        """生成完整步骤列表（适用于 event_types 已提前填充的场景）。

        若 event_types 为空而 event_source 存在，会包含 fetch/filter 步骤，
        但 routing 技能无法精确匹配；此时退化为不加 extra_rag。
        """
        steps = ["numerical_predict"]

        if context.event_source or context.has_events:
            steps += ["fetch_events", "filter_events"]
            steps += self.plan_event_steps(context)

        steps.append("fuse")
        steps += self.plan_postprocess(context)
        return steps

    # ------------------------------------------------------------------
    # 诊断工具
    # ------------------------------------------------------------------

    def describe_plan(self, context: PipelineContext) -> str:
        """返回当前技能状态和计划步骤的文字描述（用于日志/调试）."""
        all_skills = self.library.load_all()
        lines = [
            f"SkillLibrary: {len(all_skills)} skills total",
        ]
        for s in all_skills:
            lines.append(
                f"  [{s.skill_type.value}] {s.skill_id}  "
                f"confidence={s.confidence:.2f}  use={s.use_count}  "
                f"action={s.action}"
            )
        plan = self.plan_full(context)
        lines.append(f"Planned steps: {' → '.join(plan)}")
        return "\n".join(lines)
