"""智能体间通信的数据结构定义."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """协调器 -> 数值预测智能体."""

    history_data_path: str
    forecast_horizon: int = 192
    n_channels: int = 2
    target_date: Optional[str] = None


class NumericalPrediction(BaseModel):
    """数值预测智能体 -> 协调器."""

    forecast: List[List[float]]        # (C, H) 反归一化后的预测值
    channel_names: List[str]
    confidence: Optional[float] = None
    ground_truth: Optional[List[List[float]]] = None      # (C, H) 同窗口的真实值（用于可视化）
    forecast_timestamps: Optional[List[str]] = None       # 长度 H，每步的 ISO 时间戳


class EventInfo(BaseModel):
    """单条事件."""

    source: str = "unknown"
    title: str = ""
    content: str = ""
    event_time: str = ""
    location: Optional[str] = None
    event_type: Optional[str] = None
    event_id: Optional[str] = None
    station_complex_id: Optional[str] = None
    station_complex: Optional[str] = None
    channel_name: Optional[str] = None
    distance_m: Optional[float] = None
    venue_name: Optional[str] = None
    # ── 分级与 enrichment（由 event_tier / event_enrichment 填充）──────────
    impact_tier: str = "C"  # A / B / C / D
    event_category: Optional[str] = None  # sports / concert / parade / ...
    enriched_description: Optional[str] = None
    impact_confidence: str = "low"  # high / medium / low
    should_adjust_forecast: bool = False


class EventRelevance(BaseModel):
    """事件语义与站点相关性筛选结果."""

    event_key: str
    title: str = ""
    event_time: str = ""
    station_complex_id: Optional[str] = None
    affected_channels: List[str] = Field(default_factory=list)
    semantic_category: str = "unknown"
    relevance_score: float = 0.0
    is_high_impact_candidate: bool = False
    evidence: List[str] = Field(default_factory=list)
    # 预期影响方向提示：disruption 类事件标记 "decrease"，引导后续 LLM 分析
    # None 表示方向由 LLM 自行判断
    expected_direction: Optional[str] = None


class EventImpact(BaseModel):
    """LLM 输出的事件影响评估."""

    event_summary: str
    affected_channels: List[str] = Field(default_factory=list)
    impact_direction: str = "neutral"  # "increase" / "decrease" / "neutral"
    impact_magnitude: float = 0.0  # 0.0 ~ 1.0
    impact_duration_hours: float = 0.0
    reasoning: str = ""
    event_time: Optional[str] = None   # ISO datetime，用于时间感知融合
    calibrated_magnitude: Optional[float] = None  # 技能校准后的幅度，优先于 impact_magnitude
    # 分析来源标注：用于实验日志可复现性追踪
    # "llm"          — LLM 正常返回并解析成功
    # "rule_fallback"— LLM 不可用或 JSON 解析失败，使用规则推断
    analysis_source: str = "llm"
    # 推理链字段（用于论文实验可视化）
    rag_context_preview: Optional[str] = None   # 检索到的历史证据摘要（前300字）
    llm_reasoning_brief: Optional[str] = None   # LLM 推理摘要（来自 reasoning 字段）
    impact_tier: str = "B"  # 来源事件的 Tier，用于融合 cap


class FinalPrediction(BaseModel):
    """最终融合输出."""

    raw_forecast: List[List[float]]
    adjusted_forecast: List[List[float]]
    events_considered: List[EventImpact] = Field(default_factory=list)
    explanation: str = ""
    channel_names: Optional[List[str]] = None
    ground_truth: Optional[List[List[float]]] = None      # 同窗口真实值，用于可视化
    forecast_timestamps: Optional[List[str]] = None       # 预测窗口时间戳
    fusion_scope: str = "all"
    channels_adjusted: List[str] = Field(default_factory=list)
    channels_passthrough: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# AutoSkill / 动态管线相关 Schema
# ──────────────────────────────────────────────────────────────

class SkillType(str, Enum):
    """技能类型枚举."""

    CALIBRATION = "calibration"  # 量化：校正 LLM 估计的影响幅度
    ROUTING     = "routing"      # 路由：动态增减管线步骤
    CHANNEL     = "channel"      # 通道：修正受影响通道列表
    DURATION    = "duration"     # 时长：修正事件影响持续时间


class Skill(BaseModel):
    """可持久化、可检索的技能单元."""

    skill_id: str
    skill_type: SkillType
    trigger: Dict[str, Any] = Field(default_factory=dict)   # 触发条件，e.g. {"event_type": "concert"}
    action: Dict[str, Any] = Field(default_factory=dict)    # 执行动作，e.g. {"calibration_ratio": 1.3}
    confidence: float = 0.5       # 置信度 [0, 1]
    use_count: int = 0
    support_count: int = 0        # 被经验支持次数（实际误差方向与技能一致）
    last_updated: str = ""
    source_trace_ids: List[str] = Field(default_factory=list)  # 产生该技能的经验轨迹 ID


class PipelineContext(BaseModel):
    """编排器传递给 SkillDispatcher 的上下文快照."""

    data_path: str
    event_source: Optional[str] = None
    forecast_horizon: int = 192
    n_channels: int = 2
    event_types: List[str] = Field(default_factory=list)  # 采集到的事件类型列表（filter_events 后填充）
    has_events: bool = False       # filter_events 后若有窗口内事件则为 True
    has_major_events: bool = False  # Tier A/B 重大活动，为 True 时才走 LLM 校准
    has_ground_truth: bool = False # numerical_predict 后若有 ground_truth 则为 True
    numerical_confidence: Optional[float] = None  # 来自 NumericalPrediction.confidence
    channel_map_path: Optional[str] = None
    station_ids: List[str] = Field(default_factory=list)
    event_relevance_threshold: float = 0.45
    max_llm_events_per_window: int = 80
    # ── 数据泄漏防护 ─────────────────────────────────────────────────────
    # update_skills=False 时跳过 skill_learn 步骤（评估/测试模式下使用）
    update_skills: bool = True
    # rag_cutoff_date：RAG 检索时只使用该日期之前的历史文档；
    # 通常由 orchestrator 自动设置为预测窗口起始日（forecast_timestamps[0][:10]），
    # 防止检索到与测试预测窗口重叠的实测事件数据。
    rag_cutoff_date: Optional[str] = None
    event_station_whitelist: List[str] = Field(default_factory=list)
    fusion_channel_names: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
