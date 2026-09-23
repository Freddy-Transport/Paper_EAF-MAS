"""全局配置 — 自动适配本地 Mac 与 AutoDL 环境."""

import os
from pathlib import Path

import torch

# ---- 项目根目录 (agents/ 的上一级) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env_file(env_path: str | Path | None = None, override: bool = False) -> bool:
    """加载项目根目录 `.env`（支持 `export KEY=value` 与 `KEY=value` 格式）.

    仅在环境变量尚未设置时写入（override=False），避免覆盖用户已在 shell 中 export 的值。
    """
    path = Path(env_path) if env_path else Path(PROJECT_ROOT) / ".env"
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if not override and key in os.environ and os.environ[key]:
            continue
        os.environ[key] = value
    return True


# 启动时自动加载 0206moment/.env（若存在）
load_env_file()

# ---- 设备与精度 ----
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ---- 模型与数据路径（相对 PROJECT_ROOT，AutoDL / 本地通用）----
MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EVENT_DATA_PATH = os.path.join(DATA_DIR, "text_feat_0802.24.csv")  # Excel 格式
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "agents", "knowledge_base")

# ---- 残差型 RAG 知识库（模型边际修正量，由 build_residual_rag.py 生成）----
# 存储 (actual - model_pred) / model_pred 的历史统计，供 LLM 参考边际修正量
# 设为 None 则回退到原有知识库
RESIDUAL_KB_DIR = os.path.join(PROJECT_ROOT, "agents", "knowledge_base_residual")

# 是否启用残差知识库（False=使用原始专家知识库，True=使用残差知识库）
USE_RESIDUAL_KB = True

# ---- LLM 配置（兼容 LLM_API_KEY 和 OPENAI_API_KEY 两种变量名）----
LLM_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "EMPTY")
)
LLM_BASE_URL = (
    os.environ.get("LLM_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
)
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")

# ---- Moment 模型参数（与训练时保持一致）----
SEQ_LEN = 512
PATCH_LEN = 8
PATCH_STRIDE_LEN = 8
FORECAST_HORIZON = 192
D_MODEL = 1024
N_CHANNELS = 2
NUM_PROMPT_TOKENS = 16
PG_DROPOUT_PROB = 0.3
MAX_CHANNELS = 128
PG_NUM_HEADS = 4
USE_GRAPH = True
N_GCN_LAYERS = 2
GRAPH_EMBED_DIM = 32
GRAPH_DROPOUT = 0.1

# ---- RAG 配置 ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5

# ---- Linear Probing 回退配置 ----
# 当用户未上传自定义模型权重时，自动用官方 MOMENT Linear Probing 训练并缓存权重
LP_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "outputs", "lp_fallback_nyc_top128")
LP_DATA_PATH = os.path.join(DATA_DIR, "Brooklyn_Bacaly_two_modal.csv")
LP_MAX_EPOCH = 5          # 训练轮数（5 轮足以使预测头收敛）
LP_LR = 1e-4              # 学习率
LP_BATCH_SIZE = 64        # 批大小

# ---- AutoSkill / 动态管线配置 ----
# 技能库持久化文件（JSONL，每行一个 Skill JSON 对象）
SKILL_LIBRARY_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "skill_library.jsonl")
# 技能最低置信度阈值：低于此值的技能不参与管线决策
SKILL_MIN_CONFIDENCE = 0.2
# SkillExtractor：误差超过此比例才生成路由技能（触发 extra_rag）
SKILL_ROUTING_ERROR_THRESHOLD = 0.15

# ---- 站点通道映射文件（Top-N JSON，含 rank/borough/station_complex_id）----
# 用于 SkillExtractor 细粒度 trigger（event_type + borough + rank_group）
CHANNEL_MAP_PATH = os.path.join(DATA_DIR, "nyc_top128_channel_map.json")
VENUE_STATION_MAP_PATH = os.path.join(DATA_DIR, "venue_station_map.json")

# 事件分级与 enrichment（P0+）
EVENT_TIER_ENABLED = True

# Formal evaluation uses explicit manifests and cell-level calibration.
PREDICTION_MODE = os.environ.get("PREDICTION_MODE", "eafmas_formal")
EVENT_ADAPTER_PATH = os.path.join(
    PROJECT_ROOT, "experiments", "outputs", "event_adapter", "frozen_moment"
)
EVENT_ADAPTER_PHASE2_PATH = EVENT_ADAPTER_PATH  # head frozen; same adapter dir
