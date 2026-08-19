#!/bin/bash
# =============================================================
# AutoDL 一键运行脚本
# 使用方式：
#   source .env  # 或手动 export LLM_API_KEY="..."
#   bash scripts/run_on_autodl.sh
#
# 支持透传额外参数给 agents.run，例如：
#   bash scripts/run_on_autodl.sh --forecast_horizon 96 --output my_result.json
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- 自动加载 .env 文件（如果存在）----
if [ -f ".env" ]; then
    echo "Loading environment from .env ..."
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ---- HuggingFace 镜像（AutoDL 国内环境）----
if [ -z "$HF_ENDPOINT" ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
fi

echo "===== Multi-Agent Traffic Prediction System ====="
echo "Project root: $PROJECT_ROOT"
echo "Device:       $(python -c 'import torch; print("cuda:0 (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo 'unknown')"
echo "HF endpoint:  $HF_ENDPOINT"

# ---- 检查 LLM API Key ----
if [ -z "$LLM_API_KEY" ]; then
    echo ""
    echo "WARNING: LLM_API_KEY not set."
    echo "  Event analysis will run in fallback mode (neutral impact for all events)."
    echo "  To enable LLM: cp .env.example .env && fill in your key, then: source .env"
else
    echo "LLM model:    ${LLM_MODEL:-gpt-4o} via ${LLM_BASE_URL:-https://api.openai.com/v1}"
fi

# ---- 检查数据文件 ----
echo ""
DATA_FILE="$PROJECT_ROOT/data/Brooklyn_Bacaly_two_modal.csv"
EVENT_FILE="$PROJECT_ROOT/data/text_feat_0802.24.csv"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Traffic data not found: $DATA_FILE"
    echo "  Please upload the data/ directory to AutoDL first."
    exit 1
fi

if [ ! -f "$EVENT_FILE" ]; then
    echo "WARNING: Event data not found: $EVENT_FILE"
    echo "  Event analysis will be skipped."
    EVENT_FILE=""
fi

# ---- 构建知识库（如果不存在）----
if [ ! -f "agents/knowledge_base/index.faiss" ]; then
    echo ""
    echo "[Step 0] Building RAG knowledge base V2 (first run only)..."
    if [ -n "$EVENT_FILE" ]; then
        # V2：基于真实交通数据 + 事件数据，计算统计显著性，可选 LLM 增强
        if [ -n "$LLM_API_KEY" ]; then
            echo "  Mode: data-driven + LLM enhanced"
            python -m agents.build_knowledge_base_v2 \
                --traffic_path data/Brooklyn_Bacaly_two_modal.csv \
                --event_path data/text_feat_0802.24.csv \
                --channel_names taxi subway \
                --use_llm
        else
            echo "  Mode: data-driven (no LLM key, using template)"
            python -m agents.build_knowledge_base_v2 \
                --traffic_path data/Brooklyn_Bacaly_two_modal.csv \
                --event_path data/text_feat_0802.24.csv \
                --channel_names taxi subway
        fi
    else
        # 无事件文件时退回种子文档模式
        echo "  Mode: seed documents only (no event file)"
        python -m agents.build_knowledge_base --seeds_only
    fi
fi

# ---- 运行预测 ----
echo ""
echo "[Running] Multi-Agent Prediction Pipeline..."
echo "---"

if [ -n "$EVENT_FILE" ]; then
    python -m agents.run \
        --data_path data/Brooklyn_Bacaly_two_modal.csv \
        --event_path data/text_feat_0802.24.csv \
        --forecast_horizon 192 \
        --n_channels 2 \
        --channel_names taxi subway \
        --output results.json \
        "$@"
else
    python -m agents.run \
        --data_path data/Brooklyn_Bacaly_two_modal.csv \
        --forecast_horizon 192 \
        --n_channels 2 \
        --channel_names taxi subway \
        --output results.json \
        "$@"
fi

echo ""
echo "===== Done ====="
echo "Results saved to: $PROJECT_ROOT/results.json"
