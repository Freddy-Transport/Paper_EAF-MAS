#!/usr/bin/env bash
# 全流程（规则增强）跑完后，用 LLM 增强验真事件并重建 KB/Fusion，便于效果对比。
set -euo pipefail

DATA_ROOT="/root/autodl-tmp/纽约地铁数据处理"
FRAMEWORK="/root/autodl-tmp/0206moment"

if [[ -f "$FRAMEWORK/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FRAMEWORK/.env"
  set +a
fi

cd "$FRAMEWORK"

echo "=== LLM 增强验真事件（796 条量级）==="
python scripts/enrich_venue_events.py \
  --input "$DATA_ROOT/outputs/venue37_analysis/venue37_verified_events.json" \
  --output "$DATA_ROOT/outputs/venue37_analysis/venue37_verified_events_llm.json" \
  --use-llm

echo "=== 重建 venue37 知识库（LLM 事件）==="
python agents/build_knowledge_base_v2.py \
  --traffic-csv data/nyc_top128_station_hourly_flow.csv \
  --events-json "$DATA_ROOT/outputs/venue37_analysis/venue37_verified_events_llm.json" \
  --channel_map data/nyc_top128_channel_map.json \
  --output-dir agents/knowledge_base_venue37_llm \
  --station-whitelist "$DATA_ROOT/outputs/venue37_station_ids.json" \
  --no_seeds \
  --use_llm

python build_residual_rag.py \
  --events "$DATA_ROOT/outputs/venue37_analysis/venue37_verified_events_llm.json" \
  --csv data/nyc_top128_station_hourly_flow.csv \
  --output agents/knowledge_base_residual_venue37_llm

echo "=== Fusion 预测（LLM 增强版）==="
python run_venue37_fusion.py \
  --date 2023-05-20 \
  --events-json "$DATA_ROOT/outputs/venue37_analysis/venue37_verified_events_llm.json" \
  --kb-dir agents/knowledge_base_venue37_llm \
  --residual-kb-dir agents/knowledge_base_residual_venue37_llm \
  --save-trace traces/venue37_fusion_llm_run01.json \
  --device cuda:0

echo "对比基线 trace: traces/venue37_fusion_run01.json"
echo "对比 LLM trace:  traces/venue37_fusion_llm_run01.json"
