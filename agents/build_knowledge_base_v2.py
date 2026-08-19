"""知识库 V2 — LLM 驱动的交通数据与事件数据时间对齐分析.

相比 build_knowledge_base.py 的种子文档方式，V2 基于真实数据：
  1. 解析事件日期，与交通时序数据按日对齐
  2. 统计事件日 vs 非事件日（区分工作日/周末）的客流差异
  3. 计算变化率和 z-score 显著性
  4. 可选：用 LLM (Qwen3/GPT) 将统计结果转化为高质量自然语言知识文档

Usage:
    # 纯数据模板模式（无需 LLM API）
    python -m agents.build_knowledge_base_v2 \
        --traffic_path data/Brooklyn_Bacaly_two_modal.csv \
        --event_path data/text_feat_0802.24.csv

    # LLM 增强模式
    export LLM_API_KEY="your-key"
    python -m agents.build_knowledge_base_v2 \
        --traffic_path data/Brooklyn_Bacaly_two_modal.csv \
        --event_path data/text_feat_0802.24.csv \
        --use_llm
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents import config as cfg

logger = logging.getLogger(__name__)


# ============================================================
# Step 1: 时间对齐 — 将事件与交通数据按日期匹配
# ============================================================
class TrafficEventAligner:
    """将交通时序数据与事件数据按日期对齐，计算事件日与非事件日的客流差异."""

    def __init__(self, traffic_path: str, channel_names: Optional[List[str]] = None):
        self.df = pd.read_csv(traffic_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["date_only"] = self.df["date"].dt.date
        self.df["hour"] = self.df["date"].dt.hour
        self.df["weekday"] = self.df["date"].dt.weekday

        self.channel_cols = [c for c in self.df.columns if c not in ["date", "date_only", "hour", "weekday"]]
        self.channel_names = channel_names or self.channel_cols
        logger.info(
            "Traffic data: %d rows, date range %s ~ %s, channels: %s",
            len(self.df),
            self.df["date"].min().date(),
            self.df["date"].max().date(),
            self.channel_cols,
        )

    def compute_daily_stats(self) -> pd.DataFrame:
        """计算每日各通道的统计量（日总量、日均值、日最大值）."""
        daily = self.df.groupby("date_only")[self.channel_cols].agg(["sum", "mean", "max"])
        daily.columns = ["_".join(col) for col in daily.columns]
        daily["weekday"] = pd.to_datetime(daily.index).weekday
        daily["is_weekend"] = daily["weekday"].isin([5, 6])
        return daily

    def compute_baselines(self) -> Dict:
        """计算非事件日的基线统计量（按工作日/周末分组）."""
        daily = self.compute_daily_stats()
        baselines = {}
        for ch in self.channel_cols:
            col = f"{ch}_sum"
            baselines[ch] = {
                "weekday_avg": float(daily.loc[~daily["is_weekend"], col].mean()),
                "weekend_avg": float(daily.loc[daily["is_weekend"], col].mean()),
                "overall_avg": float(daily[col].mean()),
                "weekday_std": float(daily.loc[~daily["is_weekend"], col].std()),
                "weekend_std": float(daily.loc[daily["is_weekend"], col].std()),
            }
        return baselines

    def compute_hourly_pattern(self, target_date) -> Optional[Dict]:
        """计算指定日期的逐小时客流模式."""
        mask = self.df["date_only"] == target_date
        if not mask.any():
            return None
        hourly = self.df.loc[mask].groupby("hour")[self.channel_cols].sum()
        return {ch: hourly[ch].tolist() for ch in self.channel_cols}

    def align_events(self, events: List[dict]) -> List[dict]:
        """将事件与交通数据按日期对齐，计算事件日的实际影响."""
        baselines = self.compute_baselines()
        daily = self.compute_daily_stats()
        aligned = []

        for event in events:
            event_date_str = str(event.get("event_time", ""))
            try:
                event_date = pd.to_datetime(event_date_str).date()
            except Exception:
                continue

            if event_date not in daily.index:
                continue

            row = daily.loc[event_date]
            is_weekend = bool(row["is_weekend"])

            impact_data = {}
            for ch in self.channel_cols:
                actual = float(row[f"{ch}_sum"])
                baseline_key = "weekend_avg" if is_weekend else "weekday_avg"
                baseline = baselines[ch][baseline_key]
                baseline_std = baselines[ch]["weekend_std" if is_weekend else "weekday_std"]

                change_pct = (actual - baseline) / baseline * 100 if baseline > 0 else 0.0
                z_score = (actual - baseline) / baseline_std if baseline_std > 0 else 0.0

                impact_data[ch] = {
                    "actual_daily_total": round(actual, 1),
                    "baseline_daily_total": round(baseline, 1),
                    "change_percent": round(change_pct, 2),
                    "z_score": round(z_score, 2),
                    "is_significant": abs(z_score) > 1.5,
                }

            hourly = self.compute_hourly_pattern(event_date)

            aligned.append({
                "event": event,
                "event_date": str(event_date),
                "is_weekend": is_weekend,
                "impact_data": impact_data,
                "hourly_pattern": hourly,
            })

        logger.info("Aligned %d / %d events with traffic data.", len(aligned), len(events))
        return aligned


# ============================================================
# Step 2: 结构化过滤 — 按时间/地点/显著性筛选
# ============================================================
def filter_significant_events(aligned: List[dict], min_z_score: float = 1.0) -> List[dict]:
    """过滤出至少一个通道有显著影响的事件."""
    filtered = []
    for item in aligned:
        has_significant = any(
            abs(d["z_score"]) >= min_z_score for d in item["impact_data"].values()
        )
        if has_significant:
            filtered.append(item)
    logger.info("Filtered to %d significant events (|z| >= %.1f).", len(filtered), min_z_score)
    return filtered




def load_channel_metadata(channel_map_path: Optional[str]) -> Dict[str, dict]:
    """读取 Top-N 站点通道映射；没有映射时返回空字典。"""
    if not channel_map_path:
        return {}
    path = Path(channel_map_path)
    if not path.exists():
        logger.warning("Channel map not found: %s", channel_map_path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = payload.get("channels", payload if isinstance(payload, list) else [])
    return {str(row.get("channel_name")): row for row in channels if row.get("channel_name")}


def generate_normal_pattern_documents(aligner: TrafficEventAligner, channel_map_path: Optional[str] = None) -> List[dict]:
    """为无事件窗口生成常规客流模式知识文档。"""
    channel_meta = load_channel_metadata(channel_map_path)
    df = aligner.df.copy()
    docs = []
    for channel in aligner.channel_cols:
        hourly = df.groupby("hour")[channel].mean()
        peak_hour = int(hourly.idxmax()) if len(hourly) else -1
        weekday_mean = float(df.loc[df["weekday"] < 5, channel].mean())
        weekend_mean = float(df.loc[df["weekday"] >= 5, channel].mean())
        meta = channel_meta.get(channel, {})
        station_name = meta.get("station_complex", channel)
        borough = meta.get("borough", "")
        routes = meta.get("routes", "")
        text = (
            f"Normal ridership pattern for channel {channel} ({station_name}). "
            f"Borough: {borough}; routes: {routes}. "
            f"Average weekday hourly ridership is {weekday_mean:.1f}; "
            f"average weekend hourly ridership is {weekend_mean:.1f}; "
            f"typical peak hour is {peak_hour}:00. "
            "Use this as baseline context when no high-impact event is present."
        )
        docs.append({
            "text": text,
            "meta": {
                "source": "normal_pattern",
                "channel": channel,
                "station_complex_id": meta.get("station_complex_id", ""),
                "borough": borough,
            },
        })
    return docs

# ============================================================
# Step 3a: 模板化知识文档生成（无需 LLM）
# ============================================================
def generate_template_documents(aligned_events: List[dict]) -> List[dict]:
    """从对齐数据直接生成知识文档（无需 LLM API）."""
    documents = []
    for item in aligned_events:
        event = item["event"]
        day_type = "Weekend" if item["is_weekend"] else "Weekday"
        parts = [
            f"Event: {event.get('title', 'Unknown Event')}.",
            f"Date: {item['event_date']} ({day_type}).",
            f"Location: {event.get('location', 'Barclays Center')}.",
            f"Type: {event.get('event_type', 'unknown')}.",
        ]

        for ch, data in item["impact_data"].items():
            sig = "statistically significant" if data["is_significant"] else "within normal range"
            direction = "increase" if data["change_percent"] > 0 else "decrease"
            parts.append(
                f"Measured impact on {ch}: {direction} of {abs(data['change_percent']):.1f}% "
                f"vs {day_type.lower()} baseline "
                f"(actual={data['actual_daily_total']:.0f}, baseline={data['baseline_daily_total']:.0f}, "
                f"z-score={data['z_score']:.2f}, {sig})."
            )

        documents.append({
            "text": " ".join(parts),
            "meta": {
                "event_date": item["event_date"],
                "event_type": event.get("event_type", "unknown"),
                "location": event.get("location", ""),
                "borough": event.get("borough", ""),
                "station_complex_id": event.get("station_complex_id", ""),
                "day_type": "weekend" if item["is_weekend"] else "weekday",
                "source": "data_analysis",
            },
        })
    return documents


# ============================================================
# Step 3b: LLM 增强知识文档生成
# ============================================================
LLM_KNOWLEDGE_PROMPT = """\
You are a transportation data analyst. Based on the following real data, \
generate a structured knowledge document about this event's impact on transit ridership.

## Event
- Title: {title}
- Date: {event_date} ({day_type})
- Location: {location}
- Type: {event_type}
- Description: {content}

## Measured Traffic Impact
{impact_summary}

## Task
Write a concise knowledge document in English:

Event: [name and type].
Date: [date and day type].
Location: [location].
Measured Impact: [for each channel, actual percentage change and significance].
Likely Cause: [brief analysis].
Duration: [estimated].
Confidence: [high/medium/low based on z-score].

Output ONLY the knowledge document text.
"""


def generate_llm_documents(
    aligned_events: List[dict],
    api_key: str,
    base_url: str,
    model_name: str,
    llm_top_n: int = 800,
) -> List[dict]:
    """用 LLM 从对齐数据中生成高质量知识文档.

    只对 Z-score 绝对值最高的 llm_top_n 条事件调用 LLM，其余用模板生成，
    避免对 3 万+ 条事件逐条调 API 导致运行时间过长。

    Args:
        llm_top_n: 最多调用 LLM 的事件数（按最大 |z_score| 排序取 top）。
                   默认 800，约 20 分钟完成（按 ~1.5s/call 估算）。
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.error("langchain_openai not installed. Falling back to template mode.")
        return generate_template_documents(aligned_events)

    if not api_key:
        logger.warning("LLM_API_KEY not set. Falling back to template mode.")
        return generate_template_documents(aligned_events)

    # 按最大 |z_score| 降序排列，取 top-N 用 LLM，其余用模板
    def _max_abs_z(item):
        return max((abs(d.get("z_score", 0)) for d in item["impact_data"].values()), default=0)

    sorted_events = sorted(aligned_events, key=_max_abs_z, reverse=True)
    llm_events = sorted_events[:llm_top_n]
    template_events = sorted_events[llm_top_n:]

    total = len(aligned_events)
    print(
        f"\n[LLM知识库] 共 {total} 条显著事件。"
        f"LLM 处理 top-{len(llm_events)} 条（|z|最大），"
        f"其余 {len(template_events)} 条用模板。",
        flush=True,
    )

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, temperature=0.2, max_tokens=512)
    documents = []

    try:
        from tqdm import tqdm
        _iter = tqdm(llm_events, desc="LLM文档生成", unit="事件")
    except ImportError:
        _iter = llm_events

    for item in _iter:
        event = item["event"]
        day_type = "Weekend" if item["is_weekend"] else "Weekday"

        impact_lines = []
        for ch, data in item["impact_data"].items():
            sig = "SIGNIFICANT" if data["is_significant"] else "not significant"
            impact_lines.append(
                f"- {ch}: daily total = {data['actual_daily_total']:.0f} "
                f"(baseline = {data['baseline_daily_total']:.0f}, "
                f"change = {data['change_percent']:+.1f}%, "
                f"z-score = {data['z_score']:.2f}, {sig})"
            )

        prompt = LLM_KNOWLEDGE_PROMPT.format(
            title=event.get("title", "Unknown"),
            event_date=item["event_date"],
            day_type=day_type,
            location=event.get("location", "Barclays Center"),
            event_type=event.get("event_type", "unknown"),
            content=event.get("content", event.get("title", "")),
            impact_summary="\n".join(impact_lines),
        )

        try:
            response = llm.invoke(prompt)
            doc_text = response.content.strip()
            source = "llm_analyzed"
        except Exception as e:
            logger.warning("LLM call failed for '%s': %s. Using template.", event.get("title", "")[:30], e)
            template_docs = generate_template_documents([item])
            doc_text = template_docs[0]["text"] if template_docs else ""
            source = "template_fallback"

        documents.append({
            "text": doc_text,
            "meta": {
                "event_date": item["event_date"],
                "event_type": event.get("event_type", "unknown"),
                "location": event.get("location", ""),
                "borough": event.get("borough", ""),
                "station_complex_id": event.get("station_complex_id", ""),
                "day_type": "weekend" if item["is_weekend"] else "weekday",
                "source": source,
            },
        })
        logger.info("Generated: %s (%s)", event.get("title", "")[:40], item["event_date"])

    # 其余事件用模板生成
    if template_events:
        print(f"[LLM知识库] 剩余 {len(template_events)} 条用模板补全...", flush=True)
        documents.extend(generate_template_documents(template_events))

    llm_count = sum(1 for d in documents if d["meta"].get("source") == "llm_analyzed")
    print(f"[LLM知识库] 完成：LLM生成 {llm_count} 条，模板补全 {len(documents)-llm_count} 条。", flush=True)
    return documents


# ============================================================
# 知识库构建主流程
# ============================================================
def build_v2_knowledge_base(
    output_dir: str,
    traffic_path: str,
    event_path: str,
    use_llm: bool = False,
    min_z_score: float = 1.0,
    include_seeds: bool = True,
    channel_names: Optional[List[str]] = None,
    channel_map_path: Optional[str] = None,
    llm_rescore_path: Optional[str] = None,
    llm_top_n: int = 800,
    station_whitelist: Optional[List[str]] = None,
):
    """V2 知识库构建主入口.

    Args:
        llm_rescore_path: LLM 重打分后的 CSV 路径（relevance_preview_llm.csv）。
            若提供，则将 LLM 判定为低相关（relevance_score < 0.45）的事件从
            对齐列表中过滤掉，提升知识库文档质量。
    """
    from agents.build_knowledge_base import SEED_DOCUMENTS, build_from_seeds
    from agents.event_fetcher import EventFetcher
    from agents.rag_pipeline import RAGPipeline

    # Step 0: 种子文档基础
    if include_seeds:
        build_from_seeds(output_dir)

    # Step 1: 加载事件数据
    fetcher = EventFetcher()
    events = fetcher.fetch(event_path)

    # Step 2: 时间对齐
    aligner = TrafficEventAligner(traffic_path, channel_names)
    if not events:
        logger.warning("No events loaded from %s; building normal-pattern knowledge only.", event_path)
        documents = generate_normal_pattern_documents(aligner, channel_map_path)
        pipeline = RAGPipeline(knowledge_base_dir=output_dir)
        pipeline.add_documents([d["text"] for d in documents], [d["meta"] for d in documents])
        return

    event_dicts = [e.model_dump() for e in events]
    if station_whitelist:
        whitelist = set(station_whitelist)
        before = len(event_dicts)
        event_dicts = [
            e for e in event_dicts
            if str(e.get("station_complex_id", "") or "") in whitelist
        ]
        logger.info(
            "Station whitelist filter: %d → %d events (venue37).",
            before, len(event_dicts),
        )
    logger.info("Loaded %d events.", len(event_dicts))

    # 若提供 LLM 重打分结果，过滤掉低相关事件
    if llm_rescore_path and os.path.isfile(llm_rescore_path):
        import pandas as _pd
        llm_df = _pd.read_csv(llm_rescore_path)
        # 构建 title+time 的低相关 key 集合
        low_rel_keys = set()
        for _, row in llm_df.iterrows():
            if float(row.get("relevance_score", 1.0)) < 0.45:
                low_rel_keys.add(str(row.get("event_key", "")))
        before = len(event_dicts)
        event_dicts = [
            e for e in event_dicts
            if "|".join([str(e.get("title", "")), str(e.get("event_time", "")), str(e.get("location", ""))]) not in low_rel_keys
        ]
        logger.info(
            "LLM rescore filter: %d → %d events (removed %d low-relevance).",
            before, len(event_dicts), before - len(event_dicts),
        )

    aligned = aligner.align_events(event_dicts)
    if not aligned:
        logger.warning("No events aligned with traffic data date range.")
        return

    normal_documents = generate_normal_pattern_documents(aligner, channel_map_path)

    # Step 3: 结构化过滤 + Tier A/B（仅重大活动进专家案例库）
    from agents.event_tier import classify_impact_tier, tier_is_kb_eligible

    significant = filter_significant_events(aligned, min_z_score=max(min_z_score, 1.2))
    tier_significant = []
    for item in significant:
        ev = item.get("event", {})
        tier = classify_impact_tier(ev)
        if tier_is_kb_eligible(tier):
            item["impact_tier"] = tier
            tier_significant.append(item)
    logger.info(
        "Tier+significance filter: %d → %d events for KB",
        len(significant),
        len(tier_significant),
    )
    significant = tier_significant

    # Step 4: 生成知识文档（仅验真后的重大活动）
    if use_llm:
        documents = generate_llm_documents(significant, cfg.LLM_API_KEY, cfg.LLM_BASE_URL, cfg.LLM_MODEL, llm_top_n=llm_top_n)
    else:
        documents = generate_template_documents(significant)

    for d in documents:
        d.setdefault("meta", {})["impact_tier"] = d["meta"].get("impact_tier", "B")

    documents = normal_documents + documents

    # Step 5: 写入向量知识库
    pipeline = RAGPipeline(knowledge_base_dir=output_dir)
    texts = [d["text"] for d in documents]
    metas = [d["meta"] for d in documents]
    pipeline.add_documents(texts, metas)
    logger.info("Added %d data-driven documents to knowledge base.", len(documents))

    # Step 6: 保存分析报告（确保目录存在）
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "alignment_report.json")
    report = {
        "summary": {
            "total_events": len(event_dicts),
            "aligned_events": len(aligned),
            "significant_events": len(significant),
            "documents_generated": len(documents),
            "normal_pattern_documents": len(normal_documents),
            "baselines": aligner.compute_baselines(),
        },
        "events": [],
    }
    for a, d in zip(aligned, documents[: len(aligned)]):
        report["events"].append({
            "title": a["event"].get("title", ""),
            "date": a["event_date"],
            "is_weekend": a["is_weekend"],
            "impact": a["impact_data"],
            "generated_text": d["text"][:500],
        })

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Report saved to %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Base V2 (data-driven + LLM)")
    parser.add_argument("--traffic_path", type=str, default="data/Brooklyn_Bacaly_two_modal.csv")
    parser.add_argument("--event_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=cfg.KNOWLEDGE_BASE_DIR)
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for knowledge generation")
    parser.add_argument("--min_z_score", type=float, default=1.0, help="Min z-score for significant events")
    parser.add_argument("--no_seeds", action="store_true", help="Skip seed documents")
    parser.add_argument("--channel_names", type=str, nargs="+", default=None)
    parser.add_argument("--channel_map", type=str, default=None, help="Top-N station channel map JSON")
    parser.add_argument("--traffic-csv", type=str, default=None, help="Alias for --traffic_path")
    parser.add_argument("--events-json", type=str, default=None, help="Alias for --event_path")
    parser.add_argument("--output-dir", type=str, default=None, help="Alias for --output_dir")
    parser.add_argument(
        "--station-whitelist",
        type=str,
        default=None,
        help="JSON file with station_complex_ids to index (venue37)",
    )
    args = parser.parse_args()

    if args.traffic_csv:
        args.traffic_path = args.traffic_csv
    if args.events_json:
        args.event_path = args.events_json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if not Path(args.traffic_path).is_absolute():
        args.traffic_path = str(Path(cfg.PROJECT_ROOT) / args.traffic_path)

    channel_map = args.channel_map
    if channel_map and not Path(channel_map).is_absolute():
        channel_map = str(Path(cfg.PROJECT_ROOT) / channel_map)

    event_path = args.event_path or cfg.EVENT_DATA_PATH
    if not Path(event_path).is_absolute():
        event_path = str(Path(cfg.PROJECT_ROOT) / event_path)

    station_whitelist = None
    if args.station_whitelist:
        sw_path = Path(args.station_whitelist)
        if not sw_path.is_absolute():
            sw_path = Path(cfg.PROJECT_ROOT) / sw_path
        if sw_path.is_file():
            payload = json.loads(sw_path.read_text(encoding="utf-8"))
            station_whitelist = payload.get("station_complex_ids", payload if isinstance(payload, list) else [])

    build_v2_knowledge_base(
        output_dir=args.output_dir,
        traffic_path=args.traffic_path,
        event_path=event_path,
        use_llm=args.use_llm,
        min_z_score=args.min_z_score,
        include_seeds=not args.no_seeds,
        channel_names=args.channel_names,
        channel_map_path=channel_map,
        station_whitelist=station_whitelist,
    )


if __name__ == "__main__":
    main()
