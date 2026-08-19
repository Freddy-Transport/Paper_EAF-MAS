#!/usr/bin/env python3
"""筛选重大活动日 + 同站对照日，生成 experiment plan JSON.

用法:
  python scripts/select_experiment_dates.py \\
      --query "Times Sq-42 St" \\
      --station-id N060 \\
      --n-major 5 --n-control 5 \\
      --output data/experiment_10runs_plan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.event_tier import classify_impact_tier


def load_events_from_json(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("events", data if isinstance(data, list) else [])


def load_events_from_relevance_csv(path: Path) -> list:
    import pandas as pd

    df = pd.read_csv(path)
    events = []
    for _, row in df.iterrows():
        chs = row.get("affected_channels", "")
        events.append({
            "title": row.get("title", ""),
            "event_time": row.get("event_time", ""),
            "station_complex_id": row.get("station_complex_id", ""),
            "channel_name": chs if isinstance(chs, str) and chs.startswith("N") else "",
            "content": row.get("title", ""),
        })
    return events


def station_match(ev: dict, station_id: str) -> bool:
    sid = str(ev.get("station_complex_id", ""))
    ch = str(ev.get("channel_name", ""))
    return station_id in sid or station_id in ch


def analyze(events: list, station_id: str, date_start: str, date_end: str, target_channel: str):
    by_date_ab: dict = defaultdict(set)
    by_date_any: dict = defaultdict(int)

    for ev in events:
        et = ev.get("event_time", "")[:10]
        if not et or et < date_start or et > date_end:
            continue
        if not station_match(ev, station_id):
            continue
        by_date_any[et] += 1
        tier = classify_impact_tier(ev, target_channels=[target_channel])
        if tier in ("A", "B"):
            title = (ev.get("title") or "")[:40]
            by_date_ab[et].add(title)

    major_scores = sorted(
        [(d, len(titles)) for d, titles in by_date_ab.items()],
        key=lambda x: -x[1],
    )
    all_days = set(by_date_any.keys())
    ab_days = set(by_date_ab.keys())
    control_pool = sorted(all_days - ab_days)
    # 补充：预览库中完全无 N060 记录的工作日
    import pandas as pd

    for d in pd.date_range(date_start, date_end):
        ds = d.strftime("%Y-%m-%d")
        if ds not in all_days and d.weekday() < 5:
            control_pool.append(ds)
    control_pool = sorted(set(control_pool))

    return major_scores, control_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Times Sq-42 St")
    parser.add_argument("--station-id", default="N060")
    parser.add_argument("--target-channel", default="N060__Times_Sq_42_St")
    parser.add_argument("--date-start", default="2023-01-29")
    parser.add_argument("--date-end", default="2023-06-21")
    parser.add_argument("--n-major", type=int, default=5)
    parser.add_argument("--n-control", type=int, default=5)
    parser.add_argument("--events-json", default=str(PROJECT_ROOT / "data/nyc_top128_station_events.json"))
    parser.add_argument(
        "--relevance-csv",
        default=str(PROJECT_ROOT.parent / "纽约地铁数据处理/outputs/top128/nyc_top128_event_relevance_preview.csv"),
    )
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/experiment_10runs_plan.json"))
    args = parser.parse_args()

    events_path = Path(args.events_json)
    if events_path.is_file():
        events = load_events_from_json(events_path)
        source = str(events_path)
    else:
        rel_path = Path(args.relevance_csv)
        if not rel_path.is_file():
            print("未找到事件数据，请先生成 nyc_top128_station_events.json", file=sys.stderr)
            sys.exit(1)
        events = load_events_from_relevance_csv(rel_path)
        source = str(rel_path)

    major_scores, control_pool = analyze(
        events, args.station_id, args.date_start, args.date_end, args.target_channel
    )

    majors = major_scores[: args.n_major]
    controls = control_pool[: args.n_control]

    major_runs = [
        {
            "date": d,
            "run_type": "major",
            "note": f"Tier A/B 物理事件约 {n} 个（数据源: {Path(source).name}）",
        }
        for d, n in majors
    ]
    control_runs = [
        {
            "date": d,
            "run_type": "control",
            "note": "同站无 Tier A/B 重大活动（对照）",
        }
        for d in controls
    ]

    interleaved = []
    mi, ci = 0, 0
    while mi < len(major_runs) or ci < len(control_runs):
        if mi < len(major_runs):
            interleaved.append(major_runs[mi])
            mi += 1
        if ci < len(control_runs):
            interleaved.append(control_runs[ci])
            ci += 1

    plan = {
        "description": f"{args.query} ({args.station_id}) 重大活动日 vs 对照日",
        "query": args.query,
        "station_complex_id": args.station_id,
        "data_source": source,
        "runs": interleaved,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out} ({len(interleaved)} runs)")
    for i, r in enumerate(interleaved, 1):
        print(f"  {i:02d} [{r['run_type']:7s}] {r['date']}  {r['note']}")


if __name__ == "__main__":
    main()
