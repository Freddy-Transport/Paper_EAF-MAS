#!/usr/bin/env python3
"""LLM 增强场馆事件描述（venue37 流水线）."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

FRAMEWORK_DIR = Path("/root/autodl-tmp/0206moment")
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

from agents import config as cfg

cfg.load_env_file()
from agents.event_enrichment import enrich_event_rule, enrich_events
from agents.schemas import EventInfo

logger = logging.getLogger(__name__)

ENRICH_PROMPT_EXTRA = """
Focus on NYC subway ridership near the matched station. Include:
- expected crowd direction (increase/decrease/neutral)
- peak hours relative to event start
- whether this is a major venue event worth forecast adjustment
"""


def row_to_event(row: dict) -> EventInfo:
    return EventInfo(
        source=str(row.get("source", "venue37_curated")),
        title=str(row.get("title", row.get("event_name", ""))),
        content=str(row.get("description", row.get("content", ""))),
        event_time=str(row.get("start_datetime", row.get("event_time", "")))[:19],
        location=str(row.get("venue_name", row.get("event_location", ""))),
        event_type=str(row.get("event_type", "unknown")),
        station_complex_id=str(row.get("station_complex_id", "") or None),
        station_complex=str(row.get("station_complex", "") or None),
        venue_name=str(row.get("venue_name", "") or None),
        distance_m=float(row["distance_m"]) if row.get("distance_m") not in (None, "") else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier-min", default="B", choices=["A", "B", "C"])
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    path = Path(args.input)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("events", [])
    else:
        rows = pd.read_csv(path).to_dict("records")

    if args.limit:
        rows = rows[: args.limit]

    use_llm = args.use_llm or bool(cfg.LLM_API_KEY)
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    min_tier = tier_order[args.tier_min]

    event_objs = [row_to_event(row) for row in rows]
    enriched_objs = enrich_events(event_objs, use_llm=use_llm)

    enriched_events = []
    for ev, row in zip(enriched_objs, rows):
        if tier_order.get(ev.impact_tier, 9) > min_tier:
            continue
        payload = ev.model_dump()
        payload.update(
            {
                "station_complex_id": row.get("station_complex_id"),
                "station_complex": row.get("station_complex"),
                "venue_name": row.get("venue_name"),
                "source_url": row.get("source_url", ""),
            }
        )
        enriched_events.append(payload)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"events": enriched_events}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"enriched {len(enriched_events)} events -> {out}")


if __name__ == "__main__":
    main()
