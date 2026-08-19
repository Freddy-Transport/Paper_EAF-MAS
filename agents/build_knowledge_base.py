"""构建 RAG 知识库 — 基于种子文档和事件数据.

Usage:
    python -m agents.build_knowledge_base --seeds_only
    python -m agents.build_knowledge_base --event_path data/text_feat_0802.24.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents import config as cfg

logger = logging.getLogger(__name__)

SEED_DOCUMENTS = [
    {
        "text": (
            "Event: NBA Brooklyn Nets Home Game at Barclays Center. "
            "Date: typical game nights (October-April). "
            "Impact: Metro ridership around Barclays Center increases by 30-50% "
            "during pre-game (2h before) and post-game (1h after). "
            "Taxi drop-offs increase by 40-60% in the same period. "
            "Duration: approximately 5 hours."
        ),
        "meta": {"type": "sports", "location": "Barclays Center", "source": "expert"},
    },
    {
        "text": (
            "Event: Major Concert at Barclays Center (capacity ~19,000). "
            "Date: various dates year-round. "
            "Impact: Metro ridership increases by 40-70% around Atlantic Ave station "
            "2 hours before and 1 hour after event. "
            "Taxi demand surges by 50-80% post-event. "
            "Duration: approximately 6 hours."
        ),
        "meta": {"type": "concert", "location": "Barclays Center", "source": "expert"},
    },
    {
        "text": (
            "Event: New Year's Eve (December 31 - January 1). "
            "Impact: Subway ridership increases by 60-100% from 8PM to 3AM. "
            "Taxi demand surges significantly after midnight. "
            "Duration: approximately 8 hours."
        ),
        "meta": {"type": "holiday", "location": "citywide", "source": "expert"},
    },
    {
        "text": (
            "Event: Thanksgiving (4th Thursday of November). "
            "Impact: Overall public transit ridership decreases by 20-40% during the day. "
            "Evening ridership may increase near event venues. "
            "Duration: full day."
        ),
        "meta": {"type": "holiday", "location": "citywide", "source": "expert"},
    },
    {
        "text": (
            "Event: Severe Weather (snowstorm, heavy rain, extreme cold). "
            "Impact: Subway ridership decreases by 10-30%. "
            "Taxi demand increases by 15-25% as commuters avoid walking. "
            "Duration: varies, typically 12-48 hours."
        ),
        "meta": {"type": "weather", "location": "citywide", "source": "expert"},
    },
    {
        "text": (
            "Event: Subway Service Disruption (planned or unplanned). "
            "Impact: Affected line ridership drops significantly. "
            "Taxi and ride-share demand near affected stations increases by 30-50%. "
            "Duration: varies from hours to days."
        ),
        "meta": {"type": "disruption", "location": "varies", "source": "expert"},
    },
    {
        "text": (
            "Event: Regular weekday commute pattern (Monday-Friday). "
            "Impact: Subway ridership peaks at 7-9 AM and 5-7 PM. "
            "Taxi demand peaks at 8-10 AM and 6-8 PM. "
            "Weekend ridership is typically 40-60% of weekday levels."
        ),
        "meta": {"type": "pattern", "location": "citywide", "source": "expert"},
    },
    {
        "text": (
            "Event: Independence Day (July 4th). "
            "Impact: Daytime transit ridership decreases by 20-30% (holiday). "
            "Evening ridership increases near fireworks viewing areas. "
            "Duration: evening into night."
        ),
        "meta": {"type": "holiday", "location": "citywide", "source": "expert"},
    },
    {
        "text": (
            "Event: Boxing or UFC Fight Night at Barclays Center. "
            "Impact: Similar to NBA game but with higher peak concentration. "
            "Metro ridership increases 35-55% near event time. "
            "Post-event taxi surge is sharper (60-90% increase) as events end later. "
            "Duration: approximately 5-6 hours."
        ),
        "meta": {"type": "sports", "location": "Barclays Center", "source": "expert"},
    },
    {
        "text": (
            "Event: Graduation Ceremony at Barclays Center. "
            "Impact: Metro ridership increases by 20-35% in the morning. "
            "Taxi drop-offs increase by 25-40%. "
            "Duration: approximately 4 hours."
        ),
        "meta": {"type": "ceremony", "location": "Barclays Center", "source": "expert"},
    },
]


def build_from_seeds(output_dir: str):
    """用种子文档构建初始知识库."""
    from agents.rag_pipeline import RAGPipeline

    documents = [d["text"] for d in SEED_DOCUMENTS]
    metadatas = [d["meta"] for d in SEED_DOCUMENTS]

    pipeline = RAGPipeline(knowledge_base_dir=output_dir)
    pipeline.build_knowledge_base(documents, metadatas)
    logger.info("Seed knowledge base built with %d documents.", len(documents))


def try_add_event_data(output_dir: str, event_path: str):
    """从事件数据文件追加文档到知识库."""
    from agents.event_fetcher import EventFetcher
    from agents.rag_pipeline import RAGPipeline

    fetcher = EventFetcher()
    events = fetcher.fetch(event_path)

    if not events:
        logger.warning("No events loaded from %s", event_path)
        return

    documents = []
    metadatas = []
    for ev in events:
        text = (
            f"Event: {ev.title}. "
            f"Date: {ev.event_time}. "
            f"Location: {ev.location or 'Barclays Center'}. "
            f"Description: {ev.content}. "
            f"Type: {ev.event_type or 'unknown'}."
        )
        documents.append(text)
        metadatas.append({
            "source": ev.source,
            "event_time": ev.event_time,
            "type": ev.event_type or "unknown",
        })

    pipeline = RAGPipeline(knowledge_base_dir=output_dir)
    pipeline.add_documents(documents, metadatas)
    logger.info("Added %d event documents from %s", len(documents), event_path)


def main():
    parser = argparse.ArgumentParser(description="Build RAG Knowledge Base")
    parser.add_argument("--traffic_path", type=str, default=None)
    parser.add_argument("--event_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=cfg.KNOWLEDGE_BASE_DIR)
    parser.add_argument("--seeds_only", action="store_true", help="Only use seed documents")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    build_from_seeds(args.output_dir)

    if args.event_path and not args.seeds_only:
        event_path = args.event_path
        if not Path(event_path).is_absolute():
            event_path = str(Path(cfg.PROJECT_ROOT) / event_path)
        try_add_event_data(args.output_dir, event_path)

    logger.info("Knowledge base ready at: %s", args.output_dir)


if __name__ == "__main__":
    main()
