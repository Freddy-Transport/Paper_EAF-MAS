#!/usr/bin/env python3
"""生成 venue37 与 Top128 channel_map 的 Fusion 白名单."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path("/root/autodl-tmp/纽约地铁数据处理")
FRAMEWORK_DIR = Path("/root/autodl-tmp/0206moment")
for p in (PROJECT_DIR, FRAMEWORK_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from subway_event_pipeline.top_stations import intersect_with_channel_map, load_venue37_station_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--venue37-ids",
        default=str(PROJECT_DIR / "outputs" / "venue37_station_ids.json"),
    )
    parser.add_argument(
        "--venue-matches",
        default=str(PROJECT_DIR / "outputs" / "venue_station_matches.csv"),
    )
    parser.add_argument(
        "--channel-map",
        default=str(FRAMEWORK_DIR / "data" / "nyc_top128_channel_map.json"),
    )
    parser.add_argument(
        "--output",
        default=str(FRAMEWORK_DIR / "data" / "venue37_fusion_channels.json"),
    )
    args = parser.parse_args()

    ids_path = Path(args.venue37_ids)
    if ids_path.is_file():
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        venue37_ids = payload.get("station_complex_ids", [])
    else:
        venue37_ids = load_venue37_station_ids(args.venue_matches)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(
            json.dumps(
                {
                    "station_complex_ids": venue37_ids,
                    "source": str(args.venue_matches),
                    "station_count": len(venue37_ids),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    channel_map = json.loads(Path(args.channel_map).read_text(encoding="utf-8"))
    result = intersect_with_channel_map(venue37_ids, channel_map)
    result["fusion_scope"] = "venue37"
    result["source_channel_map"] = str(args.channel_map)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"venue37 stations: {result['venue37_total']}")
    print(f"intersection with Top128: {result['intersection_count']}")
    print(f"fusion channels: {len(result['channel_names'])}")
    if result["missing_from_top_n"]:
        print(f"missing from Top128: {result['missing_from_top_n']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
