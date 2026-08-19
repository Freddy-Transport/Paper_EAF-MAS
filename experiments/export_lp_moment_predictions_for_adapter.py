#!/usr/bin/env python3
"""Export real LP-MOMENT predictions for event residual-adapter training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_post_training.config import EventPostTrainingConfig, resolve_lp_model_path
from event_post_training.lp_prediction_exporter import export_predictions_for_adapter_training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    p.add_argument("--lp_model_path", default="experiments/outputs/lp_top128")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--negative_ratio", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--splits", default="train,val")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EventPostTrainingConfig(
        traffic_csv=PROJECT_ROOT / args.traffic_csv,
        events_json=PROJECT_ROOT / args.events_json,
        channel_map_path=PROJECT_ROOT / args.channel_map,
        lp_model_path=PROJECT_ROOT / args.lp_model_path,
        output_dir=PROJECT_ROOT / Path(args.output_csv).parent,
        device=args.device,
    )
    resolved_lp, source = resolve_lp_model_path(cfg.lp_model_path, project_root=PROJECT_ROOT)
    manifest = export_predictions_for_adapter_training(
        cfg=cfg,
        resolved_lp_model_path=resolved_lp,
        out_csv=PROJECT_ROOT / args.output_csv,
        stride=args.stride,
        negative_ratio=args.negative_ratio,
        max_windows=args.max_windows,
        device=args.device,
        splits=tuple(x.strip() for x in args.splits.split(',') if x.strip()),
    )
    manifest["lp_model_source"] = source
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
