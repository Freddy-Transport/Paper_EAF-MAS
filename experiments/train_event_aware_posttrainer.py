#!/usr/bin/env python3
"""Train event-aware residual adapter for Top128 forecasts."""

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
from event_post_training.sample_builder import build_training_arrays, save_moment_cache
from event_post_training.trainer import train_adapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    p.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    p.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    p.add_argument("--moment_predictions_csv", default=None)
    p.add_argument("--generate_moment_predictions", action="store_true")
    p.add_argument("--strict_real_moment_predictions", action="store_true", default=True)
    p.add_argument("--allow_deterministic_fallback", action="store_true")
    p.add_argument("--lp_model_path", default="experiments/outputs/lp_top128")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["frozen_moment", "peft_moment"], default="frozen_moment")
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--negative_ratio", type=float, default=1.0)
    p.add_argument("--residual_bound", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stride", type=int, default=24)
    return p.parse_args()



def validate_prediction_policy(args) -> None:
    if args.mode == "peft_moment":
        raise SystemExit("PEFT-head training is disabled for the current paper framework; use --mode frozen_moment.")
    if not args.allow_deterministic_fallback and not args.moment_predictions_csv and not args.generate_moment_predictions:
        raise SystemExit("Real LP-MOMENT residual adapter training requires --moment_predictions_csv or --generate_moment_predictions. Use --allow_deterministic_fallback only for explicit debugging.")

def main() -> None:
    args = parse_args()
    validate_prediction_policy(args)
    device = args.device
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    cfg = EventPostTrainingConfig(
        traffic_csv=PROJECT_ROOT / args.traffic_csv,
        events_json=PROJECT_ROOT / args.events_json,
        channel_map_path=PROJECT_ROOT / args.channel_map,
        lp_model_path=PROJECT_ROOT / args.lp_model_path,
        output_dir=PROJECT_ROOT / args.output_dir,
        residual_bound=args.residual_bound,
        device=device,
    )
    resolved_lp, source = resolve_lp_model_path(cfg.lp_model_path, project_root=PROJECT_ROOT)
    generated_prediction_manifest = None
    moment_predictions_csv = PROJECT_ROOT / args.moment_predictions_csv if args.moment_predictions_csv else None
    if args.generate_moment_predictions:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        moment_predictions_csv = cfg.output_dir / "moment_predictions_train_val.csv"
        generated_prediction_manifest = export_predictions_for_adapter_training(
            cfg=cfg,
            resolved_lp_model_path=resolved_lp,
            out_csv=moment_predictions_csv,
            stride=args.stride,
            negative_ratio=args.negative_ratio,
            max_windows=args.max_windows,
            device=device,
            splits=("train", "val"),
        )
    samples = build_training_arrays(
        cfg.traffic_csv,
        cfg.events_json,
        cfg.channel_map_path,
        cfg.seq_len,
        cfg.horizon,
        cfg.train_rows,
        cfg.val_rows,
        cfg.residual_bound,
        stride=args.stride,
        negative_ratio=args.negative_ratio,
        max_windows=args.max_windows,
        moment_predictions_csv=moment_predictions_csv,
        splits=("train", "val"),
        strict_prediction_csv=bool(args.strict_real_moment_predictions and not args.allow_deterministic_fallback),
        allow_deterministic_fallback=bool(args.allow_deterministic_fallback),
    )
    manifest = {
        "traffic_csv": str(cfg.traffic_csv),
        "events_json": str(cfg.events_json),
        "channel_map": str(cfg.channel_map_path),
        "requested_lp_model_path": str(cfg.lp_model_path),
        "resolved_lp_model_path": str(resolved_lp),
        "lp_model_source": source,
        "sample_counts": {k: len(v) for k, v in samples.items()},
        "moment_predictions_csv": str(moment_predictions_csv) if moment_predictions_csv else None,
        "generated_prediction_manifest": generated_prediction_manifest,
        "prediction_sources": sorted({s.get("prediction_source", "unknown") for rows in samples.values() for s in rows}),
        "real_lp_moment_predictions": bool(moment_predictions_csv and not args.allow_deterministic_fallback),
        "deterministic_fallback_used": any(s.get("prediction_source") == "deterministic_lag_cache" for rows in samples.values() for s in rows),
        "moment_head_training": False,
        "note": "formal adapter training uses real LP-MOMENT predictions; deterministic lag fallback requires --allow_deterministic_fallback",
    }
    if args.allow_deterministic_fallback and not moment_predictions_csv:
        save_moment_cache(samples, cfg.output_dir / "moment_cache")
        manifest["moment_cache_dir"] = str(cfg.output_dir / "moment_cache")
    result = train_adapter(
        samples,
        cfg.output_dir,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=64,
        residual_bound=cfg.residual_bound,
        device=device,
        manifest=manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
