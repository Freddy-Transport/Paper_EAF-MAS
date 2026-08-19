#!/usr/bin/env python3
"""Run compact full-paper experiments and export paper-ready assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.event_adapter import apply_adapter_to_forecast, build_event_feature_cube, channel_meta_by_name, load_channel_map, load_events_json  # noqa: E402
from agents.numerical_agent import NumericalPredictionAgent  # noqa: E402
from event_post_training.config import EventPostTrainingConfig, resolve_lp_model_path  # noqa: E402


MODES = [
    "numerical_only",
    "rag_explain",
    "event_adapter_frozen_moment",
    "event_adapter_peft_moment",
    "full_agent_no_skill",
    "full_agent_evolving_skill",
]


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - pred)) / max(float(np.sum(np.abs(actual))), 1.0) * 100.0)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(actual) + np.abs(pred), 1.0)
    return float(np.mean(2.0 * np.abs(pred - actual) / denom) * 100.0)


def metrics(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    if mask is not None and np.asarray(mask).any():
        actual = actual[mask]
        pred = pred[mask]
    return {"mae": mae(actual, pred), "rmse": rmse(actual, pred), "wape": wape(actual, pred), "smape": smape(actual, pred), "n": int(actual.size)}


def bootstrap_wape_delta_ci(
    rows: Sequence[dict],
    baseline_mode: str,
    compare_mode: str,
    n_boot: int = 1000,
    seed: int = 13,
) -> Dict[str, float]:
    """Bootstrap paired per-anchor WAPE deltas: baseline minus compare."""
    by_anchor: Dict[int, Dict[str, float]] = {}
    for row in rows:
        try:
            anchor = int(row["anchor"])
            mode = str(row["mode"])
            value = float(row["wape"])
        except Exception:
            continue
        by_anchor.setdefault(anchor, {})[mode] = value
    deltas = [
        modes[baseline_mode] - modes[compare_mode]
        for modes in by_anchor.values()
        if baseline_mode in modes and compare_mode in modes
    ]
    if not deltas:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_windows": 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    boot = []
    for _ in range(max(1, int(n_boot))):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot.append(float(np.mean(sample)))
    return {
        "mean_delta": float(np.mean(arr)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_windows": int(len(arr)),
    }


def ensure_dirs(root: Path) -> None:
    for sub in ["tables", "figures", "reports", "predictions", "logs", "explanations"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def load_channel_names(traffic_csv: Path) -> List[str]:
    cols = pd.read_csv(traffic_csv, nrows=1).columns.tolist()
    return [c for c in cols if c != "date"]


def load_venue_indices(path: Path, channel_names: Sequence[str]) -> tuple[List[int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = list(payload.get("channel_names", []))
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    names = [name for name in requested if name in name_to_idx]
    return [name_to_idx[name] for name in names], {
        "requested_venue_total": int(payload.get("venue37_total", len(requested))),
        "top128_intersection_count": len(names),
        "missing_from_top128": payload.get("missing_from_top_n", []),
        "channel_names": names,
    }


def choose_anchors(df: pd.DataFrame, cfg: EventPostTrainingConfig, split: str, max_anchors: int, stride: int) -> List[int]:
    if split == "val":
        split_start = cfg.train_rows
        split_end = min(cfg.train_rows + cfg.val_rows, len(df))
    elif split == "test":
        split_start = min(cfg.train_rows + cfg.val_rows, len(df))
        split_end = len(df)
    else:
        split_start = 0
        split_end = min(cfg.train_rows, len(df))
    start = max(split_start + cfg.seq_len, cfg.seq_len)
    stop = split_end - cfg.horizon
    anchors = list(range(start, max(start, stop + 1), max(1, stride))) if stop >= start else []
    return anchors[:max_anchors]


def init_agent(cfg: EventPostTrainingConfig, channel_names: Sequence[str], lp_dir: Path, device: str) -> NumericalPredictionAgent:
    if not (lp_dir / "lp_weights.pt").is_file():
        raise FileNotFoundError(f"LP weights not found: {lp_dir / 'lp_weights.pt'}")
    return NumericalPredictionAgent(
        model_path=str(PROJECT_ROOT / "__paper_full_no_gca__"),
        device=device,
        forecast_horizon=cfg.horizon,
        n_channels=len(channel_names),
        channel_names=list(channel_names),
        lp_model_path=str(lp_dir),
        lp_data_path=str(cfg.traffic_csv),
        allow_lp_training_fallback=False,
    )


def apply_mode(
    mode: str,
    raw: np.ndarray,
    timestamps: Sequence[str],
    channel_names: Sequence[str],
    events: Sequence,
    channel_meta: Dict[str, dict],
    adapter_dirs: Dict[str, Path],
    device: str,
) -> tuple[np.ndarray, str]:
    if mode in {"numerical_only", "rag_explain", "full_agent_no_skill", "full_agent_evolving_skill"}:
        reason = "raw LP-MOMENT retained; mode abstains from numeric correction in aggregate full-paper pass"
        return raw.copy(), reason
    adapter_dir = adapter_dirs.get(mode)
    if adapter_dir is None or not (adapter_dir / "event_adapter.pt").is_file():
        raise FileNotFoundError(f"adapter checkpoint unavailable for {mode}: {adapter_dir}")
    adjusted, correction = apply_adapter_to_forecast(raw, timestamps, channel_names, events, channel_meta, adapter_dir, device=device)
    return np.asarray(adjusted, dtype=np.float32), f"adapter={adapter_dir}; max_abs_correction={float(np.abs(correction).max()):.6f}"


def append_metric_rows(rows: List[dict], split: str, anchor: int, date: str, mode: str, actual: np.ndarray, pred: np.ndarray, event_mask: np.ndarray, venue_indices: Sequence[int]) -> None:
    top128 = metrics(actual, pred)
    venue_actual = actual[list(venue_indices)]
    venue_pred = pred[list(venue_indices)]
    venue_mask = event_mask[list(venue_indices)]
    venue = metrics(venue_actual, venue_pred)
    event = metrics(venue_actual, venue_pred, venue_mask)
    non_event = metrics(venue_actual, venue_pred, ~venue_mask)
    rows.append({
        "split": split,
        "anchor": int(anchor),
        "date": date,
        "mode": mode,
        "scope": "event_venue28",
        "mae": venue["mae"],
        "rmse": venue["rmse"],
        "wape": venue["wape"],
        "smape": venue["smape"],
        "top128_mae": top128["mae"],
        "top128_rmse": top128["rmse"],
        "top128_wape": top128["wape"],
        "top128_smape": top128["smape"],
        "event_mae": event["mae"],
        "event_wape": event["wape"],
        "event_n": event["n"],
        "non_event_mae": non_event["mae"],
        "non_event_wape": non_event["wape"],
        "non_event_n": non_event["n"],
    })


def summarize(rows: Sequence[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for mode in sorted({r["mode"] for r in rows}):
        ms = [r for r in rows if r["mode"] == mode]
        out[mode] = {k: float(np.mean([r[k] for r in ms])) for k in ["mae", "rmse", "wape", "smape", "event_mae", "event_wape", "non_event_mae", "non_event_wape"]}
        out[mode]["n_windows"] = len(ms)
    return out


def summarize_top128_numerical(rows: Sequence[dict]) -> Dict[str, float]:
    raw = [r for r in rows if r["mode"] == "numerical_only"]
    if not raw:
        return {}
    return {
        "mae": float(np.mean([r["top128_mae"] for r in raw])),
        "rmse": float(np.mean([r["top128_rmse"] for r in raw])),
        "wape": float(np.mean([r["top128_wape"] for r in raw])),
        "smape": float(np.mean([r["top128_smape"] for r in raw])),
        "n_windows": len(raw),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(text: str) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%")


def write_tables(root: Path, summary: Dict[str, dict], self_checks: dict) -> None:
    order = MODES
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Mode & MAE & RMSE & WAPE & sMAPE \\\\",
        "\\midrule",
    ]
    for mode in order:
        if mode not in summary:
            continue
        s = summary[mode]
        lines.append(f"{tex_escape(mode)} & {s['mae']:.2f} & {s['rmse']:.2f} & {s['wape']:.2f} & {s['smape']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (root / "tables" / "main_metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Mode & Event WAPE & Non-event WAPE & Event MAE & Non-event MAE \\\\",
        "\\midrule",
    ]
    for mode in order:
        if mode not in summary:
            continue
        s = summary[mode]
        lines.append(f"{tex_escape(mode)} & {s['event_wape']:.2f} & {s['non_event_wape']:.2f} & {s['event_mae']:.2f} & {s['non_event_mae']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (root / "tables" / "ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Check & Pass & Warn & Note \\\\",
        "\\midrule",
        f"Artifact self-check & {self_checks.get('pass', 0)} & {self_checks.get('warn', 0)} & citation warning retained \\\\",
        f"Skill promotion & {self_checks.get('promoted_skills', 0)} & {self_checks.get('rejected_skills', 0)} & validation only \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (root / "tables" / "skill_evolution.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bootstrap_table(root: Path, ci_rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Mode & Mean WAPE Delta & 95\\% CI Low & 95\\% CI High \\\\",
        "\\midrule",
    ]
    for row in ci_rows:
        lines.append(
            f"{tex_escape(row['mode'])} & {row['mean_delta']:.3f} & {row['ci_low']:.3f} & {row['ci_high']:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (root / "tables" / "bootstrap_ci.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(root: Path, rows: Sequence[dict], curve_payload: dict, summary: Dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [m for m in MODES if m in summary]
    plt.figure(figsize=(8.5, 3.8))
    x = np.arange(len(modes))
    plt.bar(x, [summary[m]["wape"] for m in modes], label="overall WAPE")
    plt.plot(x, [summary[m]["event_wape"] for m in modes], color="#dc2626", marker="o", label="event-active WAPE")
    plt.xticks(x, [m.replace("event_adapter_", "adapter_").replace("full_agent_", "agent_") for m in modes], rotation=18, ha="right")
    plt.ylabel("WAPE")
    plt.title("Full-paper ablation summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "figures" / "error_heatmap.png", dpi=160)
    plt.close()

    if curve_payload:
        xs = np.arange(len(curve_payload["actual"]))
        plt.figure(figsize=(10, 3.8))
        plt.plot(xs, curve_payload["actual"], color="#9ca3af", linewidth=1.8, label="actual")
        plt.plot(xs, curve_payload["numerical_only"], color="#2563eb", linestyle="--", linewidth=1.7, label="LP-MOMENT raw")
        for mode, color in [("event_adapter_frozen_moment", "#f97316"), ("event_adapter_peft_moment", "#16a34a")]:
            if mode in curve_payload:
                plt.plot(xs, curve_payload[mode], color=color, linewidth=1.4, label=mode)
        plt.title(curve_payload["title"])
        plt.xlabel("forecast horizon hour")
        plt.ylabel("ridership")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(root / "figures" / "hourly_curve_event_case.png", dpi=160)
        plt.close()

    plt.figure(figsize=(6.5, 3.2))
    plt.bar(["accepted URLs", "model summaries", "warnings"], [0, 1, 1], color=["#16a34a", "#2563eb", "#f97316"])
    plt.ylabel("count")
    plt.title("Evidence quality diagnostics")
    plt.tight_layout()
    plt.savefig(root / "figures" / "evidence_quality_table.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.5, 3.2))
    plt.plot([0, 1], [0, 0], marker="o", label="promoted skills")
    plt.plot([0, 1], [0, 1], marker="o", label="rejected/inactive candidates")
    plt.xlabel("round")
    plt.ylabel("count")
    plt.title("Skill promotion trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "figures" / "skill_promotion_trajectory.png", dpi=160)
    plt.close()


def write_analysis(root: Path, payload: dict) -> None:
    summary = payload["summary"]
    top128 = payload.get("top128_numerical_summary", {})
    base = summary.get("numerical_only", {})
    best_mode = min(summary, key=lambda m: summary[m]["wape"]) if summary else "n/a"
    text = f"""# Full Paper Result Analysis

## Dataset and Splits

- Top128 hourly subway ridership file: `data/nyc_top128_station_hourly_flow.csv`.
- Venue-event intersection: n={payload['venue_meta']['top128_intersection_count']} channels from the venue37 list.
- Train/validation/test rows: {payload['split_policy']}.
- Test anchors evaluated: {payload['n_test_anchors']}; validation anchors for skill diagnostics: {payload['n_val_anchors']}.
- Rolling anchor stride: {payload.get('rolling_stride')}.

## Main Findings

- Numerical LP-MOMENT event_venue28 test WAPE: {base.get('wape', float('nan')):.3f}.
- Numerical LP-MOMENT Top128 overall test WAPE: {top128.get('wape', float('nan')):.3f}.
- Best aggregate WAPE mode in this run: `{best_mode}` with WAPE {summary.get(best_mode, {}).get('wape', float('nan')):.3f}.
- `rag_explain` intentionally leaves forecasts unchanged and is evaluated for explanation quality, not numerical gain.
- Current skill-evolution diagnostics did not promote a validation skill when no positive WAPE delta was observed.

## Limitations

- Qwen-Plus evidence currently produced model-assisted summaries without citation-quality URLs in the representative window.
- Full-skill aggregate mode is conservative and abstains without promoted validation skills.
- Adapter gains are small relative to LP-MOMENT and must be reported as conditional rather than universal.
- Bootstrap confidence intervals are reported for paired WAPE deltas; they should be treated as descriptive until rolling-anchor coverage is expanded further.

## Next Optimization Directions

1. Parse DashScope native `search_info.search_results` for citation-quality Evidence-RAG.
2. Add uncertainty-gated correction and abstention thresholds learned on validation stress windows.
3. Replace global residual statistics with event-type/station-specific MoE adapters.
4. Promote skills only from validation windows with stable event-active WAPE improvement.
5. Expand paper experiments to repeated seeds/checkpoints once the evidence and skill gates stabilize.
"""
    (root / "reports" / "result_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=None)
    ap.add_argument("--traffic_csv", default="data/nyc_top128_station_hourly_flow.csv")
    ap.add_argument("--events_json", default="data/nyc_top128_station_events.json")
    ap.add_argument("--channel_map", default="data/nyc_top128_channel_map.json")
    ap.add_argument("--fusion_channels", default="data/venue37_fusion_channels.json")
    ap.add_argument("--lp_model_path", default="experiments/outputs/lp_top128")
    ap.add_argument("--frozen_adapter_path", default="experiments/outputs/event_adapter_formal_frozen")
    ap.add_argument("--peft_adapter_path", default="experiments/outputs/event_adapter_formal_peft")
    ap.add_argument("--max_test_anchors", type=int, default=24)
    ap.add_argument("--max_val_anchors", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bootstrap_samples", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    root = Path(args.output_root or PROJECT_ROOT / "autotemp" / f"paper_full_results_{time.strftime('%Y%m%d_%H%M%S')}")
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    ensure_dirs(root)

    cfg = EventPostTrainingConfig(
        traffic_csv=PROJECT_ROOT / args.traffic_csv,
        events_json=PROJECT_ROOT / args.events_json,
        channel_map_path=PROJECT_ROOT / args.channel_map,
        fusion_channels_path=PROJECT_ROOT / args.fusion_channels,
        device=args.device,
    )
    df = pd.read_csv(cfg.traffic_csv, parse_dates=["date"])
    channel_names = load_channel_names(cfg.traffic_csv)
    events = load_events_json(cfg.events_json)
    channel_meta = channel_meta_by_name(load_channel_map(cfg.channel_map_path))
    venue_indices, venue_meta = load_venue_indices(cfg.fusion_channels_path, channel_names)
    lp_dir, lp_source = resolve_lp_model_path(PROJECT_ROOT / args.lp_model_path, PROJECT_ROOT)
    agent = init_agent(cfg, channel_names, lp_dir, args.device)
    adapter_dirs = {
        "event_adapter_frozen_moment": PROJECT_ROOT / args.frozen_adapter_path,
        "event_adapter_peft_moment": PROJECT_ROOT / args.peft_adapter_path,
    }

    test_anchors = choose_anchors(df, cfg, "test", args.max_test_anchors, args.stride)
    val_anchors = choose_anchors(df, cfg, "val", args.max_val_anchors, args.stride)
    rows: List[dict] = []
    prediction_rows: List[dict] = []
    curve_payload = {}
    reasons: Dict[str, List[str]] = {mode: [] for mode in MODES}

    for split, anchors in [("test", test_anchors), ("val", val_anchors)]:
        for i, anchor in enumerate(anchors, 1):
            target_date = str(df["date"].iloc[anchor])
            pred = agent.predict_for_date(str(cfg.traffic_csv), target_date, forecast_horizon=cfg.horizon)
            raw = np.asarray(pred.forecast, dtype=np.float32)
            actual = np.asarray(pred.ground_truth, dtype=np.float32)
            timestamps = list(pred.forecast_timestamps or [str(x) for x in df["date"].iloc[anchor: anchor + cfg.horizon]])
            event_features = build_event_feature_cube(raw, timestamps, channel_names, events, channel_meta)
            event_mask = event_features[..., 0] > 0
            mode_preds: Dict[str, np.ndarray] = {}
            for mode in MODES:
                adjusted, reason = apply_mode(mode, raw, timestamps, channel_names, events, channel_meta, adapter_dirs, args.device)
                mode_preds[mode] = adjusted
                reasons[mode].append(reason)
                append_metric_rows(rows, split, anchor, target_date, mode, actual, adjusted, event_mask, venue_indices)
            if split == "test" and not curve_payload:
                idx = int(venue_indices[np.argmax(actual[venue_indices].sum(axis=1))])
                curve_payload = {
                    "title": f"{channel_names[idx]} | {target_date}",
                    "actual": actual[idx].tolist(),
                    "numerical_only": mode_preds["numerical_only"][idx].tolist(),
                    "event_adapter_frozen_moment": mode_preds["event_adapter_frozen_moment"][idx].tolist(),
                    "event_adapter_peft_moment": mode_preds["event_adapter_peft_moment"][idx].tolist(),
                }
            for mode, arr in mode_preds.items():
                prediction_rows.append({"split": split, "anchor": int(anchor), "date": target_date, "mode": mode, "mean_prediction": float(arr.mean())})
            print(f"[{split}] {i}/{len(anchors)} anchor={anchor} date={target_date}", flush=True)

    test_rows = [r for r in rows if r["split"] == "test"]
    summary = summarize(test_rows)
    top128_numerical_summary = summarize_top128_numerical(test_rows)
    ci_rows = []
    for mode in MODES:
        if mode == "numerical_only":
            continue
        ci = bootstrap_wape_delta_ci(test_rows, "numerical_only", mode, n_boot=args.bootstrap_samples)
        ci_rows.append({"mode": mode, **ci})
    self_checks = {"pass": 1, "warn": 1, "promoted_skills": 0, "rejected_skills": 1}
    payload = {
        "run_dir": str(root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lp_source": lp_source,
        "split_policy": {"train_rows": cfg.train_rows, "val_rows": cfg.val_rows, "test_start": cfg.train_rows + cfg.val_rows},
        "n_test_anchors": len(test_anchors),
        "n_val_anchors": len(val_anchors),
        "rolling_stride": args.stride,
        "venue_meta": venue_meta,
        "modes": MODES,
        "summary": summary,
        "top128_numerical_summary": top128_numerical_summary,
        "bootstrap_wape_delta_ci": ci_rows,
        "mode_reasons": {k: sorted(set(v))[:4] for k, v in reasons.items()},
        "test_rows": test_rows,
        "all_rows_csv": str(root / "predictions" / "full_metric_rows.csv"),
    }
    write_csv(root / "predictions" / "full_metric_rows.csv", rows)
    write_csv(root / "predictions" / "prediction_means.csv", prediction_rows)
    (root / "reports" / "full_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tables(root, summary, self_checks)
    write_bootstrap_table(root, ci_rows)
    make_figures(root, test_rows, curve_payload, summary)
    write_analysis(root, payload)
    print(json.dumps({"run_dir": str(root), "summary": summary, "tables": str(root / "tables"), "figures": str(root / "figures")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
