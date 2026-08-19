#!/usr/bin/env python3
"""Event-stratified result analysis for EAF-MAS full-test metrics.

The script reads existing full-stack result rows and stratifies matched
PT-MOMENT vs frozen event-adapter metrics by forecast-time event context.  It
does not train models, call LLMs, update skills, or write into the provenance
run root.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_MODES = {"pt_moment", "event_adapter_frozen"}


@dataclass(frozen=True)
class StratifiedRunConfig:
    bootstrap_samples: int = 2000
    min_group_n: int = 20
    random_seed: int = 13
    horizon: int = 192


def ensure_dirs(root: Path) -> None:
    for sub in ("tables", "figures", "reports", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_events(path: str | Path) -> List[dict]:
    payload = read_json(path)
    rows = payload.get("events", payload) if isinstance(payload, dict) else payload
    return [dict(row) for row in rows or []]


def find_anchor_plan(run_root: str | Path, explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"anchor plan not found: {path}")
    root = Path(run_root)
    for rel in ("reports/full_anchor_plan.csv", "full_anchor_plan.csv", "autoskill_skillbench_full/reports/full_anchor_plan.csv"):
        path = root / rel
        if path.is_file():
            return path
    matches = sorted(root.rglob("*anchor*plan*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"could not locate full_anchor_plan.csv under {root}")


def venue_group_from_event(event: Mapping[str, Any]) -> str:
    text = " ".join(str(event.get(k, "") or "") for k in ("venue_name", "location", "title", "content")).lower()
    if "times square" in text or "father duffy" in text or "broadway theater" in text:
        return "Times Square / Broadway"
    if "madison square garden" in text or "penn station" in text or "herald sq" in text:
        return "MSG / Penn / Herald Sq"
    if "central park" in text:
        return "Central Park"
    if "yankee" in text or "161 st" in text:
        return "Yankee Stadium"
    if "barclays" in text or "atlantic av" in text:
        return "Barclays Center"
    if "lincoln center" in text:
        return "Lincoln Center"
    if "rockefeller" in text:
        return "Rockefeller Center"
    if "hudson yards" in text:
        return "Hudson Yards"
    if "forest hills" in text or "us open" in text or "flushing" in text:
        return "Queens Events"
    venue = str(event.get("venue_name") or event.get("location") or "").split("|")[0].split(":")[0].strip()
    if not venue:
        return "Other / Unknown"
    return venue[:48]


def station_rank_group_from_event(event: Mapping[str, Any]) -> str:
    try:
        rank = float(event.get("station_rank") or 999)
    except Exception:
        rank = 999.0
    if rank <= 32:
        return "top32"
    if rank <= 64:
        return "top64"
    if rank <= 128:
        return "top128"
    return "unknown"


def _event_time(event: Mapping[str, Any]) -> pd.Timestamp | None:
    raw = event.get("event_time") or event.get("start_time") or event.get("date")
    if not raw:
        return None
    try:
        ts = pd.Timestamp(raw)
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def prepare_event_index(events_json: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    from experiments.run_ccfa_fullstack_final import is_qwenplus_live_candidate, normalize_event_for_fullstack

    rows = []
    for event in load_events(events_json):
        normalized = normalize_event_for_fullstack(event)
        ts = _event_time(normalized)
        if ts is None:
            continue
        tier = str(normalized.get("impact_tier") or "C").upper()
        event_type = str(normalized.get("event_type") or "unknown").strip() or "unknown"
        relevance = float(normalized.get("relevance_score") or 0.0)
        is_high_value = bool(is_qwenplus_live_candidate(normalized) or tier in {"A", "B"})
        rows.append(
            {
                "event_time": ts,
                "title": normalized.get("title") or "",
                "event_type": event_type,
                "impact_tier": tier,
                "venue_group": venue_group_from_event(normalized),
                "station_rank_group": station_rank_group_from_event(normalized),
                "relevance_score": relevance,
                "is_high_value_event": is_high_value,
                "is_A_or_B_event": tier in {"A", "B"},
                "peak_hour_event": ts.hour in {7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22},
                "station_rank": normalized.get("station_rank"),
            }
        )
    if not rows:
        empty = pd.DataFrame(columns=["event_time"])
        return empty, np.array([], dtype="datetime64[ns]")
    df = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    return df, df["event_time"].to_numpy(dtype="datetime64[ns]")


def dominant_event(events: pd.DataFrame) -> Mapping[str, Any] | None:
    if events.empty:
        return None
    tier_priority = {"A": 4, "B": 3, "C": 2, "D": 1}
    ranked = events.copy()
    ranked["_tier_priority"] = ranked["impact_tier"].map(tier_priority).fillna(0)
    ranked["_rank_key"] = list(
        zip(
            ranked["_tier_priority"],
            ranked["is_high_value_event"].astype(int),
            ranked["relevance_score"].fillna(0.0),
        )
    )
    return ranked.sort_values(["_tier_priority", "is_high_value_event", "relevance_score"], ascending=False).iloc[0].to_dict()


def build_anchor_event_strata(
    anchor_plan: pd.DataFrame,
    events_df: pd.DataFrame,
    event_times: np.ndarray,
    horizon: int = 192,
) -> pd.DataFrame:
    rows = []
    test_anchors = anchor_plan[anchor_plan["split"].eq("test")].copy()
    test_anchors["date"] = pd.to_datetime(test_anchors["date"])
    if "horizon_end_time" in test_anchors.columns:
        test_anchors["horizon_end_time"] = pd.to_datetime(test_anchors["horizon_end_time"])
    for anchor in test_anchors.itertuples(index=False):
        start = pd.Timestamp(anchor.date)
        anchor_horizon = int(getattr(anchor, "horizon", horizon) or horizon)
        end_exclusive = start + pd.Timedelta(hours=anchor_horizon)
        if len(event_times):
            left = int(np.searchsorted(event_times, np.datetime64(start), side="left"))
            right = int(np.searchsorted(event_times, np.datetime64(end_exclusive), side="left"))
            window_events = events_df.iloc[left:right]
        else:
            window_events = events_df.iloc[0:0]
        dom = dominant_event(window_events)
        if dom is None:
            rows.append(
                {
                    "split": "test",
                    "anchor": int(anchor.anchor),
                    "date": str(start),
                    "horizon": anchor_horizon,
                    "dominant_event_type": "no_event",
                    "dominant_impact_tier": "none",
                    "dominant_venue_group": "no_event",
                    "station_rank_group": "none",
                    "high_value_event_count": 0,
                    "event_count": 0,
                    "has_A_or_B_event": False,
                    "peak_hour_overlap": False,
                }
            )
        else:
            rows.append(
                {
                    "split": "test",
                    "anchor": int(anchor.anchor),
                    "date": str(start),
                    "horizon": anchor_horizon,
                    "dominant_event_type": str(dom.get("event_type") or "unknown"),
                    "dominant_impact_tier": str(dom.get("impact_tier") or "unknown"),
                    "dominant_venue_group": str(dom.get("venue_group") or "Other / Unknown"),
                    "station_rank_group": str(dom.get("station_rank_group") or "unknown"),
                    "high_value_event_count": int(window_events["is_high_value_event"].sum()),
                    "event_count": int(len(window_events)),
                    "has_A_or_B_event": bool(window_events["is_A_or_B_event"].any()),
                    "peak_hour_overlap": bool(window_events["peak_hour_event"].any()),
                }
            )
    return pd.DataFrame(rows)


def merge_anchor_metrics(full_metric_rows: pd.DataFrame, anchor_strata: pd.DataFrame) -> pd.DataFrame:
    rows = full_metric_rows[full_metric_rows["split"].eq("test")].copy()
    modes = set(rows["mode"].unique())
    missing = REQUIRED_MODES - modes
    if missing:
        raise ValueError(f"event stratified analysis requires modes {sorted(REQUIRED_MODES)}, missing {sorted(missing)}")
    raw = rows[rows["mode"].eq("pt_moment")].set_index("anchor")
    adapter = rows[rows["mode"].eq("event_adapter_frozen")].set_index("anchor")
    common = sorted(set(raw.index) & set(adapter.index))
    if not common:
        raise ValueError("no matched anchors between pt_moment and event_adapter_frozen")
    merged = anchor_strata.set_index("anchor").loc[common].reset_index()
    merged["raw_event_active_wape"] = raw.loc[common, "event_active_wape"].to_numpy(dtype=float)
    merged["adapter_event_active_wape"] = adapter.loc[common, "event_active_wape"].to_numpy(dtype=float)
    merged["event_active_n"] = raw.loc[common, "event_active_n"].to_numpy(dtype=float)
    merged["raw_non_event_wape"] = raw.loc[common, "non_event_wape"].to_numpy(dtype=float)
    merged["adapter_non_event_wape"] = adapter.loc[common, "non_event_wape"].to_numpy(dtype=float)
    merged["non_event_n"] = raw.loc[common, "non_event_n"].to_numpy(dtype=float)
    merged["event_active_gain"] = merged["raw_event_active_wape"] - merged["adapter_event_active_wape"]
    merged.loc[merged["event_active_n"] <= 0, "event_active_gain"] = np.nan
    merged["non_event_degradation"] = merged["adapter_non_event_wape"] - merged["raw_non_event_wape"]
    return merged


def bootstrap_ci(values: Sequence[float], samples: int = 2000, seed: int = 13) -> tuple[float, float]:
    arr = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(0, len(arr), len(arr))
        means.append(float(np.mean(arr[idx])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mean(series: pd.Series) -> float:
    arr = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def summarize_group(df: pd.DataFrame, group_col: str, min_group_n: int, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False, sort=False):
        gain = group["event_active_gain"].replace([np.inf, -np.inf], np.nan).dropna()
        low, high = bootstrap_ci(gain, samples=bootstrap_samples, seed=seed)
        rows.append(
            {
                group_col: str(value),
                "n_anchors": int(group["anchor"].nunique()),
                "event_active_n": int(group["event_active_n"].sum()),
                "non_event_n": int(group["non_event_n"].sum()),
                "raw_event_active_wape": _mean(group["raw_event_active_wape"]),
                "adapter_event_active_wape": _mean(group["adapter_event_active_wape"]),
                "event_active_gain": _mean(gain),
                "gain_ci95_low": low,
                "gain_ci95_high": high,
                "positive_gain_rate": float((gain > 0).mean()) if len(gain) else float("nan"),
                "harmful_anchor_rate": float((gain < 0).mean()) if len(gain) else float("nan"),
                "raw_non_event_wape": _mean(group["raw_non_event_wape"]),
                "adapter_non_event_wape": _mean(group["adapter_non_event_wape"]),
                "non_event_degradation": _mean(group["non_event_degradation"]),
                "event_count_mean": _mean(group["event_count"]),
                "high_value_event_count_mean": _mean(group["high_value_event_count"]),
                "small_n_flag": bool(group["anchor"].nunique() < int(min_group_n)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["small_n_flag", "n_anchors", group_col], ascending=[True, False, True]).reset_index(drop=True)


def write_latex_table(df: pd.DataFrame, path: Path, group_col: str) -> None:
    cols = [
        group_col,
        "n_anchors",
        "raw_event_active_wape",
        "adapter_event_active_wape",
        "event_active_gain",
        "gain_ci95_low",
        "gain_ci95_high",
        "positive_gain_rate",
        "harmful_anchor_rate",
        "small_n_flag",
    ]
    present = [c for c in cols if c in df.columns]
    latex = df[present].to_latex(index=False, float_format=lambda x: f"{x:.4f}", escape=True)
    path.write_text(latex, encoding="utf-8")


def write_group_outputs(root: Path, merged: pd.DataFrame, cfg: StratifiedRunConfig) -> Dict[str, pd.DataFrame]:
    specs = {
        "event_type": "dominant_event_type",
        "impact_tier": "dominant_impact_tier",
        "venue_group": "dominant_venue_group",
        "station_rank_group": "station_rank_group",
    }
    outputs = {}
    for name, col in specs.items():
        summary = summarize_group(merged, col, cfg.min_group_n, cfg.bootstrap_samples, cfg.random_seed)
        outputs[name] = summary
        csv_path = root / "tables" / f"{name}_gain_summary.csv"
        tex_path = root / "tables" / f"{name}_gain_summary.tex"
        summary.to_csv(csv_path, index=False)
        write_latex_table(summary, tex_path, col)
    return outputs


def _clean_label(value: str, max_len: int = 34) -> str:
    value = str(value)
    return value if len(value) <= max_len else value[: max_len - 1] + "."


def write_figures(root: Path, merged: pd.DataFrame, summaries: Mapping[str, pd.DataFrame], min_group_n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Event type x tier heatmap.
    heat = (
        merged.groupby(["dominant_event_type", "dominant_impact_tier"], dropna=False)
        .agg(event_active_gain=("event_active_gain", "mean"), n_anchors=("anchor", "nunique"))
        .reset_index()
    )
    top_types = (
        heat.groupby("dominant_event_type")["n_anchors"].sum().sort_values(ascending=False).head(12).index.tolist()
    )
    heat = heat[heat["dominant_event_type"].isin(top_types)]
    types = top_types or ["no_data"]
    tiers = [t for t in ["A", "B", "C", "D", "none", "unknown"] if t in set(heat["dominant_impact_tier"])]
    tiers = tiers or sorted(heat["dominant_impact_tier"].unique().tolist()) or ["none"]
    matrix = np.full((len(types), len(tiers)), np.nan)
    counts = np.zeros((len(types), len(tiers)), dtype=int)
    for _, row in heat.iterrows():
        i = types.index(row["dominant_event_type"])
        j = tiers.index(row["dominant_impact_tier"])
        matrix[i, j] = float(row["event_active_gain"])
        counts[i, j] = int(row["n_anchors"])
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.34 * len(types) + 1.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([_clean_label(t, 42) for t in types])
    ax.set_title("Event-active WAPE gain by event type and impact tier")
    ax.set_xlabel("Impact tier")
    ax.set_ylabel("Dominant event type")
    for i in range(len(types)):
        for j in range(len(tiers)):
            if counts[i, j] > 0:
                text = f"{matrix[i, j]:.2f}\nn={counts[i, j]}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("WAPE gain vs PT-MOMENT")
    fig.tight_layout()
    fig.savefig(root / "figures" / "event_type_gain_heatmap.png", dpi=220)
    fig.savefig(root / "figures" / "event_type_gain_heatmap.pdf")
    plt.close(fig)

    def forest(summary: pd.DataFrame, group_col: str, path_stem: str, title: str, max_rows: int = 14) -> None:
        data = summary.copy()
        if group_col == "dominant_venue_group":
            data = data.head(max_rows)
        data = data.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(data))
        mean = data["event_active_gain"].astype(float).to_numpy()
        low = data["gain_ci95_low"].astype(float).to_numpy()
        high = data["gain_ci95_high"].astype(float).to_numpy()
        xerr_low = np.where(np.isfinite(low), mean - low, 0.0)
        xerr_high = np.where(np.isfinite(high), high - mean, 0.0)
        colors = ["#b04a3a" if bool(flag) else "#2f6f9f" for flag in data["small_n_flag"]]
        fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.34 * len(data) + 1.2)))
        ax.axvline(0.0, color="#333333", linewidth=1.0)
        ax.errorbar(mean, y, xerr=[xerr_low, xerr_high], fmt="none", ecolor="#999999", capsize=3, zorder=1)
        ax.scatter(mean, y, c=colors, s=40, zorder=2)
        labels = [f"{_clean_label(v, 36)} (n={n})" for v, n in zip(data[group_col], data["n_anchors"])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Event-active WAPE gain vs PT-MOMENT")
        ax.set_title(title)
        ax.text(0.01, 0.02, f"Red markers: n < {min_group_n}; interpret cautiously.", transform=ax.transAxes, fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "figures" / f"{path_stem}.png", dpi=220)
        fig.savefig(root / "figures" / f"{path_stem}.pdf")
        plt.close(fig)

    forest(summaries["impact_tier"], "dominant_impact_tier", "impact_tier_gain_forest", "Gain by dominant impact tier")
    forest(summaries["venue_group"], "dominant_venue_group", "venue_group_gain_forest", "Gain by dominant venue group")

    scatter = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["event_active_gain"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if scatter.empty:
        ax.text(0.5, 0.5, "No finite event-active gain values", ha="center", va="center")
    else:
        colors = np.where(scatter["has_A_or_B_event"].astype(bool), "#c65f24", "#386fa4")
        ax.scatter(scatter["event_count"], scatter["event_active_gain"], c=colors, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel("Events inside 192h forecast horizon")
    ax.set_ylabel("Event-active WAPE gain vs PT-MOMENT")
    ax.set_title("Event density and event-active gain")
    ax.text(0.01, 0.02, "Orange: anchor contains A/B event; blue: no A/B event.", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "figures" / "event_count_vs_gain_scatter.png", dpi=220)
    fig.savefig(root / "figures" / "event_count_vs_gain_scatter.pdf")
    plt.close(fig)


def write_interpretation(root: Path, summaries: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Event-Stratified Results Interpretation",
        "",
        "This analysis stratifies matched PT-MOMENT and frozen event-adapter results by forecast-time event context. It is a robustness diagnostic, not a new training run.",
        "",
        "## Main Paper Candidates",
        "- Use `impact_tier_gain_forest` to discuss A/B-tier event robustness.",
        "- Use `event_type_gain_heatmap` to show which event categories have enough support and which remain small-n.",
        "- Use `event_count_vs_gain_scatter` to show whether gains concentrate in dense or high-value event windows.",
        "",
        "## Conservative Findings",
    ]
    tier = summaries.get("impact_tier", pd.DataFrame())
    if not tier.empty:
        if tier["dominant_impact_tier"].nunique() <= 1:
            only = str(tier["dominant_impact_tier"].iloc[0])
            lines.append(
                f"- Impact-tier stratification has only one dominant tier (`{only}`) in the current full test split. "
                "Use the tier figure as a coverage/limitation diagnostic rather than evidence of cross-tier robustness."
            )
        for label in ["A", "B", "C", "D", "none"]:
            match = tier[tier["dominant_impact_tier"].astype(str).eq(label)]
            if not match.empty:
                row = match.iloc[0]
                lines.append(
                    f"- Tier `{label}`: n={int(row['n_anchors'])}, gain={float(row['event_active_gain']):.4f}, "
                    f"CI=[{float(row['gain_ci95_low']):.4f}, {float(row['gain_ci95_high']):.4f}], small_n={bool(row['small_n_flag'])}."
                )
    negatives = []
    for name, df in summaries.items():
        if df.empty or "event_active_gain" not in df:
            continue
        group_col = [c for c in df.columns if c.startswith("dominant_") or c == "station_rank_group"][0]
        bad = df[(df["event_active_gain"] < 0) & (~df["small_n_flag"])]
        for _, row in bad.iterrows():
            negatives.append(f"{name}:{row[group_col]} (n={int(row['n_anchors'])}, gain={float(row['event_active_gain']):.4f})")
    if negatives:
        lines.append("- Non-small subgroups with negative gain: " + "; ".join(negatives[:12]) + ".")
    else:
        lines.append("- No non-small subgroup shows negative mean event-active gain under the current `min_group_n` setting.")
    venue = summaries.get("venue_group", pd.DataFrame())
    if not venue.empty:
        main_venues = venue[~venue["small_n_flag"]].head(5)
        if not main_venues.empty:
            venue_bits = [
                f"{row['dominant_venue_group']} (n={int(row['n_anchors'])}, gain={float(row['event_active_gain']):.4f})"
                for _, row in main_venues.iterrows()
            ]
            lines.append("- Non-small venue groups with reportable support: " + "; ".join(venue_bits) + ".")
    small_n_total = sum(int(df["small_n_flag"].sum()) for df in summaries.values() if not df.empty)
    lines.extend(
        [
            f"- Small-n subgroup entries across tables: {small_n_total}; these should be treated as appendix diagnostics.",
            "",
            "## Claims To Avoid",
            "- Do not claim every event type or venue improves unless the corresponding subgroup has adequate sample size and positive gain.",
            "- Do not use this analysis to claim RAG, AutoSkill, or LLM agents improve numerical forecasting.",
            f"- Cell-level analysis executed: `{manifest.get('cell_level_executed')}`. If false, report only anchor-level robustness.",
        ]
    )
    (root / "reports" / "event_stratified_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_cell_level_status(root: Path, prediction_cache_npz: str | Path | None) -> dict:
    if not prediction_cache_npz:
        status = {"cell_level_executed": False, "reason": "prediction_cache_npz_not_provided"}
    else:
        path = Path(prediction_cache_npz)
        if not path.is_file():
            status = {"cell_level_executed": False, "reason": "prediction_cache_npz_not_found", "path": str(path)}
        else:
            status = {
                "cell_level_executed": False,
                "reason": "cache_exists_but_lacks_required_event_cell_metadata_for_stratified_cell_analysis",
                "path": str(path),
            }
    (root / "reports" / "cell_level_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def run_event_stratified_analysis(
    output_root: str | Path,
    full_metric_rows: str | Path,
    anchor_plan_csv: str | Path,
    events_json: str | Path,
    venue_channels_json: str | Path | None = None,
    channel_map_json: str | Path | None = None,
    prediction_cache_npz: str | Path | None = None,
    run_root: str | Path | None = None,
    bootstrap_samples: int = 2000,
    min_group_n: int = 20,
    random_seed: int = 13,
    horizon: int = 192,
) -> dict:
    root = Path(output_root)
    ensure_dirs(root)
    cfg = StratifiedRunConfig(bootstrap_samples=bootstrap_samples, min_group_n=min_group_n, random_seed=random_seed, horizon=horizon)
    metrics = pd.read_csv(full_metric_rows)
    anchors = pd.read_csv(anchor_plan_csv)
    events_df, event_times = prepare_event_index(events_json)
    anchor_strata = build_anchor_event_strata(anchors, events_df, event_times, horizon=horizon)
    merged = merge_anchor_metrics(metrics, anchor_strata)
    anchor_strata["small_n_flag"] = False
    anchor_strata.to_csv(root / "tables" / "anchor_event_strata.csv", index=False)
    merged.to_csv(root / "tables" / "anchor_metric_strata.csv", index=False)
    summaries = write_group_outputs(root, merged, cfg)
    write_figures(root, merged, summaries, min_group_n=min_group_n)
    cell_status = maybe_cell_level_status(root, prediction_cache_npz)
    manifest = {
        "status": "completed",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "run_root": str(run_root) if run_root else None,
        "full_metric_rows": str(full_metric_rows),
        "anchor_plan_csv": str(anchor_plan_csv),
        "events_json": str(events_json),
        "venue_channels_json": str(venue_channels_json) if venue_channels_json else None,
        "channel_map_json": str(channel_map_json) if channel_map_json else None,
        "prediction_cache_npz": str(prediction_cache_npz) if prediction_cache_npz else None,
        "anchor_count": int(merged["anchor"].nunique()),
        "event_count": int(len(events_df)),
        "horizon": int(horizon),
        "bootstrap_samples": int(bootstrap_samples),
        "min_group_n": int(min_group_n),
        "random_seed": int(random_seed),
        "cell_level_executed": bool(cell_status.get("cell_level_executed")),
        "cell_level_status": cell_status,
        "claim_boundary": "Anchor-level robustness analysis only; no model training, LLM calls, or skill updates.",
        "outputs": {
            "anchor_event_strata": "tables/anchor_event_strata.csv",
            "anchor_metric_strata": "tables/anchor_metric_strata.csv",
            "event_type": "tables/event_type_gain_summary.csv",
            "impact_tier": "tables/impact_tier_gain_summary.csv",
            "venue_group": "tables/venue_group_gain_summary.csv",
            "station_rank_group": "tables/station_rank_group_gain_summary.csv",
        },
    }
    (root / "reports" / "event_stratified_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_interpretation(root, summaries, manifest)
    return {"manifest": manifest, "summaries": summaries, "anchor_strata": anchor_strata, "merged": merged}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Analyze event-stratified full-test results.")
    parser.add_argument("--run_root", default=str(PROJECT_ROOT / "autotemp" / "ccfa_fullstack_summary_only_20260619_123041"))
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "autotemp" / f"revision_trc_tits_ccfa_{timestamp}" / "event_stratified_analysis"))
    parser.add_argument("--full_metric_rows", default=None)
    parser.add_argument("--anchor_plan_csv", default=None)
    parser.add_argument("--events_json", default=str(PROJECT_ROOT / "data" / "nyc_top128_station_events.json"))
    parser.add_argument("--venue_channels_json", default=str(PROJECT_ROOT / "data" / "venue37_fusion_channels.json"))
    parser.add_argument("--channel_map_json", default=str(PROJECT_ROOT / "data" / "nyc_top128_channel_map.json"))
    parser.add_argument("--prediction_cache_npz", default=None)
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--min_group_n", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_root = Path(args.run_root)
    full_metric_rows = Path(args.full_metric_rows) if args.full_metric_rows else run_root / "predictions" / "full_metric_rows.csv"
    anchor_plan_csv = find_anchor_plan(run_root, args.anchor_plan_csv)
    prediction_cache = args.prediction_cache_npz
    if prediction_cache is None:
        candidate = run_root / "predictions" / "full_test_prediction_cache.npz"
        prediction_cache = str(candidate) if candidate.is_file() else None
    run_event_stratified_analysis(
        output_root=args.output_root,
        full_metric_rows=full_metric_rows,
        anchor_plan_csv=anchor_plan_csv,
        events_json=args.events_json,
        venue_channels_json=args.venue_channels_json,
        channel_map_json=args.channel_map_json,
        prediction_cache_npz=prediction_cache,
        run_root=run_root,
        bootstrap_samples=args.bootstrap_samples,
        min_group_n=args.min_group_n,
        random_seed=args.random_seed,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
