#!/usr/bin/env python3
"""Dataset-level station/channel gate heatmap visualizations.

This script consumes the gate trace produced by
``run_gate_coverage_diagnostics.py``. It does not run forecasts, call LLMs,
update Skill memory, or change forecast arrays.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.paper_workflow import display_station_name  # noqa: E402


DEFAULT_GATE_TRACE = (
    "autotemp/revision_trc_tits_ccfa_20260624_155014/"
    "gate_coverage_diagnostics/predictions/gate_trace_rows.parquet"
)
DEFAULT_CHANNEL_MAP = "data/nyc_top128_channel_map.json"
ACTION_ORDER = ["corrected", "allowed_no_numeric_effect", "abstained", "unchanged_or_not_candidate"]
ACTION_COLORS = {
    "corrected": "#e15759",
    "allowed_no_numeric_effect": "#f28e2b",
    "abstained": "#4e79a7",
    "unchanged_or_not_candidate": "#bab0ab",
}
TRACE_COLUMNS = [
    "station_or_channel",
    "has_structured_event",
    "event_type",
    "impact_tier",
    "venue_name",
    "controller_allowed",
    "controller_abstained",
    "correction_applied",
    "bound_clipped",
    "reason_code",
]


def ensure_dirs(root: Path) -> None:
    for sub in ("tables", "figures", "reports", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def read_gate_trace(path: str | Path) -> pd.DataFrame:
    trace_path = Path(path)
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"gate trace not found: {trace_path}. Run experiments/run_gate_coverage_diagnostics.py first."
        )
    if trace_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(trace_path, columns=TRACE_COLUMNS)
    elif trace_path.suffix.lower() == ".csv":
        df = pd.read_csv(trace_path, usecols=lambda c: c in set(TRACE_COLUMNS))
    else:
        raise ValueError(f"unsupported gate trace format: {trace_path.suffix}")
    missing = [col for col in TRACE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"gate trace missing required columns: {missing}")
    for col in ["has_structured_event", "controller_allowed", "controller_abstained", "correction_applied", "bound_clipped"]:
        if df[col].dtype != bool:
            df[col] = df[col].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["event_type", "impact_tier", "venue_name", "reason_code", "station_or_channel"]:
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def load_channel_metadata(path: str | Path | None) -> Dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    rows = payload.get("channels", payload) if isinstance(payload, dict) else payload
    meta: Dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        channel = str(row.get("channel_name") or row.get("station_or_channel") or "").strip()
        if channel:
            meta[channel] = dict(row)
    return meta


def station_rank_group(rank: Any) -> str:
    try:
        value = float(rank)
    except Exception:
        return "unknown"
    if value <= 32:
        return "top32"
    if value <= 64:
        return "top64"
    if value <= 128:
        return "top128"
    return "unknown"


def add_gate_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    corrected = out["correction_applied"].astype(bool)
    allowed_no_effect = out["controller_allowed"].astype(bool) & ~corrected
    abstained = out["controller_abstained"].astype(bool) & ~corrected & ~allowed_no_effect
    unchanged = ~(corrected | allowed_no_effect | abstained)
    out["gate_action"] = "unchanged_or_not_candidate"
    out.loc[abstained, "gate_action"] = "abstained"
    out.loc[allowed_no_effect, "gate_action"] = "allowed_no_numeric_effect"
    out.loc[corrected, "gate_action"] = "corrected"
    return out


def attach_station_metadata(summary: pd.DataFrame, channel_meta: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    out = summary.copy()
    ranks = []
    labels = []
    groups = []
    for station in out["station_or_channel"].astype(str):
        meta = dict(channel_meta.get(station, {}))
        rank = meta.get("rank")
        ranks.append(rank if rank is not None else np.nan)
        labels.append(display_station_name(station, meta, max_len=42))
        groups.append(station_rank_group(rank))
    out["station_rank"] = ranks
    out["station_rank_group"] = groups
    out["station_label"] = labels
    return out


def build_station_summary(df: pd.DataFrame, channel_meta: Mapping[str, Mapping[str, Any]], min_station_cells: int) -> pd.DataFrame:
    grouped = (
        df.groupby("station_or_channel", dropna=False)
        .agg(
            all_cells=("station_or_channel", "size"),
            event_candidate_cells=("has_structured_event", "sum"),
            controller_allowed_cells=("controller_allowed", "sum"),
            corrected_cells=("correction_applied", "sum"),
            abstained_cells=("controller_abstained", "sum"),
            clipped_cells=("bound_clipped", "sum"),
        )
        .reset_index()
    )
    action_counts = (
        df.groupby(["station_or_channel", "gate_action"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ACTION_ORDER, fill_value=0)
        .reset_index()
    )
    grouped = grouped.merge(action_counts, on="station_or_channel", how="left")
    for col in ACTION_ORDER:
        grouped[col] = grouped[col].fillna(0).astype(int)
    grouped["allowed_no_numeric_effect_cells"] = grouped["allowed_no_numeric_effect"]
    grouped["unchanged_or_not_candidate_cells"] = grouped["unchanged_or_not_candidate"]
    grouped["controller_allowed_rate"] = grouped["controller_allowed_cells"] / grouped["all_cells"].clip(lower=1)
    grouped["correction_applied_rate"] = grouped["corrected_cells"] / grouped["all_cells"].clip(lower=1)
    grouped["abstained_rate"] = grouped["abstained_cells"] / grouped["all_cells"].clip(lower=1)
    grouped["clipped_rate_of_corrected"] = grouped["clipped_cells"] / grouped["corrected_cells"].clip(lower=1)
    grouped["event_candidate_rate"] = grouped["event_candidate_cells"] / grouped["all_cells"].clip(lower=1)
    grouped["small_n_flag"] = (grouped["all_cells"] < int(min_station_cells)) | (
        grouped["event_candidate_cells"] < int(min_station_cells)
    )
    grouped = attach_station_metadata(grouped, channel_meta)
    ordered = [
        "station_or_channel",
        "station_label",
        "station_rank",
        "station_rank_group",
        "all_cells",
        "event_candidate_cells",
        "controller_allowed_cells",
        "corrected_cells",
        "abstained_cells",
        "clipped_cells",
        "allowed_no_numeric_effect_cells",
        "unchanged_or_not_candidate_cells",
        "controller_allowed_rate",
        "correction_applied_rate",
        "abstained_rate",
        "clipped_rate_of_corrected",
        "event_candidate_rate",
        "small_n_flag",
    ]
    return grouped[ordered].sort_values(["corrected_cells", "controller_allowed_cells"], ascending=False).reset_index(drop=True)


def build_action_by_station(df: pd.DataFrame, station_summary: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["station_or_channel", "gate_action"], dropna=False)
        .size()
        .reset_index(name="cells")
    )
    stations = station_summary[["station_or_channel", "station_label", "all_cells", "small_n_flag"]]
    full_index = pd.MultiIndex.from_product(
        [station_summary["station_or_channel"].astype(str).tolist(), ACTION_ORDER],
        names=["station_or_channel", "gate_action"],
    )
    counts = counts.set_index(["station_or_channel", "gate_action"]).reindex(full_index, fill_value=0).reset_index()
    counts = counts.merge(stations, on="station_or_channel", how="left")
    counts["ratio_of_station"] = counts["cells"] / counts["all_cells"].clip(lower=1)
    return counts[["station_or_channel", "station_label", "gate_action", "cells", "ratio_of_station", "small_n_flag"]]


def build_action_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["impact_tier", "gate_action"], dropna=False).size().reset_index(name="cells")
    tiers = sorted(df["impact_tier"].dropna().astype(str).unique().tolist())
    full_index = pd.MultiIndex.from_product([tiers, ACTION_ORDER], names=["impact_tier", "gate_action"])
    counts = counts.set_index(["impact_tier", "gate_action"]).reindex(full_index, fill_value=0).reset_index()
    tier_totals = counts.groupby("impact_tier")["cells"].transform("sum").clip(lower=1)
    counts["ratio_of_tier"] = counts["cells"] / tier_totals
    return counts


def build_heatmap_table(df: pd.DataFrame, station_summary: pd.DataFrame, heatmap_column: str, top_k: int) -> pd.DataFrame:
    if heatmap_column not in {"event_type", "impact_tier", "venue_name"}:
        raise ValueError("--heatmap_column must be one of event_type, impact_tier, venue_name")
    top_stations = station_summary.head(int(top_k))["station_or_channel"].astype(str).tolist()
    event_df = df[df["station_or_channel"].isin(top_stations)].copy()
    non_none = event_df[~event_df[heatmap_column].astype(str).str.lower().isin(["none", "no_event", "unknown"])]
    col_counts = non_none[heatmap_column].value_counts().head(12)
    columns = col_counts.index.astype(str).tolist()
    if not columns:
        columns = event_df[heatmap_column].value_counts().head(12).index.astype(str).tolist()
    sub = event_df[event_df[heatmap_column].isin(columns)].copy()
    rows = []
    label_map = station_summary.set_index("station_or_channel")["station_label"].to_dict()
    for (station, col_value), group in sub.groupby(["station_or_channel", heatmap_column], dropna=False):
        cells = int(len(group))
        corrected = int(group["correction_applied"].sum())
        allowed = int(group["controller_allowed"].sum())
        abstained = int(group["controller_abstained"].sum())
        rows.append(
            {
                "station_or_channel": station,
                "station_label": label_map.get(station, station),
                heatmap_column: str(col_value),
                "cells": cells,
                "corrected_cells": corrected,
                "controller_allowed_cells": allowed,
                "abstained_cells": abstained,
                "correction_applied_rate": corrected / max(1, cells),
                "controller_allowed_rate": allowed / max(1, cells),
                "abstained_rate": abstained / max(1, cells),
            }
        )
    return pd.DataFrame(rows)


def build_rank_summary(station_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        station_summary.groupby("station_rank_group", dropna=False)
        .agg(
            station_count=("station_or_channel", "nunique"),
            all_cells=("all_cells", "sum"),
            event_candidate_cells=("event_candidate_cells", "sum"),
            controller_allowed_cells=("controller_allowed_cells", "sum"),
            corrected_cells=("corrected_cells", "sum"),
            abstained_cells=("abstained_cells", "sum"),
            clipped_cells=("clipped_cells", "sum"),
        )
        .reset_index()
    )
    grouped["correction_sparsity"] = grouped["corrected_cells"] / grouped["all_cells"].clip(lower=1)
    grouped["allowed_rate"] = grouped["controller_allowed_cells"] / grouped["all_cells"].clip(lower=1)
    grouped["abstained_rate"] = grouped["abstained_cells"] / grouped["all_cells"].clip(lower=1)
    order = {"top32": 0, "top64": 1, "top128": 2, "unknown": 3}
    grouped["_order"] = grouped["station_rank_group"].map(order).fillna(99)
    return grouped.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def write_tables(root: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    tables["station_summary"].to_csv(table_dir / "station_channel_gate_summary.csv", index=False)
    tables["action_by_station"].to_csv(table_dir / "gate_action_by_station.csv", index=False)
    tables["action_by_tier"].to_csv(table_dir / "gate_action_by_tier.csv", index=False)
    tables["heatmap"].to_csv(table_dir / "station_channel_gate_heatmap_cells.csv", index=False)
    tables["rank_summary"].to_csv(table_dir / "correction_sparsity_by_station_rank.csv", index=False)
    tables["station_summary"].to_latex(
        table_dir / "station_channel_gate_summary.tex",
        index=False,
        float_format=lambda x: f"{x:.4f}",
        escape=True,
    )


def _shorten_labels(labels: Iterable[str], max_len: int = 34) -> list[str]:
    out = []
    for label in labels:
        text = str(label)
        out.append(text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…")
    return out


def write_figures(root: Path, tables: Mapping[str, pd.DataFrame], heatmap_column: str, top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    heat = tables["heatmap"].copy()
    station_summary = tables["station_summary"].copy()
    if heat.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No gate heatmap cells available", ha="center", va="center")
        ax.axis("off")
    else:
        station_order = station_summary.head(int(top_k))["station_or_channel"].astype(str).tolist()
        column_order = heat.groupby(heatmap_column)["cells"].sum().sort_values(ascending=False).index.astype(str).tolist()
        value_matrix = (
            heat.pivot_table(
                index="station_or_channel",
                columns=heatmap_column,
                values="correction_applied_rate",
                aggfunc="mean",
                fill_value=0.0,
            )
            .reindex(index=station_order, columns=column_order, fill_value=0.0)
        )
        count_matrix = (
            heat.pivot_table(index="station_or_channel", columns=heatmap_column, values="cells", aggfunc="sum", fill_value=0)
            .reindex(index=station_order, columns=column_order, fill_value=0)
        )
        label_map = station_summary.set_index("station_or_channel")["station_label"].to_dict()
        fig, ax = plt.subplots(figsize=(max(8.5, 0.62 * len(column_order) + 3), max(7.0, 0.30 * len(station_order) + 2)))
        vmax = max(0.01, float(np.nanmax(value_matrix.to_numpy(dtype=float))))
        im = ax.imshow(value_matrix.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(len(column_order)))
        ax.set_xticklabels(_shorten_labels(column_order, 24), rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(station_order)))
        ax.set_yticklabels(_shorten_labels([label_map.get(st, st) for st in station_order], 36), fontsize=8)
        ax.set_title("Dataset-level station-channel gate heatmap")
        ax.set_xlabel(heatmap_column.replace("_", " "))
        ax.set_ylabel("Station/channel")
        for i in range(value_matrix.shape[0]):
            for j in range(value_matrix.shape[1]):
                count = int(count_matrix.iloc[i, j])
                if count > 0:
                    ax.text(j, i, f"{value_matrix.iloc[i, j]:.0%}\nn={count:,}", ha="center", va="center", fontsize=5.4)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Correction applied rate")
    fig.tight_layout()
    fig.savefig(fig_dir / "station_channel_gate_heatmap.png", dpi=230)
    fig.savefig(fig_dir / "station_channel_gate_heatmap.pdf")
    plt.close(fig)

    station_order = station_summary.head(int(top_k))["station_or_channel"].astype(str).tolist()
    action = tables["action_by_station"]
    plot = action[action["station_or_channel"].isin(station_order)].copy()
    pivot = (
        plot.pivot_table(index="station_or_channel", columns="gate_action", values="ratio_of_station", aggfunc="sum", fill_value=0.0)
        .reindex(index=station_order, columns=ACTION_ORDER, fill_value=0.0)
    )
    labels = station_summary.set_index("station_or_channel").reindex(station_order)["station_label"].fillna(pd.Series(station_order, index=station_order))
    fig, ax = plt.subplots(figsize=(9.5, max(6.0, 0.28 * len(station_order) + 2)))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for action_name in ACTION_ORDER:
        values = pivot[action_name].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=ACTION_COLORS[action_name], label=action_name.replace("_", " "))
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(_shorten_labels(labels.tolist(), 36), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of station-channel-hour cells")
    ax.set_title("Gate action composition by station/channel")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_action_composition_by_station.png", dpi=230)
    fig.savefig(fig_dir / "gate_action_composition_by_station.pdf")
    plt.close(fig)

    tier = tables["action_by_tier"].copy()
    tier_pivot = tier.pivot_table(index="impact_tier", columns="gate_action", values="ratio_of_tier", fill_value=0.0)
    tier_pivot = tier_pivot.reindex(columns=ACTION_ORDER, fill_value=0.0)
    fig, ax = plt.subplots(figsize=(8.2, max(3.5, 0.45 * len(tier_pivot) + 1.8)))
    im = ax.imshow(tier_pivot.to_numpy(dtype=float), aspect="auto", cmap="PuBuGn", vmin=0.0, vmax=max(0.01, float(tier_pivot.to_numpy().max())))
    ax.set_xticks(np.arange(len(ACTION_ORDER)))
    ax.set_xticklabels([a.replace("_", "\n") for a in ACTION_ORDER], fontsize=8)
    ax.set_yticks(np.arange(len(tier_pivot.index)))
    ax.set_yticklabels(tier_pivot.index.astype(str))
    title = "Gate action by event impact tier"
    non_none_tiers = [x for x in tier_pivot.index.astype(str).tolist() if x.lower() not in {"none", "unknown"}]
    if len(non_none_tiers) <= 1:
        title += " (low tier diversity diagnostic)"
    ax.set_title(title)
    for i in range(tier_pivot.shape[0]):
        for j in range(tier_pivot.shape[1]):
            ax.text(j, i, f"{tier_pivot.iloc[i, j]:.1%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Share within tier")
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_action_by_event_tier_heatmap.png", dpi=230)
    fig.savefig(fig_dir / "gate_action_by_event_tier_heatmap.pdf")
    plt.close(fig)

    rank = tables["rank_summary"].copy()
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(rank))
    bars = ax1.bar(x, rank["correction_sparsity"], color="#59a14f", label="Correction sparsity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rank["station_rank_group"].astype(str))
    ax1.set_ylabel("Corrected cells / all cells")
    ax1.set_title("Correction sparsity by station-rank group")
    ax1.set_ylim(0, max(0.01, float(rank["correction_sparsity"].max()) * 1.25))
    for bar, (_, row) in zip(bars, rank.iterrows()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['correction_sparsity']:.2%}\n{int(row['corrected_cells']):,} cells",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2 = ax1.twinx()
    ax2.plot(x, rank["station_count"], color="#4e79a7", marker="o", label="Station count")
    ax2.set_ylabel("Station count")
    fig.tight_layout()
    fig.savefig(fig_dir / "correction_sparsity_by_station_rank.png", dpi=230)
    fig.savefig(fig_dir / "correction_sparsity_by_station_rank.pdf")
    plt.close(fig)


def write_reports(
    root: Path,
    *,
    gate_trace: str | Path,
    df: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    top_k_stations: int,
    min_station_cells: int,
    heatmap_column: str,
) -> dict:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    station_summary = tables["station_summary"].copy()
    total_rows = int(len(df))
    station_count = int(station_summary["station_or_channel"].nunique())
    non_event_correction_cells = int((~df["has_structured_event"].astype(bool) & df["correction_applied"].astype(bool)).sum())
    correction_sparsity = float(df["correction_applied"].sum() / max(1, total_rows))
    small_n_station_count = int(station_summary["small_n_flag"].sum())
    top_selected = station_summary.head(10)[
        ["station_or_channel", "station_label", "corrected_cells", "correction_applied_rate", "controller_allowed_rate"]
    ].to_dict(orient="records")
    manifest = {
        "status": "completed",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "gate_trace": str(gate_trace),
        "row_count": total_rows,
        "station_count": station_count,
        "top_k_stations": int(top_k_stations),
        "min_station_cells": int(min_station_cells),
        "heatmap_column": heatmap_column,
        "non_event_correction_cells": non_event_correction_cells,
        "correction_sparsity": correction_sparsity,
        "small_n_station_count": small_n_station_count,
        "case_level_input": False,
        "uses_a3_full_gate_trace": "gate_coverage_diagnostics" in str(gate_trace) and total_rows > 1_000_000,
        "llm_or_qwenplus_used": False,
        "autoskill_updated": False,
        "forecast_array_changed_by_skill": False,
        "outputs": {
            "tables": [
                "tables/station_channel_gate_summary.csv",
                "tables/gate_action_by_station.csv",
                "tables/gate_action_by_tier.csv",
            ],
            "figures": [
                "figures/station_channel_gate_heatmap.png",
                "figures/gate_action_composition_by_station.png",
                "figures/gate_action_by_event_tier_heatmap.png",
                "figures/correction_sparsity_by_station_rank.png",
            ],
        },
    }
    (reports / "station_channel_gate_heatmap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top3 = station_summary.head(3)
    nonzero = int((station_summary["corrected_cells"] > 0).sum())
    zero = int((station_summary["corrected_cells"] == 0).sum())
    lines = [
        "# Station-Channel Gate Heatmap Interpretation",
        "",
        "This dataset-level diagnostic reads the full gate trace and visualizes controller scope over station-channel-hour cells. It is not a numerical forecasting accuracy figure.",
        "",
        "## Dataset-level Scope",
        f"- Trace rows: `{total_rows:,}`.",
        f"- Station/channel units: `{station_count}`.",
        f"- Stations with any correction: `{nonzero}`; zero-correction stations: `{zero}`.",
        f"- Overall correction sparsity: `{correction_sparsity:.4%}`.",
        f"- Non-event correction cells: `{non_event_correction_cells}`.",
        "",
        "## Most Selected Station/Channel Units",
    ]
    for _, row in top3.iterrows():
        lines.append(
            f"- `{row['station_label']}`: `{int(row['corrected_cells']):,}` corrected cells, "
            f"correction rate `{row['correction_applied_rate']:.2%}`."
        )
    lines.extend(
        [
            "",
            "## Paper Recommendation",
            "- The station-channel heatmap and action-composition plot are suitable for the main paper if the paper needs dataset-level evidence that the controller applies sparse, station-specific calibration.",
            "- The impact-tier heatmap should be treated as diagnostic when tier diversity is low.",
            "- Full station-level rows should stay in the CSV/table appendix.",
            "",
            "## Claim Boundary",
            "- Supports: event-station-channel gate controls where bounded correction is allowed.",
            "- Does not support: RAG, AutoSkill, or LLM agents directly improve numerical forecasts.",
        ]
    )
    (reports / "station_channel_gate_heatmap_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def build_tables(
    df: pd.DataFrame,
    channel_meta: Mapping[str, Mapping[str, Any]],
    *,
    min_station_cells: int,
    top_k_stations: int,
    heatmap_column: str,
) -> Dict[str, pd.DataFrame]:
    action_df = add_gate_actions(df)
    station_summary = build_station_summary(action_df, channel_meta, min_station_cells)
    return {
        "station_summary": station_summary,
        "action_by_station": build_action_by_station(action_df, station_summary),
        "action_by_tier": build_action_by_tier(action_df),
        "heatmap": build_heatmap_table(action_df, station_summary, heatmap_column, top_k_stations),
        "rank_summary": build_rank_summary(station_summary),
    }


def run_station_channel_gate_heatmap(
    *,
    gate_trace: str | Path,
    channel_map_json: str | Path | None,
    output_root: str | Path,
    top_k_stations: int = 30,
    min_station_cells: int = 1000,
    heatmap_column: str = "event_type",
) -> dict:
    root = Path(output_root)
    ensure_dirs(root)
    df = read_gate_trace(gate_trace)
    channel_meta = load_channel_metadata(channel_map_json)
    tables = build_tables(
        df,
        channel_meta,
        min_station_cells=int(min_station_cells),
        top_k_stations=int(top_k_stations),
        heatmap_column=heatmap_column,
    )
    write_tables(root, tables)
    write_figures(root, tables, heatmap_column=heatmap_column, top_k=int(top_k_stations))
    return write_reports(
        root,
        gate_trace=gate_trace,
        df=add_gate_actions(df),
        tables=tables,
        top_k_stations=int(top_k_stations),
        min_station_cells=int(min_station_cells),
        heatmap_column=heatmap_column,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate_trace", default=DEFAULT_GATE_TRACE)
    parser.add_argument("--channel_map_json", default=DEFAULT_CHANNEL_MAP)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--top_k_stations", type=int, default=30)
    parser.add_argument("--min_station_cells", type=int, default=1000)
    parser.add_argument("--heatmap_column", choices=["event_type", "impact_tier", "venue_name"], default="event_type")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    if output_root is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_root = f"autotemp/revision_trc_tits_ccfa_{stamp}/station_channel_gate_heatmap"
    manifest = run_station_channel_gate_heatmap(
        gate_trace=args.gate_trace,
        channel_map_json=args.channel_map_json,
        output_root=output_root,
        top_k_stations=args.top_k_stations,
        min_station_cells=args.min_station_cells,
        heatmap_column=args.heatmap_column,
    )
    print(json.dumps({"status": manifest["status"], "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
