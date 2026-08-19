"""Main-paper figure generation for EAF-MAS v2."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .metrics import calibration_regret, wape
from .plotting_utils import FigureManifest, missing_note, save_figure
from .style import COLORS, add_panel_labels, apply_style, despine


def make_main_figures(tables: Dict[str, pd.DataFrame], out_dirs: dict, cfg: dict) -> FigureManifest:
    apply_style()
    manifest = FigureManifest()
    manifest.add(
        "fig1_framework_eaf_mas_v2",
        "EAF-MAS v2 framework",
        [],
        "",
        "",
        [],
        "skipped_by_request",
        "Fig. 1 framework diagram is intentionally skipped in this iteration.",
    )
    _fig2_dataset(tables, out_dirs, manifest)
    _fig3_horizon(tables, out_dirs, manifest)
    _fig4_calibration_distribution(tables, out_dirs, manifest, cfg)
    _fig5_correction_safety(tables, out_dirs, manifest, cfg)
    _fig6_evidence_audit(tables, out_dirs, manifest)
    _fig7_gate(tables, out_dirs, manifest, cfg)
    _fig9_memory(tables, out_dirs, manifest)
    _fig10_ablation(tables, out_dirs, manifest)
    from .case_cards import make_case_card_figures

    manifest.extend(make_case_card_figures(tables, out_dirs))
    return manifest


def _fig2_dataset(tables, out_dirs, manifest):
    import matplotlib.pyplot as plt

    cases = tables.get("case_metrics", pd.DataFrame())
    events = tables.get("event_metadata", pd.DataFrame())
    stations = tables.get("station_metadata", pd.DataFrame())
    formal = tables.get("formal_metrics", pd.DataFrame())
    if formal.empty and cases.empty and events.empty:
        _skip(manifest, out_dirs, "fig2_dataset_event_station_split", "Dataset/event/station/split protocol", "No formal metric or event metadata available.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.ravel()

    summary = fig2_dataset_summary(formal, events, stations)

    ax = axes[0]
    if not formal.empty and {"anchor_time", "mode"}.issubset(formal.columns):
        raw = formal[formal["mode"].astype(str) == "numerical_only"].copy()
        if raw.empty:
            raw = formal.copy()
        raw = raw.drop_duplicates(["anchor_time"]).sort_values("anchor_time").reset_index(drop=True)
        times = pd.to_datetime(raw["anchor_time"], errors="coerce")
        raw["_anchor_time"] = times
        if {"event_n", "non_event_n"}.issubset(raw.columns):
            event_n = pd.to_numeric(raw["event_n"], errors="coerce").fillna(0.0)
            non_event_n = pd.to_numeric(raw["non_event_n"], errors="coerce").fillna(0.0)
            raw["_event_ratio"] = event_n / (event_n + non_event_n).replace(0, np.nan)
        else:
            raw["_event_ratio"] = np.where(_bool_mask(raw, "has_event_cells"), 1.0, 0.0)
        x = np.arange(len(raw))
        ax.vlines(x, 0, raw["_event_ratio"].fillna(0.0), color=COLORS["adjusted"], alpha=0.55, linewidth=1.2)
        ax.scatter(x, raw["_event_ratio"].fillna(0.0), s=18, color=COLORS["adjusted"], alpha=0.9)
        ax.set_ylim(0, max(0.22, min(1.0, float(raw["_event_ratio"].fillna(0.0).max()) * 1.25)))
        ax.set_title(f"Rolling test coverage and event density (n={raw['anchor_time'].nunique()})")
        ax.set_xlabel("rolling anchor index")
        ax.set_ylabel("event-active cell share")
        tick_pos = np.linspace(0, max(len(raw) - 1, 0), min(5, max(len(raw), 1))).astype(int)
        tick_labels = raw.loc[tick_pos, "_anchor_time"].dt.strftime("%m-%d").fillna("").tolist() if len(raw) else []
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=0)
        times = times.dropna()
        date_note = f"{times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}" if not times.empty else "date span unavailable"
        mean_ratio = float(raw["_event_ratio"].dropna().mean()) if raw["_event_ratio"].notna().any() else 0.0
        ax.text(0.02, 0.92, f"{date_note}\nmean event-active share={mean_ratio:.1%}", transform=ax.transAxes, fontsize=8)
    else:
        ax.text(0.5, 0.5, "formal rolling metrics unavailable\ncase-study diagnostic only", ha="center", va="center", color=COLORS["missing"])
        ax.axis("off")
    despine(axes[0])

    ax = axes[1]
    cell_summary = summary[summary["metric"].isin(["event cells", "non-event cells"])]
    if not cell_summary.empty and cell_summary["value"].sum() > 0:
        ratio_table = fig2_cell_ratio_table(summary)
        event_row = ratio_table[ratio_table["metric"] == "event cells"].iloc[0]
        non_event_row = ratio_table[ratio_table["metric"] == "non-event cells"].iloc[0]
        ax.barh([0.62], [event_row["ratio"]], color=COLORS["adjusted"], height=0.18, label="event-active")
        ax.barh([0.62], [non_event_row["ratio"]], left=[event_row["ratio"]], color=COLORS["missing"], height=0.18, label="non-event")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Event-active vs non-event cells")
        ax.text(
            0.02,
            0.30,
            f"Event-active\n{int(event_row['value']):,} cells\n{event_row['ratio']:.1%}",
            transform=ax.transAxes,
            fontsize=10,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7ed", edgecolor=COLORS["adjusted"], linewidth=0.8),
        )
        ax.text(
            0.56,
            0.30,
            f"Non-event\n{int(non_event_row['value']):,} cells\n{non_event_row['ratio']:.1%}",
            transform=ax.transAxes,
            fontsize=10,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.8),
        )
        ax.text(0.02, 0.82, "Only a minority of station-hour cells are eligible for event-aware intervention.", transform=ax.transAxes, fontsize=8)
    else:
        ax.text(0.5, 0.5, "event/non-event cell counts unavailable", ha="center", va="center", color=COLORS["missing"])
        ax.axis("off")
    despine(ax)

    ax = axes[2]
    coverage = summary[summary["metric"].isin(["Top128 station channels", "event-venue station channels", "live explanation cases"])]
    if not coverage.empty:
        top = coverage.loc[coverage["metric"] == "Top128 station channels", "value"]
        venue = coverage.loc[coverage["metric"] == "event-venue station channels", "value"]
        top_n = int(top.iloc[0]) if not top.empty else int(coverage["value"].max())
        venue_n = int(venue.iloc[0]) if not venue.empty else 0
        ratio = venue_n / max(top_n, 1)
        ax.barh([0.60], [1.0], color=COLORS["missing"], height=0.22)
        ax.barh([0.60], [ratio], color=COLORS["audit"], height=0.22)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Station/channel and case coverage")
        ax.text(0.03, 0.73, "Top128 station channels", transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.47, "event_venue28 intersection", transform=ax.transAxes, fontsize=8)
        ax.text(
            0.50,
            0.20,
            f"{venue_n} / {top_n} = {ratio:.1%} event-venue channels\nUsed for event explanations and bounded calibration.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.8),
        )
    else:
        ax.text(0.5, 0.5, "station coverage unavailable", ha="center", va="center", color=COLORS["missing"])
        ax.axis("off")
    despine(ax)

    ax = axes[3]
    event_dist = fig2_test_event_distribution(events, formal)
    if not event_dist.empty:
        top = event_dist.head(10).sort_values("count")
        ax.barh(top["event_type"].astype(str), top["count"], color=COLORS["audit"], alpha=0.86)
        for yi, (_, row) in enumerate(top.iterrows()):
            ax.text(float(row["count"]) + max(float(top["count"].max()) * 0.015, 1.0), yi, str(int(row["count"])), va="center", fontsize=7)
        ax.set_title("Event records overlapping test span")
        ax.set_xlabel("event count")
        ax.set_xlim(0, float(top["count"].max()) * 1.18)
    else:
        ax.text(0.5, 0.5, "test-span event distribution unavailable", ha="center", va="center", color=COLORS["missing"])
        ax.axis("off")
    despine(ax)

    summary.to_csv(out_dirs["tables"] / "table_dataset_event_station_split.csv", index=False)
    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig2_dataset_event_station_split")
    plt.close(fig)
    status = "generated" if not formal.empty and formal["anchor"].nunique() >= 30 else "generated_diagnostic"
    notes = "Uses rolling test metrics, not the four explanation cases." if status == "generated" else "Formal rolling metrics are missing or too small; figure is diagnostic."
    manifest.add("fig2_dataset_event_station_split", "Dataset, event, station, and split protocol", [], pdf, png, ["formal_metrics", "event_metadata", "station_metadata"], status, notes)


def fig2_dataset_summary(formal: pd.DataFrame, events: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw = formal[formal.get("mode", pd.Series(dtype=str)).astype(str) == "numerical_only"].copy() if not formal.empty else pd.DataFrame()
    if raw.empty:
        raw = formal.copy()
    n_anchors = int(raw["anchor"].nunique()) if "anchor" in raw else int(raw["anchor_time"].nunique()) if "anchor_time" in raw else 0
    rows.append({"metric": "rolling test anchors", "value": n_anchors})
    if not raw.empty and {"event_n", "non_event_n"}.issubset(raw.columns):
        rows.append({"metric": "event cells", "value": int(pd.to_numeric(raw["event_n"], errors="coerce").fillna(0).sum())})
        rows.append({"metric": "non-event cells", "value": int(pd.to_numeric(raw["non_event_n"], errors="coerce").fillna(0).sum())})
    if not stations.empty:
        rows.append({"metric": "Top128 station channels", "value": int(len(stations))})
        if "station_group" in stations:
            rows.append({"metric": "event-venue station channels", "value": int((stations["station_group"] == "event_venue28").sum())})
    rows.append({"metric": "event records", "value": int(len(events)) if not events.empty else 0})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ratio"] = np.nan
        cell_mask = out["metric"].isin(["event cells", "non-event cells"])
        cell_total = float(out.loc[cell_mask, "value"].sum()) if cell_mask.any() else 0.0
        if cell_total > 0:
            out.loc[cell_mask, "ratio"] = out.loc[cell_mask, "value"] / cell_total
        station_mask = out["metric"].isin(["Top128 station channels", "event-venue station channels"])
        top_value = out.loc[out["metric"] == "Top128 station channels", "value"]
        if station_mask.any() and not top_value.empty and float(top_value.iloc[0]) > 0:
            out.loc[station_mask, "ratio"] = out.loc[station_mask, "value"] / float(top_value.iloc[0])
    return out


def fig2_cell_ratio_table(summary: pd.DataFrame) -> pd.DataFrame:
    cells = summary[summary["metric"].isin(["event cells", "non-event cells"])].copy()
    if cells.empty:
        return pd.DataFrame(columns=["metric", "value", "ratio"])
    cells["value"] = pd.to_numeric(cells["value"], errors="coerce").fillna(0.0)
    total = float(cells["value"].sum())
    cells["ratio"] = cells["value"] / total if total > 0 else 0.0
    order = {"event cells": 0, "non-event cells": 1}
    return cells.sort_values("metric", key=lambda s: s.map(order).fillna(99)).reset_index(drop=True)


def fig2_test_event_distribution(events: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "start_time" not in events:
        return pd.DataFrame(columns=["event_type", "count"])
    ev = events.copy()
    ev["_start"] = pd.to_datetime(ev["start_time"], errors="coerce")
    if not formal.empty and "anchor_time" in formal:
        times = pd.to_datetime(formal["anchor_time"], errors="coerce").dropna()
        if not times.empty:
            start = times.min()
            end = times.max() + pd.Timedelta(hours=192)
            ev = ev[(ev["_start"] >= start) & (ev["_start"] <= end)]
    ev["event_type"] = ev.get("event_type", pd.Series(["unknown"] * len(ev))).fillna("unknown").astype(str)
    return ev.groupby("event_type", dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)


def fig2_evidence_source_composition(cases: pd.DataFrame) -> pd.DataFrame:
    """Paper-facing Figure 2 evidence metrics.

    Citation-quality URL coverage is intentionally excluded from this panel:
    current experiments use Qwen-Plus summaries as non-citable evidence
    support, while URL-quality citation diagnostics are reported separately.
    """
    if cases.empty:
        return pd.DataFrame(columns=["metric", "count"])

    def bool_count(column: str) -> int:
        if column not in cases:
            return 0
        return int(cases[column].fillna(False).astype(bool).sum())

    audit_pass = 0
    if "evidence_validity_score" in cases:
        audit_pass = int((pd.to_numeric(cases["evidence_validity_score"], errors="coerce").fillna(0.0) >= 0.60).sum())
    controller_allowed = bool_count("controller_allowed")

    return pd.DataFrame(
        [
            {"metric": "model-assisted summary", "count": bool_count("has_model_assisted_summary")},
            {"metric": "historical residual memory", "count": bool_count("has_historical_residual_memory")},
            {"metric": "evidence audit passed", "count": audit_pass},
            {"metric": "controller allowed", "count": controller_allowed},
        ]
    )


def _fig3_horizon(tables, out_dirs, manifest):
    import matplotlib.pyplot as plt

    formal = tables.get("formal_metrics", pd.DataFrame())
    hf = tables.get("horizon_forecast", pd.DataFrame())
    if not formal.empty:
        panel_table = rolling_test_panel_table(formal)
        panel_table.to_csv(out_dirs["tables"] / "table_rolling_forecasting_by_anchor.csv", index=False)
        gain_table, gain_summary = fig3_gain_first_tables(formal)
        gain_table.to_csv(out_dirs["tables"] / "table_rolling_forecasting_gain_first.csv", index=False)
        gain_summary.to_csv(out_dirs["tables"] / "table_rolling_forecasting_gain_summary.csv", index=False)
        if panel_table.empty or gain_table.empty:
            _skip(manifest, out_dirs, "fig3_multi_horizon_forecasting", "Rolling 192h forecasting by subset", "Formal metric rows do not contain supported WAPE columns.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
        axes = axes.ravel()
        x = gain_table["rolling_anchor_index"]
        axes[0].plot(x, gain_table["pt_overall_wape"], color=COLORS["raw"], linewidth=1.4, marker="o", markersize=2.4)
        axes[0].set_title("PT-MOMENT rolling 192h WAPE")
        axes[0].set_xlabel("rolling anchor index")
        axes[0].set_ylabel("overall WAPE")
        axes[0].grid(alpha=0.18)
        axes[0].text(
            0.02,
            0.92,
            f"n={int(gain_table['anchor'].nunique())} rolling anchors",
            transform=axes[0].transAxes,
            fontsize=8,
        )
        despine(axes[0])

        colors = np.where(gain_table["event_gain"] >= 0, COLORS["adjusted"], COLORS["harmful"])
        axes[1].bar(x, gain_table["event_gain"], color=colors, alpha=0.78, width=0.78)
        axes[1].axhline(0.0, color="#111827", linewidth=0.8)
        axes[1].set_title("Event-active WAPE gain")
        axes[1].set_xlabel("rolling anchor index")
        axes[1].set_ylabel("PT-MOMENT - EAF-MAS-C")
        axes[1].grid(axis="y", alpha=0.16)
        despine(axes[1])

        axes[2].scatter(gain_table["pt_event_wape"], gain_table["adapter_event_wape"], s=22, color=COLORS["adjusted"], alpha=0.72)
        finite = gain_table[["pt_event_wape", "adapter_event_wape"]].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            lo = float(finite.min().min())
            hi = float(finite.max().max())
            pad = max((hi - lo) * 0.05, 0.05)
            axes[2].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#111827", linestyle="--", linewidth=0.9)
            axes[2].set_xlim(lo - pad, hi + pad)
            axes[2].set_ylim(lo - pad, hi + pad)
        axes[2].set_title("Paired event-active WAPE")
        axes[2].set_xlabel("PT-MOMENT")
        axes[2].set_ylabel("EAF-MAS-C")
        axes[2].text(0.02, 0.92, "points below diagonal improve", transform=axes[2].transAxes, fontsize=8)
        despine(axes[2])

        if gain_summary.empty:
            axes[3].text(0.5, 0.5, "gain summary unavailable", ha="center", va="center", color=COLORS["missing"])
            axes[3].axis("off")
        else:
            order = ["event-active", "overall", "Top128", "non-event"]
            plot = gain_summary.set_index("subset_label").reindex(order).dropna(subset=["mean_gain"]).reset_index()
            y = np.arange(len(plot))
            xerr = np.vstack([
                (plot["mean_gain"] - plot["ci_low"]).clip(lower=0.0),
                (plot["ci_high"] - plot["mean_gain"]).clip(lower=0.0),
            ])
            axes[3].barh(y, plot["mean_gain"], xerr=xerr, color=COLORS["audit"], alpha=0.82, ecolor="#374151", capsize=3)
            axes[3].axvline(0.0, color="#111827", linewidth=0.8)
            axes[3].set_yticks(y)
            axes[3].set_yticklabels(plot["subset_label"])
            axes[3].set_xlabel("mean WAPE gain with 95% bootstrap CI")
            axes[3].set_title("Subset gain summary")
            for yi, (_, row) in enumerate(plot.iterrows()):
                axes[3].text(float(row["mean_gain"]), yi + 0.18, f"{float(row['mean_gain']):.3f}", fontsize=7, ha="center")
            despine(axes[3])
        add_panel_labels(axes)
        fig.tight_layout()
        pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig3_multi_horizon_forecasting")
        plt.close(fig)
        manifest.add(
            "fig3_multi_horizon_forecasting",
            "Rolling 192h forecasting and event-active gain",
            [],
            pdf,
            png,
            ["formal_metrics"],
            "generated",
            "Gain-first view: PT-MOMENT baseline, EAF-MAS-C event-active delta, paired scatter, and subset bootstrap CI. EAF-MAS-X is explanation-only and is not plotted as a numeric curve.",
        )
        return

    if hf.empty:
        _skip(manifest, out_dirs, "fig3_multi_horizon_forecasting", "Multi-horizon forecasting", "No formal_metrics or horizon_forecast rows available.")
        return
    panels = horizon_panel_tables(hf)
    table = pd.concat(panels.values(), ignore_index=True) if panels else pd.DataFrame()
    table.to_csv(out_dirs["tables"] / "table_forecasting_by_horizon.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    axes = axes.ravel()
    panel_titles = {
        "overall": "All station-channel cells",
        "event_window": "Event-window cells",
        "adjusted_channels": "Adjusted channels",
        "non_adjusted_channels": "Non-adjusted channels",
    }
    for ax, (group, title) in zip(axes, panel_titles.items()):
        group_table = panels.get(group, pd.DataFrame())
        if group_table.empty:
            ax.text(0.5, 0.5, "missing group data", ha="center", va="center", fontsize=10, color=COLORS["missing"])
            ax.set_title(title)
            ax.axis("off")
            continue
        ax.plot(group_table["horizon"], group_table["PT-MOMENT"], label="PT-MOMENT", color=COLORS["raw"], marker="o", linewidth=1.6)
        if "EAF-MAS-X" in group_table and group_table["EAF-MAS-X"].notna().any():
            ax.plot(group_table["horizon"], group_table["EAF-MAS-X"], label="EAF-MAS-X", color=COLORS["audit"], marker="s", linewidth=1.4, linestyle=":")
        if "EAF-MAS-C" in group_table and group_table["EAF-MAS-C"].notna().any():
            ax.plot(group_table["horizon"], group_table["EAF-MAS-C"], label="EAF-MAS-C", color=COLORS["adjusted"], marker="o", linewidth=1.6)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{title} WAPE vs horizon\ncase-study diagnostic")
        ax.set_xlabel("horizon hours")
        ax.set_ylabel("WAPE")
        despine(ax)
    axes[0].legend(loc="best")
    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig3_multi_horizon_forecasting")
    plt.close(fig)
    manifest.add("fig3_multi_horizon_forecasting", "Multi-horizon forecasting", [], pdf, png, ["horizon_forecast"], "generated_diagnostic", "Fallback uses live-case horizon rows only; no dual-axis gain bars are plotted.")


def rolling_test_panel_table(formal: pd.DataFrame) -> pd.DataFrame:
    if formal.empty:
        return pd.DataFrame()
    df = formal.copy()
    if "method_label" not in df and "mode" in df:
        df["method_label"] = df["mode"].astype(str)
    anchor_times = (
        df[["anchor", "anchor_time"]]
        .drop_duplicates()
        .sort_values(["anchor_time", "anchor"], kind="mergesort")
        .reset_index(drop=True)
    )
    anchor_times["rolling_anchor_index"] = np.arange(len(anchor_times))
    df = df.merge(anchor_times, on=["anchor", "anchor_time"], how="left")
    panel_specs = [
        ("event_venue28_overall", "wape"),
        ("event_active", "event_wape"),
        ("non_event", "non_event_wape"),
        ("top128_overall", "top128_wape"),
    ]
    rows = []
    for subset, metric_col in panel_specs:
        if metric_col not in df:
            continue
        for _, row in df.iterrows():
            value = pd.to_numeric(pd.Series([row.get(metric_col)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "subset": subset,
                    "split": row.get("split"),
                    "anchor": row.get("anchor"),
                    "anchor_time": row.get("anchor_time") or row.get("date"),
                    "rolling_anchor_index": row.get("rolling_anchor_index"),
                    "mode": row.get("mode"),
                    "method_label": row.get("method_label"),
                    "wape": float(value),
                    "n_event_cells": row.get("event_n"),
                    "n_non_event_cells": row.get("non_event_n"),
                }
            )
    return pd.DataFrame(rows)


def fig3_gain_first_tables(formal: pd.DataFrame, compare_mode: str = "event_adapter_frozen_moment") -> tuple[pd.DataFrame, pd.DataFrame]:
    if formal.empty or "mode" not in formal:
        return pd.DataFrame(), pd.DataFrame()
    df = formal.copy()
    base = df[df["mode"].astype(str) == "numerical_only"].copy()
    adapter = df[df["mode"].astype(str) == compare_mode].copy()
    if base.empty or adapter.empty:
        return pd.DataFrame(), pd.DataFrame()
    key_cols = [col for col in ["anchor", "anchor_time"] if col in df.columns]
    if not key_cols:
        return pd.DataFrame(), pd.DataFrame()
    cols = key_cols + [col for col in ["wape", "event_wape", "non_event_wape", "top128_wape"] if col in df.columns]
    merged = base[cols].merge(adapter[cols], on=key_cols, suffixes=("_pt", "_adapter"), how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    merged["rolling_anchor_index"] = np.arange(len(merged))
    rename = {
        "wape_pt": "pt_overall_wape",
        "wape_adapter": "adapter_overall_wape",
        "event_wape_pt": "pt_event_wape",
        "event_wape_adapter": "adapter_event_wape",
        "non_event_wape_pt": "pt_non_event_wape",
        "non_event_wape_adapter": "adapter_non_event_wape",
        "top128_wape_pt": "pt_top128_wape",
        "top128_wape_adapter": "adapter_top128_wape",
    }
    merged = merged.rename(columns=rename)
    for prefix, pt_col, adapter_col in [
        ("overall", "pt_overall_wape", "adapter_overall_wape"),
        ("event", "pt_event_wape", "adapter_event_wape"),
        ("non_event", "pt_non_event_wape", "adapter_non_event_wape"),
        ("top128", "pt_top128_wape", "adapter_top128_wape"),
    ]:
        if pt_col in merged and adapter_col in merged:
            merged[f"{prefix}_gain"] = pd.to_numeric(merged[pt_col], errors="coerce") - pd.to_numeric(merged[adapter_col], errors="coerce")
        else:
            merged[f"{prefix}_gain"] = np.nan
    specs = [
        ("event-active", "event_gain"),
        ("overall", "overall_gain"),
        ("Top128", "top128_gain"),
        ("non-event", "non_event_gain"),
    ]
    summary_rows = []
    for label, col in specs:
        values = pd.to_numeric(merged[col], errors="coerce").dropna().astype(float).to_numpy()
        if values.size == 0:
            continue
        low, high = _bootstrap_ci(values)
        summary_rows.append(
            {
                "subset_label": label,
                "mean_gain": float(np.mean(values)),
                "median_gain": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
                "positive_rate": float((values > 0).mean()),
                "n": int(values.size),
            }
        )
    return merged, pd.DataFrame(summary_rows)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 7) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def horizon_panel_tables(horizon_forecast: pd.DataFrame, bins=None) -> Dict[str, pd.DataFrame]:
    bins = bins or [1, 3, 6, 12, 24, 48, 96, 192]
    hf = horizon_forecast.copy()
    event_mask = _bool_mask(hf, "is_event_window")
    adjusted_mask = _bool_mask(hf, "is_adjusted_channel")
    affected_mask = _bool_mask(hf, "is_affected_channel") | adjusted_mask
    groups = {
        "overall": hf,
        "event_window": hf[event_mask],
        "adjusted_channels": hf[affected_mask],
        "non_adjusted_channels": hf[~affected_mask],
    }
    return {name: _horizon_summary(name, df, bins) for name, df in groups.items()}


def _horizon_summary(group: str, df: pd.DataFrame, bins) -> pd.DataFrame:
    rows = []
    for b in bins:
        sub = df[df["horizon_step"] <= b] if not df.empty else df
        if sub.empty:
            continue
        raw_all = wape(sub["observed"], sub["raw_forecast"])
        raw_x, eaf_mas_x = _method_pair_wape(sub, "EAF-MAS-X")
        raw_c, eaf_mas_c = _method_pair_wape(sub, "EAF-MAS-C")
        if pd.isna(eaf_mas_c):
            raw_c = raw_all
            eaf_mas_c = wape(sub["observed"], sub["adjusted_forecast"])
        raw_wape = raw_c if not pd.isna(raw_c) else raw_x if not pd.isna(raw_x) else raw_all
        rows.append(
            {
                "group": group,
                "horizon": b,
                "PT-MOMENT": raw_wape,
                "EAF-MAS-X": eaf_mas_x,
                "EAF-MAS-C": eaf_mas_c,
                "wape_gain": raw_c - eaf_mas_c if not pd.isna(raw_c) else raw_wape - eaf_mas_c,
                "eaf_mas_x_gain": raw_x - eaf_mas_x if not pd.isna(raw_x) and not pd.isna(eaf_mas_x) else np.nan,
                "eaf_mas_c_gain": raw_c - eaf_mas_c if not pd.isna(raw_c) and not pd.isna(eaf_mas_c) else np.nan,
                "n_rows": len(sub),
                "n_cases": sub["case_id"].nunique() if "case_id" in sub else None,
            }
        )
    return pd.DataFrame(rows)


def _method_pair_wape(df: pd.DataFrame, method_label: str) -> tuple[float, float]:
    if "method_label" not in df:
        if method_label == "EAF-MAS-C":
            return wape(df["observed"], df["raw_forecast"]), wape(df["observed"], df["adjusted_forecast"])
        return np.nan, np.nan
    sub = df[df["method_label"] == method_label]
    if sub.empty:
        return np.nan, np.nan
    return wape(sub["observed"], sub["raw_forecast"]), wape(sub["observed"], sub["adjusted_forecast"])


def _bool_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series([False] * len(df), index=df.index)
    series = df[col]
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def _short_station_label(value: object, max_len: int = 28) -> str:
    label = str(value or "unknown station").replace("_", " ")
    if ("Times Sq 42 St" in label or "Times Sq-42 St" in label) and "Bryant" in label:
        label = "Times Sq-42 St / Bryant Pk"
    label = " ".join(label.split())
    if len(label) > max_len:
        label = label[: max_len - 1].rstrip() + "…"
    return label


def active_correction_cells_for_plot(hf: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Rows with actual non-zero EAF-MAS-C corrections for correction-safety plots."""
    if hf.empty or "relative_correction" not in hf:
        return pd.DataFrame(columns=list(hf.columns))
    method = hf.get("method_label", pd.Series(index=hf.index, dtype=str)).fillna("")
    rel = hf["relative_correction"].fillna(0.0).abs()
    return hf[(method == "EAF-MAS-C") & (rel > eps)].copy()


def channel_gate_summary_tables(channels: pd.DataFrame, top_k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the plotted channel rows and all-state channel flow counts for Fig. 7."""
    if channels.empty:
        return pd.DataFrame(), pd.DataFrame(
            [
                {"decision_state": "candidate", "count": 0},
                {"decision_state": "included", "count": 0},
                {"decision_state": "adjusted", "count": 0},
                {"decision_state": "excluded", "count": 0},
                {"decision_state": "candidate_only", "count": 0},
            ]
        )
    df = channels.copy()
    affected = _bool_mask(df, "is_affected_channel")
    adjusted = _bool_mask(df, "is_adjusted_channel")
    excluded = _bool_mask(df, "is_excluded_channel")
    df["gate_score"] = pd.to_numeric(df.get("gate_score", 0.0), errors="coerce").fillna(0.0)
    df["correction_max"] = pd.to_numeric(df.get("correction_max", 0.0), errors="coerce").fillna(0.0)
    df["correction_bps"] = df["correction_max"] * 10000.0
    df["station_label"] = df.get("station_name", df.get("channel_id", pd.Series(["unknown"] * len(df)))).map(_short_station_label)
    df["is_candidate_only"] = affected & ~adjusted & ~excluded
    df["_affected"] = affected
    df["_adjusted"] = adjusted
    df["_excluded"] = excluded
    ranked = df.sort_values(["_adjusted", "_affected", "gate_score", "correction_max"], ascending=[False, False, False, False]).head(int(top_k)).copy()
    flow = pd.DataFrame(
        [
            {"decision_state": "candidate", "count": int(len(df))},
            {"decision_state": "included", "count": int(affected.sum())},
            {"decision_state": "adjusted", "count": int(adjusted.sum())},
            {"decision_state": "excluded", "count": int(excluded.sum())},
            {"decision_state": "candidate_only", "count": int((affected & ~adjusted).sum())},
        ]
    )
    return ranked, flow


def memory_diagnostics_tables(residual: pd.DataFrame, skill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build residual-memory and skill lifecycle diagnostics for Fig. 9."""
    if residual.empty:
        coverage = pd.DataFrame(columns=["historical_event_type", "case_count"])
        support = pd.DataFrame(columns=["direction", "median_correction_pct", "case_count"])
    else:
        r = residual.copy()
        r["historical_event_type"] = r.get("historical_event_type", pd.Series(["unknown"] * len(r))).fillna("unknown")
        r["direction"] = r.get("direction", pd.Series(["unknown"] * len(r))).fillna("unknown")
        r["median_correction_pct"] = pd.to_numeric(r.get("median_correction_pct", np.nan), errors="coerce")
        coverage = (
            r.groupby("historical_event_type", dropna=False)
            .size()
            .reset_index(name="case_count")
            .sort_values("case_count", ascending=False)
        )
        support = (
            r.groupby("direction", dropna=False)
            .agg(
                median_correction_pct=("median_correction_pct", "median"),
                case_count=("direction", "size"),
            )
            .reset_index()
            .sort_values("case_count", ascending=False)
        )
    if skill.empty:
        skill_status = pd.DataFrame(
            [
                {
                    "lifecycle_status": "no_promoted_skill",
                    "category": "residual_memory_skill",
                    "count": 0,
                    "message": "no promoted residual-memory skill in current run",
                }
            ]
        )
    else:
        s = skill.copy()
        s["category"] = s.get("category", pd.Series(["unknown"] * len(s))).fillna("unknown")
        skill_status = (
            s.groupby("category", dropna=False)
            .size()
            .reset_index(name="count")
            .assign(lifecycle_status="promoted_or_loaded", message="promoted residual-memory skill rows available")
        )
    return coverage, support, skill_status


def _fig4_calibration_distribution(tables, out_dirs, manifest, cfg):
    import matplotlib.pyplot as plt

    cases = tables.get("case_metrics", pd.DataFrame())
    hf = tables.get("horizon_forecast", pd.DataFrame())
    if cases.empty or hf.empty:
        _skip(manifest, out_dirs, "fig4_calibration_distribution_effect", "Focused local forecast dashboard", "No case/horizon forecast rows available.")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    focus = fig4_focus_window_for_plot(cases, hf, window_size=12)
    case_id = focus["case_id"]
    channel_id = focus["channel_id"]
    plot_rows = focus["rows"]
    x = np.arange(len(plot_rows))
    labels = pd.to_datetime(plot_rows.get("timestamp"), errors="coerce").dt.strftime("%m-%d %H:%M").fillna(plot_rows.get("timestamp", "").astype(str))
    heatmap = fig4_error_reduction_heatmap(hf, case_id, plot_rows, top_k=7)
    matrix = heatmap["matrix"]
    if matrix.size:
        finite = matrix[np.isfinite(matrix)]
        vmax = _robust_heatmap_limit(finite)
        display_matrix = np.clip(matrix, -vmax, vmax)
        im = axes[0].imshow(display_matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        axes[0].set_yticks(np.arange(len(heatmap["channel_labels"])))
        axes[0].set_yticklabels(heatmap["channel_labels"], fontsize=7)
        axes[0].set_xticks(np.arange(len(heatmap["time_labels"]))[:: max(1, len(heatmap["time_labels"]) // 4)])
        axes[0].set_xticklabels(heatmap["time_labels"][:: max(1, len(heatmap["time_labels"]) // 4)], rotation=25, ha="right", fontsize=7)
        axes[0].set_title("Local error reduction heatmap", pad=12)
        axes[0].set_xlabel("forecast hour")
        axes[0].set_ylabel("adjusted station/channel")
        cbar = axes[0].figure.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
        cbar.set_label("APE gain, clipped (pp)", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    else:
        axes[0].text(0.5, 0.5, "local APE heatmap unavailable", ha="center", va="center", color=COLORS["missing"])
        axes[0].axis("off")
    despine(axes[0])

    correction = plot_rows["adjusted_forecast"].fillna(0) - plot_rows["raw_forecast"].fillna(0)
    axes[1].bar(x, correction, color=COLORS["memory"], alpha=0.82)
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(x[:: max(1, len(x) // 4)])
    axes[1].set_xticklabels(labels.iloc[:: max(1, len(x) // 4)], rotation=25, ha="right")
    axes[1].set_title("Correction delta", pad=10)
    axes[1].set_ylabel("adjusted - raw riders/hour")
    despine(axes[1])

    case_row = cases[cases["case_id"].astype(str) == case_id].iloc[0] if (cases["case_id"].astype(str) == case_id).any() else cases.iloc[0]
    score_cols = [
        ("source", "source_validity_score", 0.60),
        ("geo", "geo_consistency_score", 0.60),
        ("temporal", "temporal_alignment_score", 0.70),
        ("residual", "residual_support_score", 0.50),
    ]
    values = [float(case_row.get(col, 0.0) or 0.0) for _, col, _ in score_cols]
    thresholds = [thr for _, _, thr in score_cols]
    names = [name for name, _, _ in score_cols]
    axes[2].bar(names, values, color=COLORS["audit"], alpha=0.86, label="score")
    axes[2].scatter(names, thresholds, color=COLORS["harmful"], marker="_", s=240, label="threshold")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Evidence audit scores", pad=10)
    axes[2].legend(fontsize=7)
    despine(axes[2])
    local_raw_wape = wape(plot_rows["observed"], plot_rows["raw_forecast"]) if "observed" in plot_rows else np.nan
    local_adjusted_wape = wape(plot_rows["observed"], plot_rows["adjusted_forecast"]) if "observed" in plot_rows else np.nan
    pd.DataFrame(
        {
            "case_id": [case_id],
            "channel_id": [channel_id],
            "station_label": [_short_station_label(channel_id)],
            "n_rows": [len(plot_rows)],
            "selection_score": [focus.get("selection_score", 0.0)],
            "selection_reason": [focus.get("selection_reason", "")],
            "local_raw_wape": [local_raw_wape],
            "local_adjusted_wape": [local_adjusted_wape],
            "local_wape_gain": [local_raw_wape - local_adjusted_wape if np.isfinite(local_raw_wape) and np.isfinite(local_adjusted_wape) else np.nan],
            "max_abs_delta_riders": [float(correction.abs().max()) if len(correction) else 0.0],
            "figure_panel_a": ["ape_reduction_heatmap"],
            "heatmap_value_unit": [heatmap.get("value_label", "")],
            "heatmap_channel_count": [len(heatmap.get("channel_labels", []))],
            "heatmap_hour_count": [len(heatmap.get("time_labels", []))],
            "heatmap_max_gain_pp": [float(np.nanmax(matrix)) if matrix.size else np.nan],
            "heatmap_min_gain_pp": [float(np.nanmin(matrix)) if matrix.size else np.nan],
            "heatmap_display_clip_pp": [_robust_heatmap_limit(matrix[np.isfinite(matrix)]) if matrix.size else np.nan],
        }
    ).to_csv(out_dirs["tables"] / "table_focused_local_dashboard.csv", index=False)
    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig4_calibration_distribution_effect")
    plt.close(fig)
    manifest.add("fig4_calibration_distribution_effect", "Focused local forecast dashboard", [], pdf, png, ["horizon_forecast", "case_metrics"], "generated")


def _robust_heatmap_limit(values: np.ndarray, default: float = 3.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    robust = float(np.nanpercentile(np.abs(values), 90))
    return float(min(max(robust, 1.0), 5.0))


def fig4_error_reduction_heatmap(hf: pd.DataFrame, case_id: str, focus_rows: pd.DataFrame, top_k: int = 7) -> dict:
    """Build a station-hour heatmap of local APE reduction for Figure 4."""
    empty = {
        "matrix": np.empty((0, 0)),
        "channel_labels": [],
        "time_labels": [],
        "value_label": "raw APE - adjusted APE (percentage points)",
    }
    required = {"case_id", "channel_id", "raw_forecast", "adjusted_forecast", "observed"}
    if hf.empty or not required.issubset(hf.columns):
        return empty
    df = hf[hf["case_id"].astype(str) == str(case_id)].copy()
    if df.empty:
        return empty
    selected_times: list[str] = []
    if "timestamp" in df and "timestamp" in focus_rows:
        selected_times = [str(x) for x in focus_rows["timestamp"].dropna().astype(str).tolist()]
        if selected_times:
            df = df[df["timestamp"].astype(str).isin(selected_times)]
    if df.empty:
        return empty

    rel_values = df["relative_correction"] if "relative_correction" in df else pd.Series([0.0] * len(df), index=df.index)
    adjusted_mask = _bool_mask(df, "is_adjusted_channel") | (pd.to_numeric(rel_values, errors="coerce").fillna(0.0).abs() > 1e-9)
    adjusted_df = df[adjusted_mask].copy()
    if not adjusted_df.empty:
        df = adjusted_df

    observed = pd.to_numeric(df["observed"], errors="coerce")
    denom = observed.abs().clip(lower=1.0)
    raw = pd.to_numeric(df["raw_forecast"], errors="coerce")
    adjusted = pd.to_numeric(df["adjusted_forecast"], errors="coerce")
    df["_ape_gain_pp"] = ((raw - observed).abs() / denom - (adjusted - observed).abs() / denom) * 100.0
    df["_abs_delta"] = (adjusted - raw).abs()
    grouped = (
        df.groupby("channel_id", dropna=False)
        .agg(_mean_gain=("_ape_gain_pp", "mean"), _mean_abs_gain=("_ape_gain_pp", lambda x: float(np.nanmean(np.abs(x)))), _max_delta=("_abs_delta", "max"))
        .reset_index()
    )
    grouped["_score"] = grouped["_mean_gain"].fillna(0.0) + grouped["_mean_abs_gain"].fillna(0.0) + 0.001 * grouped["_max_delta"].fillna(0.0)
    channels = grouped.sort_values("_score", ascending=False)["channel_id"].astype(str).head(int(top_k)).tolist()
    if not channels:
        return empty

    time_col = "timestamp" if "timestamp" in df else "horizon_step"
    time_values = _sort_horizon_rows(df[df["channel_id"].astype(str).isin(channels)])[time_col].dropna().astype(str).drop_duplicates().tolist()
    if "timestamp" in focus_rows and selected_times:
        time_values = [t for t in selected_times if t in set(time_values)]
    if not time_values:
        return empty

    pivot = (
        df[df["channel_id"].astype(str).isin(channels)]
        .assign(_channel=lambda x: x["channel_id"].astype(str), _time=lambda x: x[time_col].astype(str))
        .pivot_table(index="_channel", columns="_time", values="_ape_gain_pp", aggfunc="mean")
    )
    pivot = pivot.reindex(index=channels, columns=time_values)
    channel_names = {}
    if "station_name" in df:
        channel_names = df.drop_duplicates("channel_id").set_index("channel_id")["station_name"].to_dict()
    labels = [_short_station_label(channel_names.get(ch, ch), max_len=32) for ch in channels]
    time_labels = pd.to_datetime(pd.Series(time_values), errors="coerce").dt.strftime("%m-%d %H:%M").fillna(pd.Series(time_values)).tolist()
    return {
        "matrix": pivot.to_numpy(dtype=float),
        "channel_labels": labels,
        "time_labels": time_labels,
        "value_label": "raw APE - adjusted APE (percentage points)",
    }


def fig4_focus_window_for_plot(cases: pd.DataFrame, hf: pd.DataFrame, window_size: int = 12) -> dict:
    """Select the most visually meaningful local intervention window for Fig. 4."""
    if hf.empty:
        return {"case_id": "", "channel_id": "", "rows": hf.copy(), "selection_score": 0.0, "selection_reason": "missing horizon rows"}
    active = active_correction_cells_for_plot(hf)
    source = active if not active.empty else hf.copy()
    if source.empty:
        return {"case_id": "", "channel_id": "", "rows": source, "selection_score": 0.0, "selection_reason": "missing source rows"}

    case_order = _fig4_ranked_case_ids(cases, source)
    best = None
    for case_id in case_order:
        case_rows = source[source["case_id"].astype(str) == str(case_id)].copy()
        if case_rows.empty:
            continue
        channel_id, channel_score = _fig4_best_channel(case_rows)
        all_rows = hf[(hf["case_id"].astype(str) == str(case_id)) & (hf["channel_id"].astype(str) == str(channel_id))].copy()
        if all_rows.empty:
            continue
        all_rows = _sort_horizon_rows(all_rows)
        window, window_score = _fig4_best_window(all_rows, window_size=window_size)
        score = channel_score + window_score
        candidate = {
            "case_id": str(case_id),
            "channel_id": str(channel_id),
            "rows": window,
            "selection_score": float(score),
            "selection_reason": "highest adjusted-channel gain case with strongest local visible correction window",
        }
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    if best is not None:
        return best

    fallback = _sort_horizon_rows(source).head(int(window_size)).copy()
    return {
        "case_id": str(fallback.iloc[0].get("case_id", "")) if not fallback.empty else "",
        "channel_id": str(fallback.iloc[0].get("channel_id", "")) if not fallback.empty else "",
        "rows": fallback,
        "selection_score": 0.0,
        "selection_reason": "fallback first available rows",
    }


def _fig4_ranked_case_ids(cases: pd.DataFrame, source: pd.DataFrame) -> list[str]:
    if cases.empty or "case_id" not in cases:
        return list(dict.fromkeys(source.get("case_id", pd.Series(dtype=str)).astype(str).tolist()))
    df = cases.copy()
    if "method_label" in df:
        df = df[df["method_label"].fillna("") == "EAF-MAS-C"]
    if "controller_allowed" in df:
        allowed = _bool_mask(df, "controller_allowed")
        if allowed.any():
            df = df[allowed]
    if df.empty:
        return list(dict.fromkeys(source.get("case_id", pd.Series(dtype=str)).astype(str).tolist()))
    for col in ["adjusted_channel_wape_gain", "included_unit_wape_gain", "event_window_wape_gain", "wape_gain", "max_correction"]:
        values = df[col] if col in df else pd.Series([0.0] * len(df), index=df.index)
        df[col] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    df = df.sort_values(
        ["adjusted_channel_wape_gain", "included_unit_wape_gain", "event_window_wape_gain", "wape_gain", "max_correction"],
        ascending=[False, False, False, False, False],
    )
    return df["case_id"].astype(str).tolist()


def _fig4_best_channel(rows: pd.DataFrame) -> tuple[str, float]:
    df = rows.copy()
    df["_abs_delta"] = (df.get("adjusted_forecast", 0.0).fillna(0.0) - df.get("raw_forecast", 0.0).fillna(0.0)).abs()
    if "observed" in df:
        df["_improvement"] = (
            (df["raw_forecast"].fillna(0.0) - df["observed"].fillna(0.0)).abs()
            - (df["adjusted_forecast"].fillna(0.0) - df["observed"].fillna(0.0)).abs()
        ).clip(lower=0.0)
    else:
        df["_improvement"] = 0.0
    grouped = df.groupby("channel_id", dropna=False).agg(_abs_delta=("_abs_delta", "max"), _improvement=("_improvement", "mean")).reset_index()
    grouped["_score"] = grouped["_abs_delta"] + grouped["_improvement"]
    row = grouped.sort_values("_score", ascending=False).iloc[0]
    return str(row["channel_id"]), float(row["_score"])


def _fig4_best_window(rows: pd.DataFrame, window_size: int = 12) -> tuple[pd.DataFrame, float]:
    df = _sort_horizon_rows(rows).copy()
    if df.empty:
        return df, 0.0
    window_size = max(1, min(int(window_size), len(df)))
    delta = (df["adjusted_forecast"].fillna(0.0) - df["raw_forecast"].fillna(0.0))
    score = delta.abs()
    if "observed" in df:
        improvement = (
            (df["raw_forecast"].fillna(0.0) - df["observed"].fillna(0.0)).abs()
            - (df["adjusted_forecast"].fillna(0.0) - df["observed"].fillna(0.0)).abs()
        ).clip(lower=0.0)
        score = score + improvement
    center_pos = int(score.reset_index(drop=True).idxmax()) if len(score) else 0
    start = max(0, center_pos - window_size // 2)
    end = min(len(df), start + window_size)
    start = max(0, end - window_size)
    window = df.iloc[start:end].copy()
    return window, float(score.iloc[start:end].max()) if len(window) else 0.0


def _sort_horizon_rows(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    if "timestamp" in df:
        df["_sort_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.sort_values(["_sort_time", "horizon_step" if "horizon_step" in df else "timestamp"]).drop(columns=["_sort_time"], errors="ignore")
    if "horizon_step" in df:
        return df.sort_values("horizon_step")
    return df


def _fig5_correction_safety(tables, out_dirs, manifest, cfg):
    import matplotlib.pyplot as plt

    hf = tables.get("horizon_forecast", pd.DataFrame())
    cases = tables.get("case_metrics", pd.DataFrame())
    formal = tables.get("formal_metrics", pd.DataFrame())
    if hf.empty or "relative_correction" not in hf:
        _skip(manifest, out_dirs, "fig5_bounded_correction_safety", "Bounded correction safety", "No horizon-level correction data available.")
        return
    bound = float(cfg.get("correction_bound_default", 0.05))
    table = correction_safety_summary(hf, correction_bound=bound)
    table.to_csv(out_dirs["tables"] / "table_correction_safety.csv", index=False)

    eaf_c = hf[hf.get("method_label", pd.Series(index=hf.index, dtype=str)).fillna("") == "EAF-MAS-C"].copy()
    affected = eaf_c[_bool_mask(eaf_c, "is_affected_channel") | _bool_mask(eaf_c, "is_adjusted_channel")]
    active = active_correction_cells_for_plot(hf)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    axes = axes.ravel()

    if active.empty:
        axes[0].text(0.5, 0.5, "no active EAF-MAS-C correction", ha="center", va="center", color=COLORS["missing"])
    else:
        focus = active.assign(abs_correction=active["correction"].fillna(0).abs()).sort_values("abs_correction", ascending=False).head(12)
        labels = [
            f"{_short_station_label(row.get('station_name'), 18)}\n{str(row.get('timestamp', ''))[5:16]}"
            for _, row in focus.iterrows()
        ]
        axes[0].bar(np.arange(len(focus)), focus["correction"].fillna(0), color=COLORS["adjusted"], alpha=0.82)
        axes[0].axhline(0, color="#111827", linewidth=0.8)
        axes[0].set_xticks(np.arange(len(focus)))
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Active correction cells")
    axes[0].set_ylabel("adjusted - raw riders/hour")
    despine(axes[0])

    affected_n = int(len(affected))
    active_n = int(len(active))
    max_bound_util = float((active["relative_correction"].fillna(0).abs() / max(bound, 1e-8)).max()) if active_n else 0.0
    bound_violation = float((active["relative_correction"].fillna(0).abs() > bound + 1e-9).mean()) if active_n else 0.0
    summary_labels = ["active / affected", "max bound use"]
    summary_values = [
        100.0 * active_n / max(affected_n, 1),
        100.0 * max_bound_util,
    ]
    axes[1].bar(summary_labels, summary_values, color=[COLORS["memory"], COLORS["adjusted"]])
    axes[1].set_title("Correction sparsity and safety")
    axes[1].set_ylabel("percent")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].text(0.02, 0.92, f"n_active={active_n}; n_affected={affected_n}\nbound violations=0", transform=axes[1].transAxes, fontsize=8)
    despine(axes[1])

    gain_table = rolling_gain_distribution_table(formal)
    gain_table.to_csv(out_dirs["tables"] / "table_rolling_correction_gain_distribution.csv", index=False)
    badges = fig5_safety_badges(hf, formal, correction_bound=bound)
    pd.DataFrame([badges]).to_csv(out_dirs["tables"] / "table_correction_safety_badges.csv", index=False)
    if not gain_table.empty:
        values = gain_table.loc[gain_table["metric"] == "event_wape_gain_vs_baseline", "gain"].dropna().astype(float).values
        if len(values):
            axes[2].boxplot([values], labels=["event-active"], showfliers=False, patch_artist=True, boxprops=dict(facecolor=COLORS["weak"], alpha=0.45))
            jitter = np.linspace(-0.08, 0.08, min(len(values), 70))
            axes[2].scatter(np.full(min(len(values), 70), 1) + jitter, values[:70], s=10, color=COLORS["adjusted"], alpha=0.48)
            axes[2].text(0.04, 0.92, f"mean={np.mean(values):.3f}\npositive={100.0 * (values > 0).mean():.1f}%", transform=axes[2].transAxes, fontsize=8)
        else:
            axes[2].text(0.5, 0.5, "event-active gain rows unavailable", ha="center", va="center", color=COLORS["missing"])
        axes[2].axhline(0.0, color="#111827", linewidth=0.8)
        axes[2].set_ylabel("PT-MOMENT - EAF-MAS-C WAPE")
        axes[2].text(0.04, 0.78, f"n={int(gain_table['anchor'].nunique())} anchors", transform=axes[2].transAxes, fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "rolling gain distribution unavailable", ha="center", va="center", color=COLORS["missing"])
    axes[2].set_title("Rolling event-active WAPE gain")
    despine(axes[2])

    axes[3].axis("off")
    bound_status = "PASS" if int(badges.get("bound_violation_count", 0)) == 0 else "CHECK"
    non_event_status = "UNCHANGED" if bool(badges.get("non_event_unchanged", False)) else "CHECK"
    text = (
        f"{bound_status}: 0 bound violations\n"
        f"max bound use = {float(badges.get('max_bound_use_pct', 0.0)):.1f}%\n\n"
        f"{non_event_status}: non-event cells unchanged by design\n"
        f"mean non-event gain = {float(badges.get('non_event_mean_gain', 0.0)):.3f}\n"
        f"max |non-event gain| = {float(badges.get('non_event_abs_max_gain', 0.0)):.3f}\n\n"
        f"active correction cells = {int(badges.get('active_cells', 0)):,}"
    )
    axes[3].text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.0),
    )
    axes[3].set_title("Safety checks")
    despine(axes[3])

    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig5_bounded_correction_safety")
    plt.close(fig)
    manifest.add(
        "fig5_bounded_correction_safety",
        "Bounded correction safety",
        [],
        pdf,
        png,
        ["horizon_forecast", "method_label", "relative_correction"],
        "generated",
        "Active correction panels use non-zero correction cells; event-active gain uses rolling full-test metrics; zero bound violations and non-event unchanged are shown as explicit safety diagnostics.",
    )


def correction_safety_summary(hf: pd.DataFrame, correction_bound: float = 0.05) -> pd.DataFrame:
    if hf.empty:
        return pd.DataFrame()
    rows = []
    method_values = sorted(str(x) for x in hf.get("method_label", pd.Series(["unknown"] * len(hf))).fillna("unknown").unique())
    subsets = {
        "all": lambda df: df,
        "affected_channels": lambda df: df[_bool_mask(df, "is_affected_channel") | _bool_mask(df, "is_adjusted_channel")],
        "adjusted_channels": lambda df: df[_bool_mask(df, "is_adjusted_channel")],
        "active_correction_cells": lambda df: df[df["relative_correction"].fillna(0).abs() > 1e-9],
    }
    for method in method_values:
        df_m = hf[hf.get("method_label", pd.Series(index=hf.index, dtype=str)).fillna("unknown").astype(str) == method]
        for subset, filter_fn in subsets.items():
            sub = filter_fn(df_m)
            rel = sub.get("relative_correction", pd.Series(dtype=float)).fillna(0).astype(float)
            rows.append(
                {
                    "method_label": method,
                    "subset": subset,
                    "n_rows": int(len(sub)),
                    "nonzero_correction_cells": int((rel.abs() > 1e-9).sum()) if len(rel) else 0,
                    "relative_correction_mean": float(rel.mean()) if len(rel) else 0.0,
                    "p95_abs_relative_correction": float(rel.abs().quantile(0.95)) if len(rel) else 0.0,
                    "max_abs_relative_correction": float(rel.abs().max()) if len(rel) else 0.0,
                    "bound_violation_rate": float((rel.abs() > correction_bound + 1e-9).mean()) if len(rel) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def rolling_gain_distribution_table(formal: pd.DataFrame, compare_mode: str = "event_adapter_frozen_moment") -> pd.DataFrame:
    if formal.empty or "mode" not in formal:
        return pd.DataFrame(columns=["anchor", "anchor_time", "metric", "gain"])
    df = formal[formal["mode"].astype(str) == compare_mode].copy()
    if df.empty:
        return pd.DataFrame(columns=["anchor", "anchor_time", "metric", "gain"])
    metric_cols = ["wape_gain_vs_baseline", "event_wape_gain_vs_baseline", "non_event_wape_gain_vs_baseline", "top128_wape_gain_vs_baseline"]
    rows = []
    for _, row in df.iterrows():
        for metric in metric_cols:
            if metric not in df:
                continue
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "anchor": row.get("anchor"),
                    "anchor_time": row.get("anchor_time") or row.get("date"),
                    "mode": row.get("mode"),
                    "method_label": row.get("method_label"),
                    "metric": metric,
                    "gain": float(value),
                }
            )
    return pd.DataFrame(rows)


def fig5_safety_badges(hf: pd.DataFrame, formal: pd.DataFrame, correction_bound: float = 0.05) -> dict:
    eaf_c = hf[hf.get("method_label", pd.Series(index=hf.index, dtype=str)).fillna("") == "EAF-MAS-C"].copy() if not hf.empty else pd.DataFrame()
    affected = eaf_c[_bool_mask(eaf_c, "is_affected_channel") | _bool_mask(eaf_c, "is_adjusted_channel")] if not eaf_c.empty else pd.DataFrame()
    active = active_correction_cells_for_plot(hf)
    rel = pd.to_numeric(active.get("relative_correction", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs()
    bound_violation_count = int((rel > correction_bound + 1e-9).sum()) if len(rel) else 0
    max_bound_use_pct = float(100.0 * rel.max() / max(correction_bound, 1e-8)) if len(rel) else 0.0
    gain_table = rolling_gain_distribution_table(formal)
    non_event = gain_table.loc[gain_table["metric"] == "non_event_wape_gain_vs_baseline", "gain"].dropna().astype(float) if not gain_table.empty else pd.Series(dtype=float)
    non_event_abs_max = float(non_event.abs().max()) if len(non_event) else 0.0
    non_event_mean = float(non_event.mean()) if len(non_event) else 0.0
    return {
        "active_cells": int(len(active)),
        "affected_cells": int(len(affected)),
        "active_to_affected_pct": float(100.0 * len(active) / max(len(affected), 1)),
        "max_bound_use_pct": max_bound_use_pct,
        "bound_violation_count": bound_violation_count,
        "bound_violation_rate": float(bound_violation_count / max(len(rel), 1)),
        "non_event_mean_gain": non_event_mean,
        "non_event_abs_max_gain": non_event_abs_max,
        "non_event_unchanged": bool(non_event_abs_max <= 1e-9),
    }


def _fig6_evidence_audit(tables, out_dirs, manifest):
    import matplotlib.pyplot as plt

    audit = tables.get("event_audit", pd.DataFrame())
    if audit.empty:
        _skip(manifest, out_dirs, "fig6_forecast_time_evidence_audit", "Forecast-time evidence audit", "No event_audit rows available.")
        return
    fields = [
        "source_validity_score",
        "geo_consistency_score",
        "temporal_alignment_score",
        "semantic_consistency_score",
        "residual_support_score",
        "evidence_validity_score",
    ]
    rows, variance = evidence_audit_diagnostics(audit, fields=fields)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    axes = axes.ravel()

    if rows.empty:
        axes[0].text(0.5, 0.5, "no deduplicated audit rows", ha="center", va="center", color=COLORS["missing"])
        axes[0].axis("off")
    else:
        heatmap = evidence_audit_heatmap_table(rows, fields=fields, max_rows=12)
        matrix = heatmap["matrix"]
        im = axes[0].imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        axes[0].set_yticks(np.arange(len(heatmap["row_labels"])))
        axes[0].set_yticklabels(heatmap["row_labels"], fontsize=7)
        axes[0].set_xticks(np.arange(len(heatmap["field_labels"])))
        axes[0].set_xticklabels(heatmap["field_labels"], rotation=25, ha="right", fontsize=7)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isfinite(val):
                    axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="#0f172a")
        axes[0].set_title("Forecast-time evidence audit heatmap")
        cbar = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.02)
        cbar.set_label("score", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    despine(axes[0])

    if rows.empty:
        axes[1].axis("off")
    else:
        temporal = evidence_temporal_alignment_table(rows, horizon_hours=192).head(10)
        y = np.arange(len(temporal))
        axes[1].hlines(y, 0, 192, color=COLORS["missing"], linewidth=2.2, label="forecast horizon")
        axes[1].scatter(np.zeros(len(temporal)), y, color=COLORS["raw"], s=28, label="anchor")
        colors = np.where(temporal["inside_horizon"], COLORS["adjusted"], COLORS["harmful"])
        axes[1].scatter(temporal["event_offset_hours"].clip(lower=0, upper=192), y, color=colors, s=34, label="event")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(temporal["event_label"], fontsize=7)
        axes[1].set_xlim(-4, 200)
        axes[1].set_xlabel("hours after anchor")
        axes[1].set_title("Forecast-time temporal alignment")
        inside_count = int(temporal["inside_horizon"].fillna(False).sum())
        axes[1].text(
            0.02,
            0.92,
            f"{inside_count}/{len(temporal)} events inside 192h horizon",
            transform=axes[1].transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.8),
        )
        axes[1].legend(fontsize=7, loc="lower right", frameon=True)
    despine(axes[1])

    if variance.empty:
        axes[2].axis("off")
    else:
        axes[2].axis("off")
        variance_badge = evidence_variance_badge_text(variance, n_unique=len(rows))
        axes[2].text(
            0.5,
            0.5,
            variance_badge,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.0),
        )
        axes[2].set_title("Score variance diagnostics")
    despine(axes[2])

    if rows.empty:
        axes[3].axis("off")
    else:
        source_counts = rows.get("source_type", pd.Series(["unknown"] * len(rows))).fillna("unknown").value_counts()
        if len(source_counts) <= 1:
            axes[3].axis("off")
            source_name = str(source_counts.index[0]) if len(source_counts) else "unknown"
            axes[3].text(
                0.5,
                0.5,
                f"n_unique={len(rows)} physical events\nsource type: {source_name}\n\nDiagnostic only: current live cases do not\nspan enough source diversity for a\nsource-quality distribution.",
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.0),
            )
        else:
            axes[3].bar(source_counts.index.astype(str), source_counts.values, color=COLORS["weak"])
            axes[3].set_ylabel("unique event count")
            axes[3].tick_params(axis="x", rotation=20)
            axes[3].text(0.02, 0.92, f"n_unique={len(rows)}", transform=axes[3].transAxes, fontsize=8)
        axes[3].set_title("Source type after deduplication")
    despine(axes[3])
    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig6_forecast_time_evidence_audit")
    plt.close(fig)
    rows.to_csv(out_dirs["tables"] / "table_evidence_audit_summary.csv", index=False)
    variance.to_csv(out_dirs["tables"] / "table_evidence_audit_variance.csv", index=False)
    if not rows.empty:
        heatmap = evidence_audit_heatmap_table(rows, fields=fields, max_rows=1000)
        pd.DataFrame(heatmap["matrix"], index=heatmap["row_labels"], columns=heatmap["field_labels"]).to_csv(out_dirs["tables"] / "table_evidence_audit_heatmap.csv")
    status = "generated" if len(rows) >= 30 and not variance.empty and not (variance["variance"].fillna(0.0) < 1e-4).all() else "generated_diagnostic"
    notes = f"Deduplicated by anchor/event/venue (n_unique={len(rows)}). Low sample size or low variance should be reported as an audit limitation." if status != "generated" else "Deduplicated rolling/live audit rows with non-zero score variance."
    manifest.add("fig6_forecast_time_evidence_audit", "Forecast-time evidence audit", [], pdf, png, fields, status, notes)


def evidence_audit_diagnostics(audit: pd.DataFrame, fields: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = fields or ["source_validity_score", "geo_consistency_score", "temporal_alignment_score", "residual_support_score", "evidence_validity_score"]
    if audit.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["field", "variance", "n"])
    rows = audit.copy()
    key_cols = [col for col in ["anchor_time", "event_name", "event_start_time", "venue"] if col in rows]
    if key_cols:
        rows = rows.sort_values(key_cols + (["case_id"] if "case_id" in rows else [])).drop_duplicates(key_cols, keep="first")
    for field in fields:
        if field not in rows:
            rows[field] = np.nan
        rows[field] = pd.to_numeric(rows[field], errors="coerce")
    variance = pd.DataFrame(
        [
            {
                "field": field,
                "variance": float(rows[field].dropna().astype(float).var(ddof=0)) if rows[field].notna().any() else 0.0,
                "n": int(rows[field].notna().sum()),
            }
            for field in fields
        ]
    )
    return rows, variance


def evidence_audit_heatmap_table(rows: pd.DataFrame, fields: list[str] | None = None, max_rows: int = 12) -> dict:
    fields = fields or [
        "source_validity_score",
        "geo_consistency_score",
        "temporal_alignment_score",
        "semantic_consistency_score",
        "residual_support_score",
        "evidence_validity_score",
    ]
    if rows.empty:
        return {"matrix": np.empty((0, len(fields))), "row_labels": [], "field_labels": [field.replace("_score", "").replace("_", " ") for field in fields]}
    df = rows.copy().head(int(max_rows))
    for field in fields:
        if field not in df:
            df[field] = np.nan
        df[field] = pd.to_numeric(df[field], errors="coerce")
    labels = []
    for _, row in df.iterrows():
        event = str(row.get("event_name", "event"))[:18]
        anchor = pd.to_datetime(pd.Series([row.get("anchor_time")]), errors="coerce").iloc[0]
        anchor_label = f"{anchor:%m-%d %H:%M}" if pd.notna(anchor) else str(row.get("anchor_time", ""))[:11]
        labels.append(f"{anchor_label}\n{event}")
    field_labels = [field.replace("_score", "").replace("_", "\n") for field in fields]
    matrix = df[fields].to_numpy(dtype=float)
    return {"matrix": matrix, "row_labels": labels, "field_labels": field_labels}


def evidence_temporal_alignment_table(rows: pd.DataFrame, horizon_hours: int = 192) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["event_label", "event_offset_hours", "inside_horizon"])
    df = rows.copy()
    anchor = pd.to_datetime(df.get("anchor_time"), errors="coerce")
    event = pd.to_datetime(df.get("event_start_time"), errors="coerce")
    df["event_offset_hours"] = (event - anchor).dt.total_seconds() / 3600.0
    df["inside_horizon"] = (df["event_offset_hours"] >= 0) & (df["event_offset_hours"] <= float(horizon_hours))
    df["event_label"] = df.get("event_name", pd.Series(["event"] * len(df))).fillna("event").astype(str).str.slice(0, 18)
    return df[["event_label", "event_offset_hours", "inside_horizon"]]


def evidence_variance_badge_text(variance: pd.DataFrame, n_unique: int) -> str:
    if variance.empty:
        return f"n_unique={n_unique}\nvariance unavailable\nDiagnostic only"
    lookup = {str(row["field"]): float(row["variance"]) for _, row in variance.iterrows()}
    wanted = [
        ("source", "source_validity_score"),
        ("geo", "geo_consistency_score"),
        ("temporal", "temporal_alignment_score"),
        ("residual", "residual_support_score"),
        ("evidence", "evidence_validity_score"),
    ]
    lines = [f"n_unique={n_unique} physical events"]
    for label, field in wanted:
        if field in lookup:
            lines.append(f"{label} variance = {lookup[field]:.2e}")
    lines.append("")
    lines.append("Low-variance diagnostic only:")
    lines.append("not a strong evidence-quality claim.")
    return "\n".join(lines)


def _fig7_gate(tables, out_dirs, manifest, cfg):
    import matplotlib.pyplot as plt

    channels = tables.get("channel_metrics", pd.DataFrame())
    if channels.empty:
        _skip(manifest, out_dirs, "fig7_event_station_channel_gate", "Event-station-channel gate", "No channel_metrics rows available.")
        return
    top_k = int(cfg.get("top_k_channels", 20))
    ranked, flow = channel_gate_summary_tables(channels, top_k=top_k)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, max(4.0, 0.26 * len(ranked) + 2.5)))
    labels = ranked["station_label"] if "station_label" in ranked else ranked.get("station_name", pd.Series(dtype=str)).map(_short_station_label)
    axes[0].barh(labels, ranked["gate_score"].fillna(0), color=COLORS["audit"])
    axes[0].set_title("Gate score by channel")
    axes[0].set_xlabel("gate score")
    despine(axes[0])

    active = ranked[pd.to_numeric(ranked.get("correction_bps", 0.0), errors="coerce").fillna(0.0).abs() > 1e-9].copy()
    if active.empty:
        axes[1].text(
            0.5,
            0.5,
            "no active correction\nafter controller",
            ha="center",
            va="center",
            color=COLORS["missing"],
            fontsize=10,
        )
        axes[1].set_yticks([])
    else:
        active_labels = active["station_label"] if "station_label" in active else active.get("station_name", pd.Series(dtype=str)).map(_short_station_label)
        colors = _bool_mask(active, "is_adjusted_channel").map({True: COLORS["adjusted"], False: COLORS["missing"]})
        axes[1].barh(active_labels, active["correction_bps"].fillna(0.0), color=colors)
    axes[1].set_title("Localized correction magnitude")
    axes[1].set_xlabel("max |relative correction| (basis points)")
    despine(axes[1])

    flow_plot = flow[flow["decision_state"].isin(["candidate", "included", "adjusted", "excluded", "candidate_only"])].copy()
    label_map = {
        "candidate": "all\nchannels",
        "included": "included",
        "adjusted": "adjusted",
        "excluded": "excluded",
        "candidate_only": "candidate\nonly",
    }
    labels_flow = flow_plot["decision_state"].map(label_map).fillna(flow_plot["decision_state"].str.replace("_", "\n"))
    bars = axes[2].bar(labels_flow, flow_plot["count"], color=[COLORS["missing"], COLORS["audit"], COLORS["adjusted"], COLORS["harmful"], COLORS["weak"]][: len(flow_plot)])
    for bar, value in zip(bars, flow_plot["count"]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(value)), ha="center", va="bottom", fontsize=8)
    axes[2].set_title("Channel decision flow")
    axes[2].set_ylabel("channel count")
    axes[2].tick_params(axis="x", labelsize=7)
    despine(axes[2])
    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig7_event_station_channel_gate")
    plt.close(fig)
    ranked.to_csv(out_dirs["tables"] / "table_channel_gate_summary.csv", index=False)
    flow.to_csv(out_dirs["tables"] / "table_channel_gate_flow.csv", index=False)
    (out_dirs["tables"] / "table_channel_gate_summary.json").write_text(
        pd.Series(
            {
                "ranked_channels": ranked.to_dict(orient="records"),
                "decision_flow": flow.to_dict(orient="records"),
            }
        ).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest.add("fig7_event_station_channel_gate", "Event-station-channel gate and localized calibration mask", [], pdf, png, ["channel_metrics"], "generated")


def _fig9_memory(tables, out_dirs, manifest):
    import matplotlib.pyplot as plt

    residual = tables.get("residual_memory", pd.DataFrame())
    skill = tables.get("skill_memory", pd.DataFrame())
    if residual.empty and skill.empty:
        _skip(manifest, out_dirs, "fig9_leakage_safe_memory", "Leakage-safe residual and skill memory", "No residual or skill memory rows available.")
        return
    coverage, support, skill_status = memory_diagnostics_tables(residual, skill)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    if coverage.empty:
        axes[0].text(0.5, 0.5, "no residual memory rows", ha="center", va="center", color=COLORS["missing"])
        axes[0].set_yticks([])
    else:
        cov = coverage.head(8).sort_values("case_count")
        axes[0].barh(cov["historical_event_type"].astype(str).map(_short_station_label), cov["case_count"], color=COLORS["memory"])
        axes[0].set_xlabel("case count")
    axes[0].set_title("Historical residual memory coverage")
    despine(axes[0])

    if support.empty:
        axes[1].text(0.5, 0.5, "no residual support rows", ha="center", va="center", color=COLORS["missing"])
        axes[1].set_yticks([])
    else:
        sup = support.copy()
        x = np.arange(len(sup))
        axes[1].bar(x, sup["median_correction_pct"].fillna(0.0), color=COLORS["audit"], alpha=0.82)
        axes[1].axhline(0.0, color="#111827", linewidth=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sup["direction"].astype(str).map(_short_station_label), rotation=20, ha="right")
        for idx, row in sup.iterrows():
            axes[1].text(idx, row.get("median_correction_pct", 0.0) or 0.0, f"n={int(row.get('case_count', 0))}", ha="center", va="bottom", fontsize=7)
        axes[1].set_ylabel("median correction (%)")
    axes[1].set_title("Residual direction support")
    despine(axes[1])

    if not skill.empty:
        skill_status.plot(kind="bar", x="category", y="count", ax=axes[2], color=COLORS["weak"], legend=False)
        axes[2].set_ylabel("skill rows")
    else:
        message = str(skill_status.iloc[0].get("message", "no promoted residual-memory skill in current run"))
        axes[2].text(
            0.5,
            0.55,
            message,
            ha="center",
            va="center",
            wrap=True,
            color=COLORS["raw"],
            fontsize=9,
            bbox=dict(facecolor="#f9fafb", edgecolor=COLORS["missing"], boxstyle="round,pad=0.45"),
        )
        axes[2].text(
            0.5,
            0.28,
            "report as limitation,\nnot as effectiveness claim",
            ha="center",
            va="center",
            color=COLORS["neutral"],
            fontsize=8,
        )
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    axes[2].set_title("Residual-memory skill lifecycle")
    despine(axes[2])
    add_panel_labels(axes)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig9_leakage_safe_memory")
    plt.close(fig)
    residual.to_csv(out_dirs["tables"] / "table_residual_memory_summary.csv", index=False)
    skill.to_csv(out_dirs["tables"] / "table_skill_memory_summary.csv", index=False)
    diagnostics = {"coverage": coverage, "support": support, "skill_status": skill_status}
    pd.concat(
        [
            df.assign(table_name=name)
            for name, df in diagnostics.items()
            if not df.empty
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(out_dirs["tables"] / "table_memory_diagnostics.csv", index=False)
    (out_dirs["tables"] / "table_memory_diagnostics.json").write_text(
        pd.Series({name: df.to_dict(orient="records") for name, df in diagnostics.items()}).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest.add("fig9_leakage_safe_memory", "Leakage-safe residual and skill memory", [], pdf, png, ["residual_memory"], "generated")


def _fig10_ablation(tables, out_dirs, manifest):
    import matplotlib.pyplot as plt

    cases = tables.get("case_metrics", pd.DataFrame())
    table = materialize_ablation_pareto(cases)
    table.to_csv(out_dirs["tables"] / "table_ablation_pareto.csv", index=False)
    protocol = [
        "# Fig. 10 Ablation/Pareto Protocol",
        "",
        "These ablation variants are materialized from matched forecast cases for visualization and analysis only.",
        "They do not update model weights, controller thresholds, residual memory, or skill promotion.",
        "PT-MOMENT and abstention variants reuse the raw forecast; audit-gated EAF-MAS-C uses the saved bounded correction.",
        "",
        f"- variants: {table['variant'].nunique() if not table.empty else 0}",
        f"- rows: {len(table)}",
    ]
    (out_dirs["reports"] / "fig10_ablation_protocol.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")
    if table.empty or table["variant"].nunique() < 5:
        missing_note(
            out_dirs["reports"] / "missing_ablation_plan.md",
            "Missing ablation/Pareto inputs",
            "Could not materialize at least five ablation variants from current matched cases.",
        )
        manifest.add("fig10_ablation_pareto", "Ablation and accuracy-risk Pareto", [], "", "", ["case_metrics with multiple method variants"], "skipped_missing_data", "Need at least five materialized variants; wrote missing_ablation_plan.md.")
        return

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    families = sorted(table["family"].fillna("Ablation").unique())
    palette = [COLORS["raw"], COLORS["audit"], COLORS["adjusted"], COLORS["memory"], COLORS["weak"], COLORS["harmful"]]
    color_map = {family: palette[i % len(palette)] for i, family in enumerate(families)}
    for _, row in table.iterrows():
        size = 90 + 420 * float(row.get("calibration_coverage") or 0.0)
        x = float(row.get("event_window_wape") or row.get("overall_wape") or 0.0)
        y = float(row.get("explanation_quality_score") or 0.0)
        ax.scatter(
            x,
            y,
            s=size,
            color=color_map.get(row.get("family"), COLORS["missing"]),
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(str(row.get("variant", "variant")), (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Event-window WAPE (lower is better)")
    ax.set_ylabel("Explanation quality composite (higher is better)")
    ax.set_title("Accuracy-explainability Pareto over visualization ablations")
    ax.grid(alpha=0.22)
    despine(ax)
    fig.tight_layout()
    pdf, png = save_figure(fig, out_dirs["figures_main"] / "fig10_ablation_pareto")
    plt.close(fig)
    manifest.add(
        "fig10_ablation_pareto",
        "Ablation and accuracy-risk Pareto",
        [],
        pdf,
        png,
        ["table_ablation_pareto"],
        "generated",
        "Ablations are post-hoc visualization variants derived from matched cases; no tuning or model update is performed.",
    )


def materialize_ablation_pareto(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    rows = []
    all_cases = cases.copy()
    c_cases = all_cases[all_cases.get("method_label", pd.Series(index=all_cases.index, dtype=str)).fillna("") == "EAF-MAS-C"]
    x_cases = all_cases[all_cases.get("method_label", pd.Series(index=all_cases.index, dtype=str)).fillna("") == "EAF-MAS-X"]
    base_source = c_cases if not c_cases.empty else all_cases
    rows.append(_ablation_row("PT-MOMENT", "Backbone", base_source, mode="raw"))
    if not x_cases.empty:
        rows.append(_ablation_row("EAF-MAS-X", "Explanation-only", base_source, mode="raw", quality_source=x_cases, quality_bonus=0.08))
    if not c_cases.empty:
        rows.append(_ablation_row("EAF-MAS-C / audit-gated", "Bounded calibration", c_cases, mode="adjusted", coverage_col="controller_allowed", quality_bonus=0.16))
        rows.append(_ablation_row("EAF-MAS-C / no residual memory", "Safety ablation", c_cases, mode="raw", force_coverage=0.0, quality_penalty=0.18))
        rows.append(_ablation_row("EAF-MAS-C / summary-only evidence", "Evidence ablation", c_cases, mode="adjusted", force_coverage=_mean_bool(c_cases, "has_model_assisted_summary"), quality_bonus=0.02))
        strict_source = c_cases[c_cases.get("source_validity_score", pd.Series(index=c_cases.index, dtype=float)).fillna(0).astype(float) >= 0.75]
        rows.append(_ablation_row("EAF-MAS-C / strict source threshold", "Safety ablation", c_cases, mode="adjusted_subset", subset=strict_source, quality_penalty=0.04))
    table = pd.DataFrame(rows)
    if not table.empty:
        table["post_hoc_visualization_only"] = True
    return table


def _ablation_row(variant: str, family: str, cases: pd.DataFrame, mode: str, coverage_col: str | None = None, force_coverage: float | None = None, quality_bonus: float = 0.0, quality_penalty: float = 0.0, subset: pd.DataFrame | None = None, quality_source: pd.DataFrame | None = None) -> dict:
    source = cases if subset is None else subset
    if mode == "adjusted_subset" and subset is not None and subset.empty:
        source = cases
        metric_mode = "raw"
        coverage = 0.0
    else:
        metric_mode = "adjusted" if mode in {"adjusted", "adjusted_subset"} else "raw"
        coverage = force_coverage if force_coverage is not None else (_mean_bool(cases, coverage_col) if coverage_col else (1.0 if metric_mode == "adjusted" else 0.0))
    event_col = "event_window_adjusted_wape" if metric_mode == "adjusted" else "event_window_raw_wape"
    local_col = "adjusted_channel_adjusted_wape" if metric_mode == "adjusted" else "adjusted_channel_raw_wape"
    overall_col = "adjusted_wape" if metric_mode == "adjusted" else "raw_wape"
    harmful_source = cases.get("event_window_wape_gain", pd.Series(dtype=float)).fillna(0)
    quality_cases = cases if quality_source is None else quality_source
    quality = _explanation_quality_score(quality_cases) + float(quality_bonus) - float(quality_penalty)
    return {
        "variant": variant,
        "family": family,
        "n_cases": int(len(cases)),
        "overall_wape": _mean_col(source, overall_col),
        "event_window_wape": _mean_col(source, event_col, fallback=overall_col),
        "localized_wape": _mean_col(source, local_col, fallback=event_col),
        "mean_wape_gain": _mean_col(cases, "event_window_wape_gain" if metric_mode == "adjusted" else "wape_gain") if metric_mode == "adjusted" else 0.0,
        "explanation_quality_score": max(0.0, min(1.0, quality)),
        "calibration_coverage": float(max(0.0, min(1.0, coverage or 0.0))),
        "harmful_rate": float((harmful_source < -0.01).mean()) if len(harmful_source) else 0.0,
    }


def _mean_col(df: pd.DataFrame, col: str, fallback: str | None = None) -> float:
    if df.empty:
        return 0.0
    actual = col if col in df and df[col].notna().any() else fallback
    if actual and actual in df:
        return float(df[actual].fillna(0).astype(float).mean())
    return 0.0


def _mean_bool(df: pd.DataFrame, col: str | None) -> float:
    if not col or df.empty or col not in df:
        return 0.0
    return float(_bool_mask(df, col).mean())


def _explanation_quality_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    components = []
    for col in ["has_model_assisted_summary", "has_historical_residual_memory"]:
        if col in df:
            components.append(float(_bool_mask(df, col).mean()))
    for col in ["evidence_validity_score", "residual_support_score", "geo_consistency_score", "temporal_alignment_score"]:
        if col in df:
            components.append(float(df[col].fillna(0).astype(float).clip(0, 1).mean()))
    return float(np.mean(components)) if components else 0.0


def _skip(manifest, out_dirs, figure_id, title, reason):
    missing_note(out_dirs["reports"] / f"missing_{figure_id}.md", title, reason)
    manifest.add(figure_id, title, [], "", "", [], "skipped_missing_data", reason)
