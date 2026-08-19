"""Structured case-card figures for event-aware forecast explanations."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .plotting_utils import FigureManifest, missing_note, save_figure
from .schema import method_label
from .style import COLORS, apply_style


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _first_json(paths: Iterable[Path]) -> dict:
    for path in paths:
        if path.is_file():
            return _read_json(path)
    return {}


def _case_dir_from_input(input_file: str) -> Path:
    path = Path(str(input_file))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.parent.parent


def build_case_card_record(case_dir: str | Path, card_title: str) -> dict:
    case_dir = Path(case_dir)
    prediction = _first_json(sorted((case_dir / "predictions").glob("*.json")))
    audit = _read_json(case_dir / "reports" / "evidence_audit.json")
    controller = _read_json(case_dir / "reports" / "calibration_controller.json")
    llm = _first_json(sorted((case_dir / "logs").glob("llm_explanation_*.json")))
    request = prediction.get("request") or {}
    decision = prediction.get("decision") or {}
    evidence = prediction.get("evidence") or {}
    result = llm.get("result") or {}
    included = decision.get("included_units") or controller.get("included_units") or audit.get("included_units") or []
    excluded = decision.get("excluded_units") or controller.get("excluded_units") or audit.get("excluded_units") or []
    historical = evidence.get("historical_event_cases") or []
    raw = prediction.get("raw_metrics") or {}
    adjusted = prediction.get("metrics") or {}
    correction_stats = decision.get("correction_stats") or controller.get("correction_stats") or {}
    mode = str(request.get("mode") or "")
    multi_hop = str(result.get("multi_hop_reasoning") or result.get("markdown") or "").strip()
    if not multi_hop:
        multi_hop = "No structured LLM multi-hop reasoning was recorded."
    return {
        "card_title": card_title,
        "case_dir": _display_case_dir(case_dir),
        "anchor_time": request.get("date"),
        "horizon": request.get("horizon"),
        "station_scope": request.get("station_scope"),
        "mode": mode,
        "method_label": method_label(mode),
        "raw_wape": raw.get("wape"),
        "adjusted_wape": adjusted.get("wape"),
        "source_validity_score": audit.get("source_validity_score", (evidence.get("evidence_audit") or {}).get("source_validity_score")),
        "geo_consistency_score": audit.get("geo_consistency_score", (evidence.get("evidence_audit") or {}).get("geo_consistency_score")),
        "temporal_alignment_score": audit.get("temporal_alignment_score", (evidence.get("evidence_audit") or {}).get("temporal_alignment_score")),
        "residual_support_score": audit.get("residual_support_score", (evidence.get("evidence_audit") or {}).get("residual_support_score")),
        "evidence_validity_score": audit.get("evidence_validity_score", (evidence.get("evidence_audit") or {}).get("evidence_validity_score")),
        "controller_allowed": bool(decision.get("controller_allowed", controller.get("controller_allowed", False))),
        "calibration_decision": "apply" if bool(decision.get("controller_allowed", controller.get("controller_allowed", False))) else "abstain",
        "correction_bound": decision.get("correction_bound") or correction_stats.get("correction_bound"),
        "max_abs_correction": correction_stats.get("max_abs_correction"),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "historical_memory_count": len(historical),
        "top_included": _summarize_units(included, "station_channel", "reason"),
        "top_excluded": _summarize_units(excluded, "station_channel", "exclusion_reason"),
        "top_memory": _paper_label_text(_summarize_memory(historical)),
        "multi_hop_reasoning": _paper_label_text(multi_hop),
        "calibration_rationale": _paper_label_text(result.get("calibration_rationale") or ""),
        "uncertainty": _paper_label_text(result.get("uncertainty_and_abstention") or ""),
    }


def _summarize_units(rows: List[dict], key: str, reason_key: str, limit: int = 3) -> str:
    if not rows:
        return "none"
    parts = []
    for row in rows[:limit]:
        parts.append(f"{row.get(key, 'unit')}: {row.get(reason_key, row.get('relation', 'selected'))}")
    return "; ".join(parts)


def _summarize_memory(rows: List[dict], limit: int = 3) -> str:
    if not rows:
        return "No train/validation historical residual memory attached."
    parts = []
    for row in rows[:limit]:
        name = row.get("historical_event_title") or row.get("event") or row.get("title") or "historical event"
        direction = row.get("residual_direction") or row.get("direction") or "unknown direction"
        correction = row.get("median_lp_moment_correction") or row.get("median_correction") or "n/a"
        split = row.get("split") or "train/val"
        parts.append(f"{name} ({split}, {direction}, median {correction})")
    return "; ".join(parts)


def _display_case_dir(case_dir: Path) -> str:
    try:
        return str(case_dir.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return case_dir.name


def _paper_label_text(text: object) -> str:
    value = str(text or "")
    return value.replace("LP-MOMENT", "PT-MOMENT").replace("LP MOMENT", "PT-MOMENT")


def _fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value or "n/a")


def _wrap(text: object, width: int = 54, max_lines: int = 8) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if not value:
        return "n/a"
    lines = textwrap.wrap(value, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + " ..."
    return "\n".join(lines)


def make_case_card_figures(tables: Dict[str, pd.DataFrame], out_dirs: dict) -> FigureManifest:
    apply_style()
    manifest = FigureManifest()
    cases = tables.get("case_metrics", pd.DataFrame())
    if cases.empty or "input_file" not in cases:
        missing_note(out_dirs["reports"] / "missing_case_cards.md", "Missing case cards", "No case_metrics input_file rows available.")
        manifest.add("fig11_case_card_intervention", "Bounded intervention case card", [], "", "", ["case_metrics.input_file"], "skipped_missing_data", "No cases available.")
        return manifest
    records = []
    selections = _select_case_card_rows(cases)
    appendix_dir = out_dirs["figures_appendix"] / "case_cards"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    for fig_id, title, row, main in selections:
        if row is None:
            manifest.add(fig_id, title, [], "", "", ["case_metrics"], "skipped_missing_data", "Required case type not found.")
            continue
        record = build_case_card_record(_case_dir_from_input(row["input_file"]), title)
        records.append(record)
        out_base = out_dirs["figures_main"] / fig_id if main else appendix_dir / fig_id
        pdf, png = _draw_case_card(record, out_base)
        manifest.add(fig_id, title, [row["input_file"]], pdf, png, ["prediction_json", "evidence_audit", "llm_log"], "generated")
    if records:
        pd.DataFrame(records).to_csv(out_dirs["tables"] / "table_case_card_metrics.csv", index=False)
    return manifest


def _select_case_card_rows(cases: pd.DataFrame):
    c_cases = cases[cases.get("method_label", pd.Series(index=cases.index, dtype=str)).fillna("") == "EAF-MAS-C"]
    x_cases = cases[cases.get("method_label", pd.Series(index=cases.index, dtype=str)).fillna("") == "EAF-MAS-X"]
    intervention = _best_row(c_cases, "adjusted_channel_wape_gain", largest=True)
    abstention = _best_row(x_cases[x_cases.get("calibration_decision", pd.Series(index=x_cases.index, dtype=str)).fillna("") == "abstain"], "evidence_validity_score", largest=True)
    risk = _best_row(c_cases, "event_window_wape_gain", largest=False)
    memory = _best_row(c_cases[c_cases.get("has_historical_residual_memory", pd.Series(index=c_cases.index, dtype=bool)).fillna(False).astype(bool)], "included_unit_wape_gain", largest=True)
    return [
        ("fig11_case_card_intervention", "Bounded Event Intervention", intervention, True),
        ("fig12_case_card_abstention", "Explanation-only Abstention", abstention, True),
        ("figA3_case_card_risk", "Weak Event-window Harm / Risk Case", risk, False),
        ("figA4_case_card_memory", "Historical Memory Supported Correction", memory, False),
    ]


def _best_row(df: pd.DataFrame, col: str, largest: bool = True):
    if df.empty:
        return None
    if col not in df:
        return df.iloc[0]
    ranked = df.copy()
    ranked[col] = ranked[col].fillna(0).astype(float)
    return ranked.sort_values(col, ascending=not largest).iloc[0]


def _draw_case_card(record: dict, out_base: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 6.9))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    fig.suptitle(record["card_title"], x=0.03, y=0.985, ha="left", fontsize=15, fontweight="bold")
    _box(axes[0], "Forecast Request", f"Method: {record['method_label']}\nAnchor: {record['anchor_time']}\nHorizon: {record['horizon']} h\nScope: {record['station_scope']}", COLORS["raw"])
    _box(axes[1], "Evidence Audit", f"Source {_fmt(record.get('source_validity_score'), 2)} | Geo {_fmt(record.get('geo_consistency_score'), 2)}\nTemporal {_fmt(record.get('temporal_alignment_score'), 2)} | Residual {_fmt(record.get('residual_support_score'), 2)}\nOverall evidence {_fmt(record.get('evidence_validity_score'), 2)}", COLORS["audit"])
    _box(axes[2], "Event-Station-Channel Relevance", f"Included units: {record['included_count']}\n{_wrap(record['top_included'], 48, 4)}\n\nExcluded units: {record['excluded_count']}\n{_wrap(record['top_excluded'], 48, 3)}", COLORS["adjusted"])
    _box(axes[3], "Historical Residual Memory", f"Matched cases: {record['historical_memory_count']}\n{_wrap(record['top_memory'], 54, 7)}", COLORS["memory"])
    decision = "allowed" if record.get("controller_allowed") else "abstained"
    _box(axes[4], "Calibration / Abstention Decision", f"Decision: {decision}\nBound: {_fmt(record.get('correction_bound'))}\nMax correction: {_fmt(record.get('max_abs_correction'))}\nRaw WAPE: {_fmt(record.get('raw_wape'))}\nAdjusted WAPE: {_fmt(record.get('adjusted_wape'))}", COLORS["weak"])
    _box(axes[5], "LLM Multi-hop Reasoning", f"{_wrap(record['multi_hop_reasoning'], 58, 8)}\n\nUncertainty: {_wrap(record['uncertainty'], 58, 3)}", COLORS["pass"])
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.35, w_pad=1.1)
    return save_figure(fig, out_base)


def _box(ax, title: str, body: str, color: str) -> None:
    ax.text(0.0, 0.98, title, va="top", fontsize=10.5, fontweight="bold", color="#111827")
    ax.text(
        0.0,
        0.86,
        body,
        va="top",
        fontsize=8.1,
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor=color, linewidth=1.6),
    )
