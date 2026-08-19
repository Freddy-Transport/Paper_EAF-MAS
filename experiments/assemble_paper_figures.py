#!/usr/bin/env python3
"""Assemble paper-ready figures for the CCF-A/TRC draft.

This script intentionally separates full-split statistical evidence from
case-study material. It does not modify raw experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "blue": "#2563eb",
    "green": "#16a34a",
    "orange": "#f97316",
    "purple": "#7c3aed",
    "red": "#dc2626",
    "gray": "#64748b",
    "light": "#f8fafc",
    "dark": "#0f172a",
    "border": "#cbd5e1",
}

AUDIT_SCORE_COLUMNS = [
    "source_validity_score",
    "geo_consistency_score",
    "temporal_alignment_score",
    "semantic_consistency_score",
    "residual_support_score",
    "evidence_validity_score",
]


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs(root: Path) -> dict[str, Path]:
    paper = root / "figure_paper"
    legacy_stash = root / ".figure_paper_legacy_case_card_pair"
    if legacy_stash.exists():
        shutil.rmtree(legacy_stash)
    if paper.exists():
        previous_pair = list((paper / "main").glob("fig7_case_card_pair.*"))
        if previous_pair:
            legacy_stash.mkdir(parents=True, exist_ok=True)
            for path in previous_pair:
                shutil.copy2(path, legacy_stash / path.name)
    if paper.exists():
        shutil.rmtree(paper)
    dirs = {
        "paper": paper,
        "main": paper / "main",
        "appendix": paper / "appendix",
        "excluded": paper / "excluded_legacy",
        "tables": paper / "tables",
        "reports": paper / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    if legacy_stash.exists():
        for path in legacy_stash.glob("fig7_case_card_pair.*"):
            shutil.copy2(path, dirs["excluded"] / path.name)
        shutil.rmtree(legacy_stash)
    return dirs


def _copy_pair(src_stem: Path, dst_stem: Path, manifest: list[dict], **meta: str) -> None:
    copied = []
    for suffix in (".png", ".pdf"):
        src = src_stem.with_suffix(suffix)
        if src.is_file():
            dst = dst_stem.with_suffix(suffix)
            shutil.copy2(src, dst)
            copied.append(str(dst))
    if copied:
        manifest.append({"filename": dst_stem.name, "status": "copied", "outputs": ";".join(copied), **meta})


def _save(fig, dst_stem: Path, manifest: list[dict], **meta: str) -> None:
    png = dst_stem.with_suffix(".png")
    pdf = dst_stem.with_suffix(".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    manifest.append({"filename": dst_stem.name, "status": "generated", "outputs": f"{png};{pdf}", **meta})


def _png_to_pdf(src_png: Path, dst_stem: Path, manifest: list[dict], **meta: str) -> None:
    import matplotlib.image as mpimg

    img = mpimg.imread(src_png)
    h, w = img.shape[:2]
    fig_w = 8.0
    fig_h = max(2.0, fig_w * h / w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    ax.set_axis_off()
    png = dst_stem.with_suffix(".png")
    pdf = dst_stem.with_suffix(".pdf")
    shutil.copy2(src_png, png)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    manifest.append({"filename": dst_stem.name, "status": "legacy_png_with_pdf_wrapper", "outputs": f"{png};{pdf}", **meta})


def _clean_text(text: str) -> str:
    text = text.replace("LP-MOMENT", "PT-MOMENT").replace("LP MOMENT", "PT-MOMENT")
    text = re.sub(r"#{1,6}\s*", "", text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\b[A-Z]\d{3}__[A-Za-z0-9_]+", "event-area station/channel", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, n: int = 310) -> str:
    text = _clean_text(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _clean_card_text(text: str) -> str:
    text = _clean_text(text)
    text = text.replace("…", "").replace("...", "")
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip(" -")


def _section_by_heading(text: str, heading: str, next_prefixes: tuple[str, ...] = ("## ",)) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(heading)}\s*$")
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_positions = []
    for prefix in next_prefixes:
        next_match = re.search(rf"(?m)^{re.escape(prefix)}", text[start:])
        if next_match:
            next_positions.append(start + next_match.start())
    end = min(next_positions) if next_positions else len(text)
    return text[start:end].strip()


def _subsection(section: str, heading: str) -> str:
    return _section_by_heading(section, heading, ("### ", "## "))


def _subsections(section: str, heading: str) -> list[str]:
    pattern = re.compile(rf"(?m)^{re.escape(heading)}\s*$")
    matches = list(pattern.finditer(section))
    out: list[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = len(section)
        next_match = re.search(r"(?m)^### ", section[start:])
        if next_match:
            end = start + next_match.start()
        out.append(section[start:end].strip())
    return out


def _last_subsection_sentence(section: str, heading: str) -> str:
    parts = [_first_complete_sentence(part) for part in _subsections(section, heading)]
    parts = [part for part in parts if part]
    return parts[-1] if parts else ""


def _card_bullets(section: str, limit: int = 3) -> list[str]:
    bullets: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            cleaned = _clean_card_text(stripped[2:])
            if cleaned:
                bullets.append(cleaned)
        if len(bullets) >= limit:
            break
    return bullets


def _kv_from_lines(section: str, key: str) -> str:
    target = key.lower().strip()
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if ":" not in stripped:
            continue
        lhs, rhs = stripped.split(":", 1)
        if lhs.lower().strip() == target:
            return _clean_card_text(rhs)
    return ""


def _first_complete_sentence(text: str) -> str:
    cleaned = _clean_card_text(text)
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def _join_complete(prefix: str, value: str) -> str:
    value = _clean_card_text(value)
    if not value:
        return ""
    sentence = f"{prefix}{value}"
    return sentence if sentence.endswith((".", "!", "?")) else sentence + "."


def _event_card_sentence(bullet: str) -> str:
    text = _clean_card_text(bullet)
    scheduled_name = re.match(r"(.+?) is scheduled for", text)
    name_match = re.search(r"\*\*([^*]+)\*\*|^([^:]+):", bullet)
    if scheduled_name:
        name = _clean_card_text(scheduled_name.group(1))
    elif name_match:
        name = _clean_card_text(name_match.group(1) or name_match.group(2))
    else:
        name = "Structured event"
    time_match = re.search(r"scheduled for ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})", text)
    time = time_match.group(1).strip() if time_match else "inside the forecast horizon"
    tier_match = re.search(r"treated as tier ([A-Z])", text)
    tier = tier_match.group(1) if tier_match else "event"
    station_match = re.search(r"linked to ([0-9]+) station/channel", text)
    station_count = station_match.group(1) if station_match else "multiple"
    includes = ""
    inc_match = re.search(r"including ([^.]+)", text)
    if inc_match:
        names = [part.strip() for part in inc_match.group(1).split(",")[:3]]
        includes = "; key stations include " + ", ".join(names)
    return f"{name} is scheduled for {time}; it is treated as tier {tier} evidence and linked to {station_count} station/channel units{includes}."


def _summary_card_sentences(summary: str) -> list[str]:
    text = _clean_card_text(summary)
    if not text or text.lower().startswith("no model-assisted"):
        return ["No Qwen-Plus model-assisted summary is available for this case."]
    event_part = text.split("Station relevance:", 1)[0].strip()
    station_part = text.split("Station relevance:", 1)[1].split("Expected ridership effect:", 1)[0].strip() if "Station relevance:" in text else ""
    bullets = []
    if event_part:
        bullets.append(_first_complete_sentence(event_part))
    if station_part:
        bullets.append(_first_complete_sentence(station_part))
    return [bullet for bullet in bullets if bullet][:2] or [_first_complete_sentence(text)]


def _audit_card_sentence(bullets: list[str]) -> str:
    values = {}
    for bullet in bullets:
        if ":" in bullet:
            key, value = bullet.split(":", 1)
            values[key.strip()] = value.strip()
    fields = [
        ("source_validity_score", "source"),
        ("geo_consistency_score", "geo"),
        ("temporal_alignment_score", "temporal"),
        ("residual_support_score", "residual"),
        ("evidence_validity_score", "overall"),
    ]
    scores = [f"{label} {values[key]}" for key, label in fields if key in values]
    flags = values.get("conflict_flags", "(none)")
    return f"Audit scores: {', '.join(scores)}; conflict flags: {flags}."


def _memory_card_sentence(bullet: str) -> str:
    text = _clean_card_text(bullet)
    event = re.search(r"\*\*([^*]+)\*\*", bullet) or re.search(r"memory,\s*([^()]+)\s*\(", text)
    direction = re.search(r"direction was \*\*([^*]+)\*\*|direction was ([a-zA-Z]+)", bullet) or re.search(r"direction was ([a-zA-Z]+)", text)
    residual = re.search(r"residual ([+\-]?[0-9.]+%)", text)
    correction = re.search(r"correction was ([+\-]?[0-9.]+%)", text)
    name = _clean_card_text(event.group(1)) if event else "Historical analogue"
    direction_text = _clean_card_text((direction.group(1) or direction.group(2)) if direction else "similar")
    residual_text = residual.group(1) if residual else "observed"
    correction_text = correction.group(1) if correction else "bounded"
    return f"{name} showed {direction_text} residual evidence ({residual_text}) with median correction {correction_text}."


def _included_card_sentence(line: str) -> str:
    text = _clean_card_text(line)
    station_match = re.match(r"([^,.;]+?) is included because", text)
    gate_match = re.search(r"gate score:\s*([0-9.]+)", text)
    relation_match = re.search(r"Relation:\s*([^.;]+)", text)
    station = station_match.group(1).strip() if station_match else "The selected station/channel"
    gate = gate_match.group(1).rstrip(".") if gate_match else "n/a"
    relation = relation_match.group(1).strip() if relation_match else "venue/station evidence"
    return f"{station} is included because {relation}; gate score {gate}."


def _controller_units_sentence(adjusted_units: str, excluded_units: str) -> str:
    adjusted = _clean_card_text(adjusted_units)
    if adjusted.lower() in {"", "(none)", "none"}:
        adjusted_summary = "No station/channel units are adjusted"
    else:
        examples = [part.strip() for part in adjusted.split(",") if part.strip()][:3]
        adjusted_summary = f"Adjusted units include {', '.join(examples)}"
    excluded = _clean_card_text(excluded_units) or "n/a"
    return f"{adjusted_summary}; excluded units: {excluded}."


def _residual_support_sentence(bullet: str) -> str:
    text = _clean_card_text(bullet)
    if text.startswith("{") or '"direction_agreement"' in text:
        direction = re.search(r'"direction_agreement":\s*"([^"]+)"', text)
        correction = re.search(r'"median_correction":\s*([+\-]?[0-9.]+)', text)
        n_eff = re.search(r'"n_eff":\s*([0-9.]+)', text)
        return (
            f"Residual support summary: direction {direction.group(1) if direction else 'n/a'}, "
            f"median correction {correction.group(1) if correction else 'n/a'}%, "
            f"N_eff {n_eff.group(1) if n_eff else 'n/a'}."
        )
    return _first_complete_sentence(text)


def _metric_sentence(label: str, line: str) -> str:
    vals = re.findall(r"(raw|adjusted) WAPE=([0-9.]+)", line or "")
    if len(vals) >= 2:
        by_name = {name: value for name, value in vals}
        delta = re.search(r"delta\(raw-adjusted\) WAPE=([+\-]?[0-9.]+)", line or "")
        suffix = f"; delta {delta.group(1)}" if delta else ""
        return f"{label}: raw WAPE {by_name.get('raw', 'n/a')} to adjusted WAPE {by_name.get('adjusted', 'n/a')}{suffix}."
    cleaned = _clean_card_text(line)
    return f"{label}: {cleaned}." if cleaned else f"{label}: n/a."


def _correction_stats_sentence(line: str) -> str:
    max_abs = re.search(r'"max_abs_correction":\s*([0-9.eE+\-]+)', line or "")
    active = re.search(r'"active_correction_cells":\s*([0-9]+)', line or "")
    bound = re.search(r'"correction_bound":\s*([0-9.eE+\-]+)', line or "")
    if max_abs or active or bound:
        return (
            f"Correction statistics: max absolute correction {max_abs.group(1) if max_abs else 'n/a'}, "
            f"active cells {active.group(1) if active else 'n/a'}, "
            f"bound {bound.group(1) if bound else 'n/a'}."
        )
    return _first_complete_sentence(line)


def _metric_brief(label: str, line: str) -> str:
    vals = re.findall(r"(raw|adjusted) WAPE=([0-9.]+)", line or "")
    if len(vals) >= 2:
        by_name = {name: value for name, value in vals}
        return f"{label}: raw {by_name.get('raw', 'n/a')} -> adjusted {by_name.get('adjusted', 'n/a')}"
    if line:
        return f"{label}: {_truncate(line, 90)}"
    return f"{label}: n/a"


def _line_after(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[-1].strip()
    return ""


def _section(text: str, heading: str, next_level: str = "## ") -> str:
    idx = text.find(heading)
    if idx < 0:
        return ""
    tail = text[idx + len(heading) :]
    marker = tail.find("\n" + next_level)
    return tail[:marker].strip() if marker >= 0 else tail.strip()


def _bullets(section: str, limit: int = 2) -> list[str]:
    out = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(stripped[2:].strip())
        if len(out) >= limit:
            break
    return out


def _find_case(root: Path, case_name: str) -> Path:
    paths = sorted(
        path
        for path in (root / "qwenplus_live_case_set_summary_only" / case_name / "explanations").glob("*.md")
        if ".ipynb_checkpoints" not in path.parts
    )
    if not paths:
        raise FileNotFoundError(f"case markdown missing: {case_name}")
    return paths[-1]


def _case_payload(root: Path, case_name: str) -> dict:
    path = _find_case(root, case_name)
    text = path.read_text(encoding="utf-8", errors="ignore")
    request = _section_by_heading(text, "## 1. Forecast Request")
    audit = _section_by_heading(text, "## 3. Forecast-time Evidence Audit")
    candidates = _section_by_heading(text, "## 4. Candidate Event-Station-Channel Units")
    memory_section = _section_by_heading(text, "## 5. Historical Residual Memory")
    controller_section = _section_by_heading(text, "## 6. Calibration Controller")
    adjusted = _section_by_heading(text, "## 7. Adjusted Forecast")
    explanation = _section_by_heading(text, "## 8. Explanation", ("## Post-hoc", "## "))

    structured = _card_bullets(_subsection(audit, "### Structured Future Events within Horizon"), 2)
    summary = _card_bullets(_subsection(audit, "### Model-Assisted Summary (Non-citable)"), 1)
    geo_temporal = _card_bullets(_subsection(audit, "### Geo-temporal consistency"), 10)
    included = _card_bullets(_subsection(candidates, "### Included"), 2)
    excluded = _card_bullets(_subsection(candidates, "### Excluded"), 1)
    memory = _card_bullets(
        _subsection(memory_section, "### Historical Event Memory from Train/Validation")
        or _subsection(memory_section, "### Matched cases"),
        2,
    )
    residual_support = _card_bullets(_subsection(memory_section, "### Residual support summary"), 1)
    controller = _subsection(controller_section, "### Calibration Decision") or controller_section
    factual = _last_subsection_sentence(explanation, "### Factual evidence")
    residual = _last_subsection_sentence(explanation, "### Residual analogues")
    rationale = _last_subsection_sentence(explanation, "### Calibration rationale")
    uncertainty = _last_subsection_sentence(explanation, "### Uncertainty")
    risk = _last_subsection_sentence(explanation, "### Risk control")
    return {
        "path": str(path),
        "anchor": _kv_from_lines(request, "Anchor time") or _line_after(text, "- Anchor time"),
        "horizon": _kv_from_lines(request, "Forecast horizon"),
        "mode": _kv_from_lines(request, "Mode") or _line_after(text, "- Mode"),
        "station_scope": _kv_from_lines(request, "Station scope"),
        "calibration_enabled": _kv_from_lines(request, "Calibration enabled") or _line_after(text, "- Calibration enabled"),
        "structured": structured,
        "summary": summary[0] if summary else "No Qwen-Plus model-assisted summary was available for this abstention case.",
        "geo_temporal": geo_temporal,
        "included": included,
        "excluded": excluded,
        "memory": memory,
        "residual_support": residual_support,
        "decision": _kv_from_lines(controller, "Calibration decision") or _line_after(controller, "- Calibration decision"),
        "confidence": _kv_from_lines(controller, "Confidence"),
        "bound": _kv_from_lines(controller, "Correction bound") or _line_after(controller, "- Correction bound"),
        "max_correction": _kv_from_lines(controller, "Max correction") or _line_after(controller, "- Max correction"),
        "adjusted_units": _kv_from_lines(controller, "Adjusted station/channel units"),
        "excluded_units": _kv_from_lines(controller, "Excluded station/channel units"),
        "event_wape": _kv_from_lines(adjusted, "Event-window subset") or _line_after(adjusted, "- Event-window subset"),
        "channel_wape": _kv_from_lines(adjusted, "Adjusted-channel subset") or _line_after(adjusted, "- Adjusted-channel subset"),
        "correction_stats": _kv_from_lines(adjusted, "Correction magnitude statistics"),
        "factual": factual,
        "residual_reasoning": residual,
        "rationale": rationale,
        "uncertainty": uncertainty,
        "risk": risk,
    }


def _normalize_physical_event_key(value: str) -> str:
    parts = str(value or "").split("|", 2)
    if len(parts) != 3:
        return ""
    title = re.sub(r"\s+", " ", parts[0].strip().lower())
    event_time = parts[1].strip()[:16]
    location = re.sub(r"\s+", " ", parts[2].strip().lower())
    return f"{title}|{event_time}|{location}"


def _physical_event_key(event: dict) -> str:
    title = re.sub(r"\s+", " ", str(event.get("title") or "").strip().lower())
    event_time = str(event.get("event_time") or event.get("start_time") or "").strip()[:16]
    location = str(event.get("location") or event.get("event_location") or "").split(" | ", 1)[0]
    location = re.sub(r"\s+", " ", location.strip().lower())
    return f"{title}|{event_time}|{location}" if title and event_time and location else ""


def _selected_controller_abstention(root: Path) -> tuple[dict, Path]:
    selection_path = root / "reports" / "selected_visualization_cases.json"
    if not selection_path.is_file():
        selection_path = PROJECT_ROOT / "experiments" / "visualization" / "outputs" / "reports" / "selected_visualization_cases.json"
    payload = _load_json(selection_path)
    selected = next(
        (
            row
            for row in payload.get("cases", [])
            if row.get("case_type") == "controller_abstention" and row.get("status") == "selected"
        ),
        None,
    )
    if not selected:
        raise ValueError(f"strict Controller-abstention case missing from {selection_path}")
    input_path = Path(str(selected.get("input_file") or ""))
    if not input_path.is_absolute():
        project_input = PROJECT_ROOT / input_path
        input_path = project_input if project_input.is_file() else root / input_path
    if not input_path.is_file():
        raise FileNotFoundError(f"selected Controller-abstention prediction missing: {input_path}")
    return selected, input_path


def _aligned_formal_qwen_summary(root: Path, physical_key: str) -> dict:
    report_path = root / "reports" / "qwenplus_live_evidence_summary.json"
    report = _load_json(report_path)
    target = _normalize_physical_event_key(physical_key)
    matches = [
        row
        for row in report.get("events", [])
        if _normalize_physical_event_key(row.get("physical_event_key", "")) == target
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one formal Qwen summary match for physical_event_key={target}; got {len(matches)}")
    summaries = matches[0].get("summaries") or []
    if not summaries:
        raise ValueError(f"formal Qwen summary list is empty for physical_event_key={target}")
    summary = summaries[0]
    summary_key = _normalize_physical_event_key(summary.get("event_key", ""))
    if summary_key != target:
        raise ValueError(f"formal Qwen summary key mismatch: event={target}, summary={summary_key}")
    return {
        "report_path": str(report_path),
        "physical_event_key": target,
        "summary_event_key": summary_key,
        "summary": summary.get("summary") or summary.get("event_summary") or "",
        "event_summary": summary.get("event_summary") or "",
        "station_relevance": summary.get("event_relevance_to_station") or "",
    }


def _controller_abstention_case_data(root: Path) -> dict:
    selected, input_path = _selected_controller_abstention(root)
    payload = _load_json(input_path)
    request = payload.get("request") or {}
    decision = payload.get("decision") or {}
    evidence = payload.get("evidence") or {}
    audit = evidence.get("evidence_audit") or {}
    thresholds = audit.get("thresholds") or {}
    correction = decision.get("correction_stats") or {}
    events = evidence.get("structured_events") or []
    if not events:
        raise ValueError("selected Controller-abstention case has no forecast-time structured event")
    event = events[0]
    physical_key = _physical_event_key(event)
    qwen = _aligned_formal_qwen_summary(root, physical_key)

    mode = request.get("mode")
    raw = np.asarray((payload.get("numerical") or {}).get("raw_forecast") or [], dtype=float)
    final = np.asarray(payload.get("adjusted_forecast") or [], dtype=float)
    final_matches_raw = raw.shape == final.shape and raw.size > 0 and bool(
        np.allclose(raw, final, rtol=0.0, atol=1e-8, equal_nan=True)
    )
    max_applied = (
        float(np.max(np.abs((final - raw) / np.maximum(np.abs(raw), 1e-8))))
        if final_matches_raw
        else float("inf")
    )
    controller_participated = bool(
        mode == "event_adapter_frozen_moment"
        and isinstance(decision.get("controller_allowed"), bool)
        and isinstance(decision.get("audit_scores"), dict)
        and correction
        and thresholds
    )
    failures = [
        field
        for field, threshold in thresholds.items()
        if field in audit and float(audit[field]) < float(threshold)
    ]
    if audit.get("severe_conflict_flags"):
        failures.append("severe_conflict")
    invariants = {
        "mode_is_event_adapter_frozen_moment": mode == "event_adapter_frozen_moment",
        "method_is_eaf_mas_c": selected.get("method_label") == "EAF-MAS-C",
        "calibration_enabled": mode == "event_adapter_frozen_moment",
        "controller_participated": controller_participated,
        "controller_allowed_false": decision.get("controller_allowed") is False,
        "abstain_true": decision.get("abstain") is True,
        "final_matches_raw_forecast": final_matches_raw,
        "max_applied_correction_is_zero": max_applied <= 1e-8,
        "failed_controller_evidence_present": bool(failures),
        "adapter_proposal_is_nonzero": float(correction.get("max_abs_correction") or 0.0) > 1e-8,
        "qwen_summary_physical_key_aligned": qwen["summary_event_key"] == physical_key,
    }
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    if failed_invariants:
        raise ValueError("invalid Figure 9(b) Controller-abstention case: " + ", ".join(failed_invariants))

    reason = re.sub(
        r"^Calibration controller abstained:\s*",
        "",
        str(decision.get("reason") or ""),
    ).strip()
    residual = audit.get("residual_support_summary") or {}
    focused = payload.get("focused_prediction") or {}
    return {
        "case_id": selected.get("case_id"),
        "input_path": str(input_path),
        "anchor": request.get("date"),
        "horizon": request.get("horizon"),
        "mode": mode,
        "event": event,
        "physical_event_key": physical_key,
        "qwen": qwen,
        "qwen_summary_used_by_case_controller": bool(evidence.get("model_assisted_summaries")),
        "audit": audit,
        "thresholds": thresholds,
        "failures": failures,
        "residual": residual,
        "historical_cases": evidence.get("historical_event_cases") or [],
        "proposal": correction,
        "controller_allowed": decision.get("controller_allowed"),
        "abstain": decision.get("abstain"),
        "reason": reason,
        "max_applied_correction": max_applied,
        "final_matches_raw": final_matches_raw,
        "raw_metrics": payload.get("raw_metrics") or {},
        "final_metrics": payload.get("metrics") or {},
        "event_metrics": payload.get("event_window_metrics") or {},
        "focused_rows": focused.get("focused_forecast_rows") or [],
        "focused_metrics": focused.get("focused_forecast_metrics") or {},
        "invariants": invariants,
    }


def _controller_abstention_blocks(data: dict) -> list[tuple[str, list[str]]]:
    event = data["event"]
    audit = data["audit"]
    thresholds = data["thresholds"]
    residual = data["residual"]
    proposal = data["proposal"]
    content = str(event.get("content") or "")
    distance = re.search(r"distance_to_station=([0-9.]+m)", content)
    event_line = (
        f"{event.get('title')} | {event.get('event_time')} | {event.get('event_type')} | "
        f"{event.get('channel_name')} ({distance.group(1) if distance else 'distance n/a'})."
    )
    end_time = re.search(r"end_datetime=([^;]+)", content)
    event_location = str(event.get("location") or "").split(" | ", 1)[0]
    qwen_summary = (
        f"The formal Qwen+ record describes {event.get('title')} at {event_location}, "
        f"from {event.get('event_time')} to {end_time.group(1).strip() if end_time else 'the recorded end time'}, "
        f"with {event.get('station_complex')} as the nearby access station."
    )
    audit_line = (
        f"Source {audit.get('source_validity_score', 0):.3f} < {thresholds.get('source_validity_score', 0):.2f} [FAIL]; "
        f"geo {audit.get('geo_consistency_score', 0):.3f} >= {thresholds.get('geo_consistency_score', 0):.2f}; "
        f"temporal {audit.get('temporal_alignment_score', 0):.3f} >= {thresholds.get('temporal_alignment_score', 0):.2f}."
    )
    residual_line = (
        f"Residual support {audit.get('residual_support_score', 0):.3f} >= {thresholds.get('residual_support_score', 0):.2f}; "
        f"overall evidence {audit.get('evidence_validity_score', 0):.3f}."
    )
    historical = data.get("historical_cases") or []
    history_example = historical[0] if historical else {}
    event_metrics = data.get("event_metrics") or {}
    event_raw = (event_metrics.get("raw") or {}).get("wape")
    event_final = (event_metrics.get("adjusted") or {}).get("wape")
    raw_wape = data["raw_metrics"].get("wape")
    final_wape = data["final_metrics"].get("wape")
    station_row = next(
        (row for row in data.get("focused_rows", []) if row.get("channel_name") == event.get("channel_name")),
        (data.get("focused_rows") or [{}])[-1],
    )
    return [
        (
            "Forecast request",
            [
                f"Anchor {data['anchor']}; horizon {data['horizon']} hours; EAF-MAS-C ({data['mode']}).",
                "calibration_enabled=True; the Calibration Controller received audit scores and a bounded adapter proposal.",
            ],
        ),
        (
            "Forecast-time event + aligned Qwen-Plus summary",
            [
                event_line,
                f"physical_event_key={data['physical_event_key']}",
                f"Formal Qwen+ record matched by that key: {qwen_summary}",
                "Alignment note: the formal summary exists, but was not attached to this case run's Controller input.",
            ],
        ),
        (
            "Evidence audit",
            [
                audit_line,
                residual_line,
                f"Failed dimension: {', '.join(data['failures'])}; conflict flags: {', '.join(audit.get('conflict_flags') or ['none'])}; severe conflicts: {', '.join(audit.get('severe_conflict_flags') or ['none'])}.",
            ],
        ),
        (
            "Historical residual support",
            [
                f"Direction {residual.get('direction_agreement')}; median LP-MOMENT correction {100 * float(residual.get('median_correction') or 0):+.1f}%; N_eff={residual.get('n_eff')}; confidence cap={residual.get('confidence_cap')}.",
                f"Example: {history_example.get('historical_event_title', 'n/a')} ({history_example.get('event_time', 'n/a')}), residual {history_example.get('historical_baseline_residual', 'n/a')}, matched correction {history_example.get('median_lp_moment_correction', 'n/a')}.",
            ],
        ),
        (
            "Adapter proposal + Controller abstention",
            [
                f"Nonzero proposal: max {100 * float(proposal.get('max_abs_correction') or 0):.3f}%; {int(proposal.get('active_correction_cells') or 0)} active cells; bound {100 * float(proposal.get('correction_bound') or 0):.1f}%.",
                f"Reason={data['reason']}; controller_allowed=False; abstain=True.",
            ],
        ),
        (
            "Raw forecast retained",
            [
                f"max applied correction = {100 * data['max_applied_correction']:.3f}%; final forecast equals PT-MOMENT raw forecast={data['final_matches_raw']}.",
                "The proposal statistics describe the rejected adapter output, not an applied correction.",
            ],
        ),
        (
            "Retrospective outcome (post-hoc only)",
            [
                f"Overall WAPE: raw {raw_wape:.3f} -> retained {final_wape:.3f}; event-window WAPE: raw {event_raw:.3f} -> retained {event_final:.3f}.",
                f"At {station_row.get('display_station_name', event.get('station_complex'))}, {station_row.get('timestamp', event.get('event_time'))}: raw/retained {float(station_row.get('raw_forecast') or 0):.1f}, actual {float(station_row.get('actual') or 0):.1f}.",
            ],
        ),
        (
            "Risk-control interpretation",
            [
                "Station/time relevance and residual support passed, but the source-evidence hard gate failed; the Controller therefore blocked an otherwise bounded nonzero proposal.",
                "Retrospective observations validate traceability only and were unavailable at decision time.",
            ],
        ),
    ]


def _case_card_blocks(data: dict) -> list[tuple[str, list[str]]]:
    request = [
        f"Anchor {data['anchor']}; horizon {data.get('horizon') or '192 hours'}; mode {data['mode']}; station scope {data.get('station_scope') or 'event_venue28'}; calibration {data['calibration_enabled']}."
    ]
    structured = []
    if data.get("structured"):
        structured.append(_event_card_sentence(data["structured"][0]))
    structured.extend(_summary_card_sentences(data["summary"]))
    if len(data.get("structured", [])) > 1:
        structured.append(_event_card_sentence(data["structured"][1]))
    audit = []
    if data.get("geo_temporal"):
        audit.append(_audit_card_sentence(data["geo_temporal"]))
    if data.get("included"):
        audit.append(_included_card_sentence(data["included"][0]))
    if data.get("excluded"):
        audit.append(_join_complete("Excluded example: ", data["excluded"][0]))
    memory = []
    for item in data.get("memory", [])[:2]:
        memory.append(_memory_card_sentence(item))
    if data.get("residual_support"):
        memory.append(_residual_support_sentence(data["residual_support"][0]))
    controller = [
        f"Decision: {data.get('decision') or 'n/a'}; confidence {data.get('confidence') or 'n/a'}; bound {data.get('bound') or 'n/a'}; max correction {data.get('max_correction') or 'n/a'}.",
        _controller_units_sentence(data.get("adjusted_units", ""), data.get("excluded_units", "")),
    ]
    reasoning = [
        data.get("factual") or "The event relevance is determined from structured event timing, venue, and station-channel matching.",
        data.get("residual_reasoning") or "Historical residual memory is used to support or reject the correction direction.",
        data.get("rationale") or "The controller applies correction only when audit scores and event-channel masks pass.",
        data.get("risk") or "Risk is controlled by bounded correction and by excluding weakly matched channels.",
    ]
    metrics = [
        _metric_sentence("Event-window subset", data.get("event_wape", "")),
        _metric_sentence("Adjusted-channel subset", data.get("channel_wape", "")),
    ]
    if data.get("correction_stats"):
        metrics.append(_correction_stats_sentence(data["correction_stats"]))
    risk = [
        data.get("uncertainty") or "Uncertainty is reported separately from post-hoc metrics.",
        data.get("risk") or "Correction remains bounded and station/channel masked.",
    ]
    blocks = [
        ("Forecast request", request),
        ("Structured event + Qwen-Plus summary", structured[:2]),
        ("Evidence audit", audit[:2]),
        ("Historical residual memory", memory[:2]),
        ("Calibration controller", controller),
        ("Multi-hop reasoning", reasoning[:4]),
        ("Post-hoc local metrics", metrics[:2]),
        ("Risk control", risk[:1]),
    ]
    return [(title, [bullet for bullet in bullets if bullet]) for title, bullets in blocks]


def _draw_text_box(ax, xy, wh, title: str, body: str, color: str) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#ffffff",
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.030, title, color=color, weight="bold", fontsize=8.4, va="top")
    wrapped = "\n".join(textwrap.wrap(_truncate(body, 195), width=55))
    txt = ax.text(x + 0.02, y + h - 0.065, wrapped, color=COLORS["dark"], fontsize=6.55, va="top", linespacing=1.12)
    txt.set_clip_path(box)


def _draw_bullet_box(ax, xy, wh, title: str, bullets: list[str], color: str, width: int = 92) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="#ffffff",
        edgecolor=color,
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(x + 0.018, y + h - 0.020, title, color=color, weight="bold", fontsize=8.4, va="top")
    wrapped_lines: list[str] = []
    for bullet in bullets:
        cleaned = _clean_card_text(bullet)
        if not cleaned:
            continue
        lines = textwrap.wrap(cleaned, width=width)
        if not lines:
            continue
        wrapped_lines.append("• " + lines[0])
        wrapped_lines.extend("  " + line for line in lines[1:])
    body = "\n".join(wrapped_lines)
    txt = ax.text(x + 0.018, y + h - 0.050, body, color=COLORS["dark"], fontsize=6.8, va="top", linespacing=1.10)
    txt.set_clip_path(box)


def generate_summary_coverage(root: Path, dst: Path, manifest: list[dict]) -> None:
    qwen = _load_json(root / "reports" / "qwenplus_live_evidence_summary.json")
    stats = qwen.get("stats", {})
    values = [
        ("High-value\nevents", int(qwen.get("unique_high_value_physical_events", 0))),
        ("Live Qwen-Plus\ncalls", int(stats.get("api_calls", 0))),
        ("Parsed model\nsummaries", int(stats.get("summary_count", 0))),
        ("Used in local\nexplanations", int(stats.get("summary_used_for_explanation", 0))),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    ax.set_axis_off()
    xs = [0.08, 0.33, 0.58, 0.83]
    for i, ((label, value), x) in enumerate(zip(values, xs)):
        color = [COLORS["blue"], COLORS["purple"], COLORS["green"], COLORS["orange"]][i]
        circ = plt.Circle((x, 0.55), 0.095, color=color, alpha=0.95)
        ax.add_patch(circ)
        ax.text(x, 0.55, f"{value}", color="white", ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x, 0.30, label, ha="center", va="center", fontsize=8.5)
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.12, 0.55), xytext=(x + 0.12, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": COLORS["gray"]})
    ax.text(
        0.5,
        0.08,
        "Qwen-Plus is used as a summary agent. URL citation acceptance is intentionally not a main-paper metric.",
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["gray"],
    )
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table="reports/qwenplus_live_evidence_summary.json",
        sample_size=str(qwen.get("unique_high_value_physical_events", 0)),
        claim="model-assisted summary coverage",
        paper_placement="Appendix evidence diagnostics",
    )


def generate_residual_memory_figure(root: Path, dst: Path, manifest: list[dict]) -> None:
    sk = _load_json(root / "autoskill_skillbench_full/reports/autoskill_skillbench_summary.json")
    cov = sk["residual_memory_coverage"]
    org = sk["residual_memory_organization"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), gridspec_kw={"width_ratios": [1.08, 1.0]})
    labels = ["Top-3 retrieval coverage\nmajor-event memory", "Top-3 retrieval coverage\ncoverage memory"]
    vals = [cov["legacy_hit_at_3"], cov["coverage_v2_hit_at_3"]]
    axes[0].bar(labels, vals, color=[COLORS["gray"], COLORS["green"]], width=0.46)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("retrieval hit rate")
    axes[0].set_title("Residual-memory coverage")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.025, f"{v:.3f}", ha="center", weight="bold")
    axes[0].text(0.5, -0.28, "Coverage memory adds weak/no-event references while staying train/validation-only.", transform=axes[0].transAxes, ha="center", fontsize=7.3, color=COLORS["gray"])

    labels2 = ["No-skill\nanalogue relevance", "Skill-guided\nanalogue relevance"]
    vals2 = [org["baseline_relevance_mean"], org["skill_selected_relevance_mean"]]
    axes[1].bar(labels2, vals2, color=[COLORS["gray"], COLORS["purple"]], width=0.46)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Residual analogue organization")
    for i, v in enumerate(vals2):
        axes[1].text(i, v + 0.025, f"{v:.3f}", ha="center", weight="bold")
    axes[1].annotate(
        f"+{org['delta']:.3f}",
        xy=(1, vals2[1]),
        xytext=(0.52, vals2[1] + 0.12),
        arrowprops={"arrowstyle": "->", "color": COLORS["purple"]},
        color=COLORS["purple"],
        weight="bold",
    )
    axes[1].set_ylabel("mean relevance score")
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table="autoskill_skillbench_full/reports/autoskill_skillbench_summary.json",
        sample_size=f"val={sk['skill_lifecycle']['val_experience_count']}; test={sk['skill_lifecycle']['test_experience_count']}",
        claim="AutoSkill-style memory improves residual analogue selection",
        paper_placement="Section 4.3 Skill evolution and residual memory organization",
    )


def generate_safe_abstention_table(root: Path, dst: Path, manifest: list[dict]) -> None:
    sk = _load_json(root / "autoskill_skillbench_full/reports/autoskill_skillbench_summary.json")
    aq = sk["abstention_quality"]
    rows = [
        ("Stress cases", f"{aq['case_count']}", "weak evidence / no-event / conflicting residual"),
        ("Abstention correctness", f"{aq['abstention_correctness']:.3f}", "controller and explanation agree"),
        ("Unsupported correction claims", f"{aq['unsupported_correction_claim_rate']:.3f}", "lower is safer"),
        ("Forecast arrays changed by Skill", "No", "Skill affects reasoning, not forecasts"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    ax.set_axis_off()
    ax.set_title("Safe abstention diagnostics", loc="left", weight="bold")
    y = 0.78
    for label, value, note in rows:
        ax.text(0.02, y, label, weight="bold", fontsize=8.8, va="center")
        ax.text(0.47, y, value, color=COLORS["green"] if value in {"No"} or value.startswith("1.") else COLORS["dark"], weight="bold", fontsize=9.2, va="center")
        ax.text(0.66, y, note, color=COLORS["gray"], fontsize=8, va="center")
        ax.axhline(y - 0.08, color=COLORS["border"], lw=0.6)
        y -= 0.18
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table="autoskill_skillbench_full/reports/autoskill_skillbench_summary.json",
        sample_size=str(aq["case_count"]),
        claim="safe abstention consistency",
        paper_placement="Appendix or Section 4.4 safety diagnostics",
    )


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _dt_key(value: Any) -> str:
    parsed = _parse_dt(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else str(value or "")


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _rank_group(rank: Any) -> str:
    try:
        value = float(rank)
    except Exception:
        return "top128"
    if value <= 32:
        return "top32"
    if value <= 64:
        return "top64"
    return "top128"


def _day_type(value: Any) -> str:
    parsed = _parse_dt(value)
    if parsed is None:
        return "unknown"
    return "weekend" if parsed.weekday() >= 5 else "weekday"


def _venue_from_key(physical_key: str) -> str:
    parts = str(physical_key or "").split("|")
    return parts[2].strip() if len(parts) >= 3 else "unknown venue"


def _parse_distance_m(row: dict[str, Any]) -> float | None:
    for key in ("distance_m", "distance_to_station"):
        try:
            value = row.get(key)
            if value not in (None, ""):
                return float(value)
        except Exception:
            pass
    content = str(row.get("content") or row.get("location") or "")
    match = re.search(r"distance_to_station=([0-9.]+)m", content)
    if match:
        return float(match.group(1))
    return None


def _load_event_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("events", [])
    else:
        rows = payload
    return [dict(row) for row in rows if isinstance(row, dict)]


def _event_lookup(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (_norm_text(row.get("title")), _dt_key(row.get("event_time")))
        lookup.setdefault(key, []).append(row)
    return lookup


def _best_memory_match(memory_rows: list[dict[str, Any]], event_type: str, tier: str, rank_group: str, day_type: str) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    for row in memory_rows:
        score = 0.0
        if _norm_text(row.get("event_type")) == _norm_text(event_type):
            score += 0.40
        if str(row.get("impact_tier") or "").upper() == str(tier or "").upper():
            score += 0.20
        if str(row.get("rank_group") or "") == str(rank_group or ""):
            score += 0.15
        if str(row.get("day_type") or "") == str(day_type or ""):
            score += 0.15
        try:
            score += min(0.10, float(row.get("n_eff") or 0.0) / 500.0)
        except Exception:
            pass
        if score > best_score:
            best_score = score
            best = row
    return best, round(float(min(best_score, 1.0)), 4)


def build_full_test_evidence_audit_rows(
    root: Path,
    events_json_path: Path | None = None,
    residual_memory_path: Path | None = None,
    horizon_hours: int = 192,
) -> list[dict[str, Any]]:
    """Build one forecast-time audit row per Qwen-Plus high-value event.

    The audit score is deterministic and traceable to summary-only Qwen-Plus
    outputs, structured event records, and train/validation residual memory.
    It intentionally does not require URL/citation evidence.
    """
    qwen_path = root / "reports" / "qwenplus_live_evidence_summary.json"
    qwen = _load_json(qwen_path)
    qwen_events = list(qwen.get("events") or [])
    if not qwen_events:
        return []
    events_json_path = events_json_path or PROJECT_ROOT / "data" / "nyc_top128_station_events.json"
    residual_memory_path = residual_memory_path or root / "autoskill_skillbench_full" / "models" / "residual_memory" / "memory_train_val_v2.json"
    structured_lookup = _event_lookup(_load_event_records(events_json_path))
    memory_rows = _load_json(residual_memory_path) if residual_memory_path.is_file() else []
    if not isinstance(memory_rows, list):
        memory_rows = []

    rows: list[dict[str, Any]] = []
    for event in qwen_events:
        title = str(event.get("title") or "")
        event_time_text = _dt_key(event.get("event_time"))
        anchor_time_text = _dt_key(event.get("first_anchor_date"))
        event_time = _parse_dt(event_time_text)
        anchor_time = _parse_dt(anchor_time_text)
        matched = structured_lookup.get((_norm_text(title), event_time_text), [])
        summary_count = int(event.get("model_assisted_summary_count") or 0)
        summary_used = bool(event.get("summary_used_for_explanation"))
        tier = str(event.get("impact_tier") or (matched[0].get("impact_tier") if matched else "B") or "B").upper()
        event_type = str(matched[0].get("event_type") if matched else "high-value event")
        distinct_channels = sorted({str(row.get("channel_name")) for row in matched if row.get("channel_name")})
        min_rank = min([float(row.get("station_rank")) for row in matched if str(row.get("station_rank", "")).replace(".", "", 1).isdigit()] or [128.0])
        distances = [d for d in (_parse_distance_m(row) for row in matched) if d is not None]
        min_distance = min(distances) if distances else None
        rank_group = _rank_group(min_rank)
        day_type = _day_type(event_time_text)
        memory, residual_score = _best_memory_match(memory_rows, event_type, tier, rank_group, day_type)

        inside_horizon = bool(anchor_time and event_time and anchor_time <= event_time < anchor_time + timedelta(hours=horizon_hours))
        source_score = 0.35
        if summary_count > 0:
            source_score = 0.65 if summary_used else 0.58
        if matched and any(str(row.get("source") or "").startswith("nyc_") for row in matched):
            source_score += 0.10
        source_score = round(float(min(source_score, 0.90)), 4)

        geo_score = 0.45
        if matched:
            geo_score = 0.62
            if min_distance is not None:
                if min_distance <= 800:
                    geo_score += 0.20
                elif min_distance <= 1200:
                    geo_score += 0.10
            elif min_rank <= 64:
                geo_score += 0.12
            if len(distinct_channels) >= 2:
                geo_score += 0.05
        geo_score = round(float(min(geo_score, 0.95)), 4)

        temporal_score = 1.0 if inside_horizon else 0.20
        tier_score = {"A": 0.92, "B": 0.84, "C": 0.66, "D": 0.52}.get(tier, 0.60)
        semantic_score = round(float(tier_score if matched else min(tier_score, 0.74)), 4)
        evidence_score = round(float(np.mean([source_score, geo_score, temporal_score, semantic_score, residual_score])), 4)

        flags = []
        if not inside_horizon:
            flags.append("outside_forecast_horizon")
        if summary_count <= 0:
            flags.append("model_assisted_summary_missing")
        if not matched:
            flags.append("structured_event_match_missing")
        if residual_score < 0.50:
            flags.append("weak_residual_memory_support")
        if geo_score < 0.60:
            flags.append("weak_geo_station_match")

        rows.append(
            {
                "physical_event_key": event.get("physical_event_key", ""),
                "event_title": title,
                "event_time": event_time_text,
                "first_anchor_date": anchor_time_text,
                "impact_tier": tier,
                "event_type": event_type,
                "venue_group": _venue_from_key(str(event.get("physical_event_key") or "")),
                "source_type": "structured_event_kb+qwenplus_model_summary" if matched else "qwenplus_model_summary",
                "source_quality": "model-assisted-summary",
                "summary_available": summary_count > 0,
                "summary_used_for_explanation": summary_used,
                "structured_match_count": len(matched),
                "matched_channel_count": len(distinct_channels),
                "matched_channels_preview": "; ".join(distinct_channels[:4]),
                "min_station_rank": int(min_rank) if min_rank == int(min_rank) else min_rank,
                "min_distance_m": "" if min_distance is None else round(float(min_distance), 2),
                "rank_group": rank_group,
                "day_type": day_type,
                "inside_horizon": inside_horizon,
                "residual_memory_event_type": "" if memory is None else memory.get("event_type", ""),
                "residual_memory_n_eff": "" if memory is None else memory.get("n_eff", ""),
                "residual_memory_median_correction": "" if memory is None else memory.get("median_correction", ""),
                "source_validity_score": source_score,
                "geo_consistency_score": geo_score,
                "temporal_alignment_score": temporal_score,
                "semantic_consistency_score": semantic_score,
                "residual_support_score": residual_score,
                "evidence_validity_score": evidence_score,
                "conflict_flags": ";".join(flags),
            }
        )
    return rows


def _write_audit_rows(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "physical_event_key",
        "event_title",
        "event_time",
        "first_anchor_date",
        "impact_tier",
        "event_type",
        "venue_group",
        "source_type",
        "source_quality",
        "summary_available",
        "summary_used_for_explanation",
        "structured_match_count",
        "matched_channel_count",
        "matched_channels_preview",
        "min_station_rank",
        "min_distance_m",
        "rank_group",
        "day_type",
        "inside_horizon",
        "residual_memory_event_type",
        "residual_memory_n_eff",
        "residual_memory_median_correction",
        *AUDIT_SCORE_COLUMNS,
        "conflict_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _score_variance(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {col: float(pd.Series([row[col] for row in rows], dtype="float64").var(ddof=0)) for col in AUDIT_SCORE_COLUMNS}


def _audit_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in AUDIT_SCORE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def generate_full_test_audit_heatmap(rows: list[dict[str, Any]], dst: Path, manifest: list[dict], root: Path) -> None:
    df = _audit_dataframe(rows)
    ordered = df.sort_values(["impact_tier", "evidence_validity_score", "event_time"], ascending=[True, False, True]).reset_index(drop=True)
    matrix = ordered[AUDIT_SCORE_COLUMNS].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 9.2))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(AUDIT_SCORE_COLUMNS)))
    ax.set_xticklabels([col.replace("_score", "").replace("_", "\n") for col in AUDIT_SCORE_COLUMNS], fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel(f"{len(rows)} high-value physical events, sorted by tier and audit profile")
    ax.set_title("Full-test forecast-time evidence audit heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("audit score", fontsize=8)
    tier_counts = ordered["impact_tier"].value_counts().sort_index()
    ax.text(
        0.02,
        0.98,
        "\n".join([f"Tier {tier}: n={count}" for tier, count in tier_counts.items()]),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color=COLORS["dark"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=COLORS["border"], alpha=0.90),
    )
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table="figure_paper/tables/full_test_evidence_audit_rows.csv",
        sample_size=str(len(rows)),
        claim="full-test high-value event evidence audit profile",
        paper_placement="Appendix or Section 4.4 evidence audit diagnostics",
    )


def generate_full_test_audit_distribution(rows: list[dict[str, Any]], dst: Path, manifest: list[dict], root: Path) -> None:
    df = _audit_dataframe(rows)
    values = [df[col].dropna().to_numpy(dtype=float) for col in AUDIT_SCORE_COLUMNS]
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    parts = ax.violinplot(values, showmeans=True, showmedians=True, widths=0.72)
    for body in parts["bodies"]:
        body.set_facecolor(COLORS["blue"])
        body.set_alpha(0.25)
        body.set_edgecolor(COLORS["blue"])
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(COLORS["dark"])
            parts[key].set_linewidth(1.0)
    for i, vals in enumerate(values, 1):
        if len(vals):
            sample = vals[:: max(1, len(vals) // 80)]
            jitter = np.linspace(-0.08, 0.08, len(sample))
            ax.scatter(np.full(len(sample), i) + jitter, sample, s=7, alpha=0.25, color=COLORS["purple"], linewidths=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("audit score")
    ax.set_xticks(range(1, len(AUDIT_SCORE_COLUMNS) + 1))
    ax.set_xticklabels([col.replace("_score", "").replace("_", "\n") for col in AUDIT_SCORE_COLUMNS], fontsize=8)
    variances = _score_variance(rows)
    low = [col.replace("_score", "").replace("_", " ") for col, var in variances.items() if var < 1e-4]
    if low:
        ax.text(
            0.99,
            0.06,
            "Low-variance dimensions:\n" + "\n".join(low[:4]),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor=COLORS["border"]),
        )
    ax.set_title("Audit score distributions across full-test high-value events")
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table="figure_paper/tables/full_test_evidence_audit_rows.csv",
        sample_size=str(len(rows)),
        claim="score distribution and low-variance diagnostic",
        paper_placement="Appendix evidence diagnostics",
    )


def generate_full_test_audit_tier_summary(rows: list[dict[str, Any]], dst: Path, manifest: list[dict], root: Path) -> None:
    df = _audit_dataframe(rows)
    tier = df.groupby("impact_tier", dropna=False)[AUDIT_SCORE_COLUMNS].mean().sort_index()
    event_type = df.groupby("event_type", dropna=False).agg(n=("event_title", "count"), evidence=("evidence_validity_score", "mean")).sort_values("n", ascending=False).head(8)
    venue = df.groupby("venue_group", dropna=False).agg(n=("event_title", "count"), evidence=("evidence_validity_score", "mean")).sort_values("n", ascending=False).head(8)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), gridspec_kw={"width_ratios": [1.05, 1.55, 1.55]})
    im = axes[0].imshow(tier.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[0].set_xticks(range(len(AUDIT_SCORE_COLUMNS)))
    axes[0].set_xticklabels(["source", "geo", "temporal", "semantic", "residual", "overall"], fontsize=7, rotation=35, ha="right")
    axes[0].set_yticks(range(len(tier.index)))
    axes[0].set_yticklabels([f"Tier {x}" for x in tier.index], fontsize=8)
    axes[0].set_title("Mean score by tier")
    for i in range(tier.shape[0]):
        for j in range(tier.shape[1]):
            axes[0].text(j, i, f"{tier.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.2)
    axes[0].text(0.02, 0.96, "cell text = mean score", transform=axes[0].transAxes, ha="left", va="top", fontsize=6.8, color=COLORS["gray"], bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor=COLORS["border"], alpha=0.85))

    axes[1].barh(range(len(event_type)), event_type["evidence"], color=COLORS["green"], alpha=0.85)
    axes[1].set_yticks(range(len(event_type)))
    axes[1].set_yticklabels([_truncate(str(idx), 24) for idx in event_type.index], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.0)
    axes[1].set_xlabel("mean evidence validity")
    axes[1].set_title("Top event types")
    for i, (_, row) in enumerate(event_type.iterrows()):
        axes[1].text(min(float(row["evidence"]) + 0.012, 0.97), i, f"n={int(row['n'])}", va="center", fontsize=7, color=COLORS["gray"])

    axes[2].barh(range(len(venue)), venue["evidence"], color=COLORS["orange"], alpha=0.85)
    axes[2].set_yticks(range(len(venue)))
    axes[2].set_yticklabels([_truncate(str(idx), 24) for idx in venue.index], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1.0)
    axes[2].set_xlabel("mean evidence validity")
    axes[2].set_title("Top venue groups")
    for i, (_, row) in enumerate(venue.iterrows()):
        axes[2].text(min(float(row["evidence"]) + 0.012, 0.97), i, f"n={int(row['n'])}", va="center", fontsize=7, color=COLORS["gray"])
    fig.suptitle("Full-test audit summary by tier, event type, and venue group", y=1.02, fontsize=11, weight="bold")
    fig.subplots_adjust(wspace=0.72)
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table="figure_paper/tables/full_test_evidence_audit_rows.csv",
        sample_size=str(len(rows)),
        claim="audit score grouping by event tier, type, and venue",
        paper_placement="Appendix evidence diagnostics",
    )


def generate_full_test_evidence_audit_package(root: Path, dirs: dict[str, Path], manifest: list[dict]) -> None:
    rows = build_full_test_evidence_audit_rows(root)
    expected = int(_load_json(root / "reports" / "qwenplus_live_evidence_summary.json").get("unique_high_value_physical_events", 0))
    if expected and len(rows) != expected:
        raise ValueError(f"full-test audit row count mismatch: expected {expected}, got {len(rows)}")
    table_path = dirs["tables"] / "full_test_evidence_audit_rows.csv"
    _write_audit_rows(rows, table_path)
    variance = _score_variance(rows) if rows else {}
    (dirs["tables"] / "full_test_evidence_audit_variance.csv").write_text(
        "field,variance,n\n" + "".join(f"{field},{value},{len(rows)}\n" for field, value in variance.items()),
        encoding="utf-8",
    )
    if rows:
        df = _audit_dataframe(rows)
        tier_summary = df.groupby("impact_tier", dropna=False)[AUDIT_SCORE_COLUMNS].mean()
        tier_summary["n"] = df.groupby("impact_tier", dropna=False)["event_title"].count()
        tier_summary.to_csv(dirs["tables"] / "full_test_evidence_audit_tier_summary.csv")
        generate_full_test_audit_heatmap(rows, dirs["appendix"] / "full_test_evidence_audit_heatmap", manifest, root)
        generate_full_test_audit_distribution(rows, dirs["appendix"] / "full_test_audit_score_distribution", manifest, root)
        generate_full_test_audit_tier_summary(rows, dirs["appendix"] / "full_test_audit_tier_summary", manifest, root)
    summary = {
        "row_count": len(rows),
        "expected_high_value_events": expected,
        "score_variance": variance,
        "low_variance_fields": [field for field, value in variance.items() if value < 1e-4],
        "source": "Qwen-Plus summary-only high-value physical events joined with structured event KB and train/validation residual memory v2",
        "claim_boundary": "Evidence audit supports explanation/controller transparency; it is not URL citation quality and not forecast accuracy.",
    }
    (dirs["reports"] / "full_test_evidence_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_single_case_card(root: Path, case_name: str, title: str, dst: Path, manifest: list[dict], color: str) -> None:
    data = _case_payload(root, case_name)
    blocks = _case_card_blocks(data)
    block_map = {block_title: bullets for block_title, bullets in blocks}
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.025, 0.975, title, fontsize=13, weight="bold", color=color, va="top")
    ax.text(
        0.025,
        0.935,
        f"Anchor {data['anchor']} | mode {data['mode']} | calibration {data['calibration_enabled']}",
        fontsize=7.8,
        color=COLORS["gray"],
        va="top",
    )
    left_x, right_x = 0.025, 0.515
    col_w = 0.46
    left_layout = [
        ("Forecast request", 0.120),
        ("Structured event + Qwen-Plus summary", 0.255),
        ("Evidence audit", 0.195),
        ("Historical residual memory", 0.220),
    ]
    right_layout = [
        ("Calibration controller", 0.160),
        ("Multi-hop reasoning", 0.365),
        ("Post-hoc local metrics", 0.145),
        ("Risk control", 0.140),
    ]
    for x, layout in ((left_x, left_layout), (right_x, right_layout)):
        y_top = 0.885
        for block_title, height in layout:
            y_top -= height
            _draw_bullet_box(ax, (x, y_top), (col_w, height), block_title, block_map.get(block_title, []), color, width=60)
            y_top -= 0.016
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table=data["path"],
        sample_size="1 case",
        claim=f"structured case card for {title}",
        paper_placement="Section 4.2 case study",
    )


def _generate_controller_abstention_case_card(root: Path, dst: Path, manifest: list[dict], color: str) -> None:
    data = _controller_abstention_case_data(root)
    blocks = _controller_abstention_blocks(data)
    block_map = {block_title: bullets for block_title, bullets in blocks}
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.025, 0.975, "Weak-Evidence / Safe Abstention", fontsize=13, weight="bold", color=color, va="top")
    ax.text(
        0.025,
        0.935,
        f"Case {data['case_id']} | anchor {data['anchor']} | EAF-MAS-C | controller_allowed=False",
        fontsize=7.8,
        color=COLORS["gray"],
        va="top",
    )
    left_x, right_x = 0.025, 0.515
    col_w = 0.46
    left_layout = [
        ("Forecast request", 0.145),
        ("Forecast-time event + aligned Qwen-Plus summary", 0.275),
        ("Evidence audit", 0.215),
        ("Historical residual support", 0.175),
    ]
    right_layout = [
        ("Adapter proposal + Controller abstention", 0.175),
        ("Raw forecast retained", 0.155),
        ("Retrospective outcome (post-hoc only)", 0.205),
        ("Risk-control interpretation", 0.200),
    ]
    for x, layout in ((left_x, left_layout), (right_x, right_layout)):
        y_top = 0.885
        for block_title, height in layout:
            y_top -= height
            _draw_bullet_box(
                ax,
                (x, y_top),
                (col_w, height),
                block_title,
                block_map.get(block_title, []),
                color,
                width=60,
            )
            y_top -= 0.016
    _save(
        fig,
        dst,
        manifest,
        source_root=str(root),
        source_table=f"{data['input_path']};{data['qwen']['report_path']}",
        sample_size="1 authentic EAF-MAS-C Controller-abstention case",
        claim="source-evidence hard-gate abstention retains the PT-MOMENT raw forecast",
        paper_placement="Section 5.10 qualitative case study, Figure 9(b)",
    )
    record = {
        "figure_panel": "Figure 9(b)",
        "title": "Weak-Evidence / Safe Abstention",
        "case_id": data["case_id"],
        "anchor": data["anchor"],
        "event": {
            "title": data["event"].get("title"),
            "event_time": data["event"].get("event_time"),
            "event_type": data["event"].get("event_type"),
            "station_channel": data["event"].get("channel_name"),
            "physical_event_key": data["physical_event_key"],
        },
        "selection_source": data["input_path"],
        "controller_condition": {
            "failed_dimensions": data["failures"],
            "scores": {field: data["audit"].get(field) for field in AUDIT_SCORE_COLUMNS},
            "thresholds": data["thresholds"],
            "reason": data["reason"],
            "controller_allowed": data["controller_allowed"],
            "abstain": data["abstain"],
        },
        "adapter_proposal": data["proposal"],
        "max_applied_correction": data["max_applied_correction"],
        "raw_forecast_retained": data["final_matches_raw"],
        "qwen_summary_alignment": {
            "status": "matched_by_physical_event_key",
            "physical_event_key": data["physical_event_key"],
            "summary_event_key": data["qwen"]["summary_event_key"],
            "formal_summary_present": True,
            "summary_used_by_case_controller": data["qwen_summary_used_by_case_controller"],
            "formal_report": data["qwen"]["report_path"],
        },
        "invariants": data["invariants"],
        "post_hoc_visualization_only": True,
    }
    record_path = dst.parent.parent / "reports" / "figure9b_case_selection.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_case_cards(root: Path, main_dir: Path, manifest: list[dict]) -> None:
    _generate_single_case_card(
        root,
        "event_intervention",
        "Bounded Event Intervention",
        main_dir / "fig7a_bounded_event_intervention_case_card",
        manifest,
        COLORS["green"],
    )
    _generate_controller_abstention_case_card(
        root,
        main_dir / "fig7b_weak_evidence_abstention_case_card",
        manifest,
        COLORS["blue"],
    )


def generate_case_card(root: Path, dst: Path, manifest: list[dict]) -> None:
    """Backward-compatible wrapper; new paper package uses separate case cards."""
    generate_case_cards(root, dst.parent, manifest)


def _exclude_legacy(root: Path, dirs: dict[str, Path], manifest: list[dict], active_sources: set[str]) -> None:
    excluded = dirs["excluded"]
    for source in [root / "figures_legacy", root / "figures_main_final", root / "figures_appendix_final"]:
        if not source.is_dir():
            continue
        for path in sorted(source.glob("*.png")):
            if path.name in active_sources:
                continue
            dst = excluded / path.name
            shutil.copy2(path, dst)
    manifest.append(
        {
            "filename": "excluded_legacy",
            "status": "archived",
            "outputs": str(excluded),
            "source_root": str(root),
            "source_table": "reports/figure_provenance_audit.csv",
            "sample_size": "n/a",
            "claim": "old, conflicting, duplicate, or diagnostic-only figures excluded from paper package",
            "paper_placement": "not for paper",
        }
    )


def assemble(run_root: str | Path) -> dict:
    _paper_style()
    root = Path(run_root)
    dirs = _ensure_dirs(root)
    manifest: list[dict] = []

    main = dirs["main"]
    appendix = dirs["appendix"]
    source_main = root / "figures_main_final"

    _copy_pair(source_main / "figA_full_split_prediction_landscape", main / "fig1_full_split_landscape", manifest, source_root=str(root), source_table="predictions/full_metric_rows.csv", sample_size="576 test anchors", claim="test split and event-station coverage", paper_placement="Section 4.1")
    _copy_pair(source_main / "figC_event_active_adapter_gain", main / "fig2_event_active_gain", manifest, source_root=str(root), source_table="predictions/full_metric_rows.csv", sample_size="576 test anchors", claim="event-active WAPE gain distribution", paper_placement="Section 4.1/4.2")
    _copy_pair(source_main / "figD_correction_safety_sparsity", main / "fig3_correction_safety", manifest, source_root=str(root), source_table="predictions/full_metric_rows.csv", sample_size="576 test anchors", claim="sparse bounded correction and non-event safety", paper_placement="Section 4.3")
    _copy_pair(source_main / "figF_autoskill_lifecycle", main / "fig4_autoskill_lifecycle", manifest, source_root=str(root), source_table="autoskill_skillbench_full/reports/autoskill_skillbench_summary.json", sample_size="2689 val / 576 test experiences", claim="AutoSkill-style replay, mutation, promotion, and test reuse", paper_placement="Section 4.3")
    _copy_pair(source_main / "figG_skill_explanation_quality_delta", main / "fig5_skill_quality_delta", manifest, source_root=str(root), source_table="autoskill_skillbench_full/tables/explanation_quality_delta_forest_ci.csv", sample_size="576 test anchors", claim="skill improves explanation-quality metrics", paper_placement="Section 4.3")
    generate_residual_memory_figure(root, main / "fig6_residual_memory_organization", manifest)
    generate_case_cards(root, main, manifest)
    generate_summary_coverage(root, appendix / "model_summary_coverage", manifest)
    generate_full_test_evidence_audit_package(root, dirs, manifest)
    generate_safe_abstention_table(root, appendix / "safe_abstention_matrix", manifest)

    active_sources = {
        "figA_full_split_prediction_landscape.png",
        "figC_event_active_adapter_gain.png",
        "figD_correction_safety_sparsity.png",
        "figF_autoskill_lifecycle.png",
        "figG_skill_explanation_quality_delta.png",
    }
    legacy_gate = root / "figures_legacy" / "fig7_event_station_channel_gate.png"
    if legacy_gate.is_file():
        _png_to_pdf(
            legacy_gate,
            appendix / "channel_gate_diagnostic",
            manifest,
            source_root=str(root),
            source_table="figures_legacy/fig7_event_station_channel_gate.png",
            sample_size="case diagnostic",
            claim="event-station-channel gate diagnostic not covered by aggregate figures",
            paper_placement="Appendix diagnostic only",
        )
        active_sources.add("fig7_event_station_channel_gate.png")

    _exclude_legacy(root, dirs, manifest, active_sources)

    manifest_path = dirs["paper"] / "manifest.csv"
    fields = ["filename", "status", "outputs", "source_root", "source_table", "sample_size", "claim", "paper_placement"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "figure_paper": str(dirs["paper"]),
        "main_count": len(list(dirs["main"].glob("*.png"))),
        "appendix_count": len(list(dirs["appendix"].glob("*.png"))),
        "excluded_legacy_count": len(list(dirs["excluded"].glob("*.png"))),
        "manifest": str(manifest_path),
    }
    (dirs["reports"] / "assembly_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (dirs["reports"] / "paper_figure_readme.md").write_text(
        "# Paper Figure Package\n\n"
        "This folder is the paper-facing figure set. Main figures use full-split statistics or polished case-card summaries. "
        "Old citation-funnel, redundant baseline, placeholder, and mixed-provenance figures are archived in `excluded_legacy/`.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.run_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
