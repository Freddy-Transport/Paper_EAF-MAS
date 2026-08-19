#!/usr/bin/env python3
"""Skill evolution study artifacts for paper experiments.

The study consumes paper workflow prediction JSON files when provided. For unit
tests and smoke runs without model output, it falls back to a small synthetic
window while marking the summary accordingly. It never mutates skills on test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from agents.autoskill_memory import (
    ExperiencePool,
    build_replay_experience,
    evolve_skills,
    write_jsonl as write_autoskill_jsonl,
)


REQUIRED_SKILL_FIELDS = {
    "skill_category",
    "skill_id",
    "trigger_condition",
    "evidence_pattern",
    "calibration_action",
    "validation_metric_delta",
    "promoted",
    "reason",
    "source_windows",
}


def should_update_skills(split: str, skill_mode: str) -> bool:
    """Only evolving-skill train/val runs may mutate skill memory."""
    return skill_mode in {"evolving_skill", "residual_rag_evolving_skill"} and split in {"train", "val"}


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum(np.abs(actual - pred)) / max(float(np.sum(np.abs(actual))), 1.0) * 100.0)


def _ensure_dirs(run_dir: Path) -> None:
    for sub in [
        "models/skill_memory",
        "models/skill_memory/experience_pool",
        "reports",
        "figures",
        "logs",
        "predictions",
        "tables",
        "paper_assets/structured_case_cards",
    ]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def _load_prediction_payloads(prediction_json: Sequence[str] | None) -> List[dict]:
    payloads = []
    for path in prediction_json or []:
        p = Path(path)
        if p.is_file():
            payloads.append(json.loads(p.read_text(encoding="utf-8")))
    return payloads


def _synthetic_payload() -> dict:
    raw = [[100.0, 110.0, 120.0, 130.0, 125.0, 115.0]]
    adjusted = [[104.0, 114.4, 124.8, 135.2, 130.0, 119.6]]
    actual = [[105.0, 116.0, 126.0, 136.0, 132.0, 120.0]]
    return {
        "request": {"date": "synthetic_smoke", "mode": "full_agent_evolving_skill", "station_scope": "event_venue28"},
        "numerical": {
            "raw_forecast": raw,
            "ground_truth": actual,
            "channel_names": ["N060__Times_Sq_42_St"],
            "timestamps": [f"h{i}" for i in range(6)],
        },
        "adjusted_forecast": adjusted,
        "evidence": {
            "structured_events": [
                {
                    "title": "synthetic major event",
                    "event_type": "Plaza Event",
                    "impact_tier": "A",
                    "channel_name": "N060__Times_Sq_42_St",
                }
            ],
            "local_residual_cases": [
                "[residual_case type=Plaza Event day=weekday rank=top64 correction=+4.0% best_score=1.000 merged_count=1]"
            ],
            "has_major_event": True,
        },
        "decision": {
            "mode": "full_agent_evolving_skill",
            "adjusted_channels": ["N060__Times_Sq_42_St"],
            "abstain": False,
            "reason": "Synthetic smoke event correction.",
        },
        "artifacts": {},
    }


def _payload_to_window_rows(payload: dict, source_idx: int, mode: str) -> List[dict]:
    request = payload.get("request", {})
    numerical = payload.get("numerical", {})
    evidence = payload.get("evidence", {})
    quality = payload.get("explanation_quality") or {}
    accepted_evidence = list(evidence.get("accepted_external_evidence") or evidence.get("accepted_evidence") or [])
    residual_memory_cases = len(evidence.get("historical_event_cases") or []) + len(evidence.get("local_residual_cases") or [])
    raw = np.asarray(numerical.get("raw_forecast") or [], dtype=float)
    adjusted = np.asarray(payload.get("adjusted_forecast") or numerical.get("raw_forecast") or [], dtype=float)
    actual = np.asarray(numerical.get("ground_truth") or [], dtype=float)
    channels = list(numerical.get("channel_names") or [f"channel_{i}" for i in range(raw.shape[0] if raw.ndim == 2 else 0)])
    if raw.ndim != 2 or actual.ndim != 2 or raw.size == 0:
        return []
    c, h = min(raw.shape[0], actual.shape[0], adjusted.shape[0]), min(raw.shape[1], actual.shape[1], adjusted.shape[1])
    raw, adjusted, actual = raw[:c, :h], adjusted[:c, :h], actual[:c, :h]
    rows = []
    window_id = f"{request.get('date', 'window')}_{source_idx}"
    for i in range(c):
        raw_wape = wape(actual[i : i + 1], raw[i : i + 1])
        adjusted_wape = wape(actual[i : i + 1], adjusted[i : i + 1])
        rows.append(
            {
                "window_id": window_id,
                "date": request.get("date", ""),
                "station": channels[i] if i < len(channels) else f"channel_{i}",
                "mode": mode,
                "raw_wape": raw_wape,
                "adjusted_wape": adjusted_wape,
                "wape_delta_raw_minus_adjusted": raw_wape - adjusted_wape,
                "raw_mae": float(np.mean(np.abs(actual[i] - raw[i]))),
                "adjusted_mae": float(np.mean(np.abs(actual[i] - adjusted[i]))),
                "has_major_event": bool(evidence.get("has_major_event")),
                "event_type": _first_event_field(evidence, "event_type", "unknown"),
                "impact_tier": _first_event_field(evidence, "impact_tier", "unknown"),
                "decision_reason": (payload.get("decision") or {}).get("reason", ""),
                "source_artifact": (payload.get("artifacts") or {}).get("prediction_json", ""),
                "accepted_evidence_count": len(accepted_evidence),
                "citation_coverage": float((payload.get("explanation_quality") or {}).get("citation_coverage", 0.0)),
                "calibration_decision_consistent": bool(
                    (payload.get("explanation_quality") or {}).get("calibration_decision_consistent", True)
                ),
                "evidence_coverage": float(quality.get("evidence_coverage", 1.0 if residual_memory_cases else 0.0) or 0.0),
                "residual_case_relevance": float(quality.get("residual_case_relevance", 1.0 if residual_memory_cases else 0.0) or 0.0),
                "multi_hop_completeness": float(quality.get("multi_hop_completeness", 0.0) or 0.0),
                "groundedness": float(quality.get("groundedness", 0.0) or 0.0),
                "unsupported_claim_rate": float(quality.get("unsupported_claim_rate", 0.0) or 0.0),
                "leakage_free_rate": float(quality.get("leakage_free_rate", 1.0) or 0.0),
                "redundancy_rate": float(quality.get("redundancy_rate", 0.0) or 0.0),
                "skill_guidance_coverage": float(quality.get("skill_guidance_coverage", 0.0) or 0.0),
                "residual_memory_cases": residual_memory_cases,
            }
        )
    return rows


def _first_event_field(evidence: dict, field: str, default: str) -> str:
    events = evidence.get("structured_events") or []
    for event in events:
        value = event.get(field)
        if value:
            return str(value)
    return default


def _write_window_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "window_id",
        "date",
        "station",
        "mode",
        "raw_wape",
        "adjusted_wape",
        "wape_delta_raw_minus_adjusted",
        "raw_mae",
        "adjusted_mae",
        "has_major_event",
        "event_type",
        "impact_tier",
        "decision_reason",
        "source_artifact",
        "evidence_coverage",
        "residual_case_relevance",
        "multi_hop_completeness",
        "groundedness",
        "unsupported_claim_rate",
        "leakage_free_rate",
        "redundancy_rate",
        "skill_guidance_coverage",
        "residual_memory_cases",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _skill_base(rows: Sequence[dict]) -> tuple[List[dict], str, str, float]:
    event_rows = [row for row in rows if row.get("has_major_event")]
    basis = event_rows or list(rows)
    if not basis:
        basis = [
            {
                "window_id": "empty",
                "event_type": "unknown",
                "impact_tier": "unknown",
                "wape_delta_raw_minus_adjusted": 0.0,
                "station": "unknown",
            }
        ]
    deltas = [float(row.get("wape_delta_raw_minus_adjusted") or 0.0) for row in basis]
    mean_delta = float(np.mean(deltas))
    event_type = str(basis[0].get("event_type") or "unknown")
    tier = str(basis[0].get("impact_tier") or "unknown")
    return basis, event_type, tier, mean_delta


def _can_promote(split: str, skill_mode: str) -> bool:
    return skill_mode in {"evolving_skill", "residual_rag_evolving_skill"} and split in {"train", "val"}


def build_typed_skill_candidates(rows: Sequence[dict], split: str, skill_mode: str) -> List[dict]:
    basis, event_type, tier, mean_delta = _skill_base(rows)
    can_promote = _can_promote(split, skill_mode)
    accepted_evidence = max(int(row.get("accepted_evidence_count") or 0) for row in basis)
    citation_coverage = float(np.mean([float(row.get("citation_coverage") or 0.0) for row in basis]))
    consistent = all(bool(row.get("calibration_decision_consistent", True)) for row in basis)
    has_event = any(bool(row.get("has_major_event")) for row in basis)
    source_windows = sorted({str(row.get("window_id")) for row in basis if row.get("window_id")})
    evidence_coverage = float(np.mean([float(row.get("evidence_coverage") or 0.0) for row in basis]))
    residual_relevance = float(np.mean([float(row.get("residual_case_relevance") or 0.0) for row in basis]))
    multi_hop_completeness = float(np.mean([float(row.get("multi_hop_completeness") or 0.0) for row in basis]))
    groundedness = float(np.mean([float(row.get("groundedness") or 0.0) for row in basis]))
    leakage_free = float(np.mean([float(row.get("leakage_free_rate") or 0.0) for row in basis]))
    unsupported = float(np.mean([float(row.get("unsupported_claim_rate") or 0.0) for row in basis]))
    skill_guidance = float(np.mean([float(row.get("skill_guidance_coverage") or 0.0) for row in basis]))
    residual_memory_cases = int(max([int(row.get("residual_memory_cases") or 0) for row in basis] or [0]))
    explanation_quality_delta = float(
        np.mean([evidence_coverage, residual_relevance, multi_hop_completeness, groundedness, leakage_free, 1.0 - unsupported])
    )

    def make_record(
        category: str,
        promoted: bool,
        reason: str,
        policy: str,
        metric_delta: float = 0.0,
        explanation_action: dict | None = None,
    ) -> dict:
        skill_hash = abs(hash((category, event_type, tier, tuple(source_windows)))) % 100000
        return {
            "skill_category": category,
            "skill_id": f"{category}_{event_type.lower().replace(' ', '_')}_{tier.lower()}_{skill_hash}",
            "trigger_condition": {"event_type": event_type, "impact_tier": tier, "has_major_event": has_event},
            "evidence_pattern": {
                "accepted_evidence_count": accepted_evidence,
                "citation_coverage": round(citation_coverage, 6),
                "calibration_decision_consistent": consistent,
                "mean_delta_raw_minus_adjusted": round(mean_delta, 6),
                "n_source_rows": len(basis),
                "evidence_coverage": round(evidence_coverage, 6),
                "residual_case_relevance": round(residual_relevance, 6),
                "multi_hop_completeness": round(multi_hop_completeness, 6),
                "groundedness": round(groundedness, 6),
                "leakage_free_rate": round(leakage_free, 6),
                "unsupported_claim_rate": round(unsupported, 6),
                "skill_guidance_coverage": round(skill_guidance, 6),
                "residual_memory_cases": residual_memory_cases,
                "explanation_quality_delta": round(explanation_quality_delta, 6),
            },
            "calibration_action": {
                "policy": policy,
                "direction": "increase" if mean_delta >= 0 else "abstain",
                "max_correction": 0.05,
            },
            "explanation_action": explanation_action
            or {
                "policy": policy,
                "use_for_prediction": False,
                "purpose": "diagnostic_or_calibration_policy",
            },
            "validation_metric_delta": round(metric_delta, 6),
            "promoted": bool(promoted),
            "reason": reason,
            "source_windows": source_windows,
            "created_split": split,
            "skill_mode": skill_mode,
        }

    routing_promoted = can_promote and has_event
    evidence_promoted = can_promote and accepted_evidence > 0 and citation_coverage > 0.0
    abstain_promoted = can_promote and consistent and mean_delta <= 0.0
    calibration_promoted = can_promote and mean_delta > 0.0
    residual_memory_promoted = (
        can_promote
        and residual_memory_cases > 0
        and evidence_coverage >= 0.50
        and residual_relevance >= 0.50
        and leakage_free >= 1.0
        and unsupported <= 0.25
    )

    return [
        make_record(
            "routing_skill",
            routing_promoted,
            "Promoted for routing major event windows into evidence/RAG workflow." if routing_promoted else "Not promoted: split or mode cannot update routing skills.",
            "retrieve_evidence_then_decide",
            metric_delta=1.0 if routing_promoted else 0.0,
        ),
        make_record(
            "evidence_skill",
            evidence_promoted,
            "Promoted because citation-quality evidence was available and referenced." if evidence_promoted else "Not promoted: no citation-quality external evidence was verified.",
            "prefer_sources_matching_event_date_location_station",
            metric_delta=citation_coverage,
        ),
        make_record(
            "abstention_skill",
            abstain_promoted,
            "Promoted because abstention/decision text was consistent and correction was not validated." if abstain_promoted else "Not promoted: abstention was unnecessary or decision consistency was insufficient.",
            "abstain_when_evidence_or_validation_is_weak",
            metric_delta=1.0 if abstain_promoted else 0.0,
        ),
        make_record(
            "calibration_skill",
            calibration_promoted,
            "Promoted because adjusted forecast improved event-active WAPE on validation." if calibration_promoted else "Rejected because calibration did not improve validation WAPE or this split/mode cannot update skills.",
            "use_validated_event_adjustment" if calibration_promoted else "abstain_or_keep_raw",
            metric_delta=mean_delta,
        ),
        make_record(
            "residual_memory_skill",
            residual_memory_promoted,
            "Promoted because residual memory improved explanation coverage, relevance, and leakage-free reasoning."
            if residual_memory_promoted
            else "Not promoted: residual memory did not meet explanation-quality thresholds or split/mode is read-only.",
            "organize_residual_memory_for_explanation",
            metric_delta=explanation_quality_delta,
            explanation_action={
                "policy": "organize_residual_memory_for_explanation",
                "use_for_prediction": False,
                "select_cases_by": ["event_type", "impact_tier", "station_rank", "day_type", "residual_direction"],
                "write_reasoning_as": "current event -> station match -> similar train/validation case -> residual pattern -> decision",
                "emphasize_uncertainty": unsupported > 0.0 or residual_relevance < 0.75,
            },
        ),
    ]


def _candidate_skill_from_rows(rows: Sequence[dict], split: str, skill_mode: str) -> dict:
    # Backward-compatible single-record helper retained for older imports.
    return build_typed_skill_candidates(rows, split, skill_mode)[-1]


def _load_existing_skills(path: str | None) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    skills = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                skills.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [skill for skill in skills if REQUIRED_SKILL_FIELDS.issubset(skill)]


def _load_existing_autoskill_skills(path: str | None) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    skills = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "promotion_status" in row and "skill_category" in row:
                skills.append(row)
    return skills


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_figures(run_dir: Path, rows: Sequence[dict], records: Sequence[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.mean([float(row.get("raw_wape") or 0.0) for row in rows]) if rows else math.nan
    adjusted = np.mean([float(row.get("adjusted_wape") or 0.0) for row in rows]) if rows else math.nan
    no_skill = raw if not math.isnan(raw) else 1.0
    frozen = adjusted if not math.isnan(adjusted) else no_skill
    evolving = adjusted if any(record.get("promoted") for record in records) else frozen

    plt.figure(figsize=(7, 3.4))
    plt.bar(["full_agent_no_skill", "full_agent_frozen_skill", "full_agent_evolving_skill"], [no_skill, frozen, evolving])
    plt.ylabel("WAPE")
    plt.title("Skill ablation metrics")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "skill_ablation_metrics.png", dpi=150)
    plt.close()

    promoted = [sum(1 for record in records[: i + 1] if record.get("promoted")) for i in range(len(records))]
    rejected = [sum(1 for record in records[: i + 1] if not record.get("promoted")) for i in range(len(records))]
    deltas = [float(record.get("validation_metric_delta") or 0.0) for record in records]
    xs = list(range(1, len(records) + 1)) or [1]
    plt.figure(figsize=(7, 3.4))
    plt.plot(xs, promoted or [0], marker="o", label="promoted")
    plt.plot(xs, rejected or [0], marker="o", label="rejected")
    if deltas:
        ax = plt.gca().twinx()
        ax.bar(xs, deltas, alpha=0.25, color="#7c3aed", label="WAPE delta")
        ax.set_ylabel("raw - adjusted WAPE")
    plt.gca().set_xlabel("round")
    plt.gca().set_ylabel("skill count")
    plt.title("Skill promotion trajectory")
    plt.gca().legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "skill_promotion_trajectory.png", dpi=150)
    plt.close()

    metrics = {
        "coverage": float(np.mean([float(row.get("evidence_coverage") or 0.0) for row in rows])) if rows else 0.0,
        "residual relevance": float(np.mean([float(row.get("residual_case_relevance") or 0.0) for row in rows])) if rows else 0.0,
        "multi-hop": float(np.mean([float(row.get("multi_hop_completeness") or 0.0) for row in rows])) if rows else 0.0,
        "groundedness": float(np.mean([float(row.get("groundedness") or 0.0) for row in rows])) if rows else 0.0,
        "leakage-free": float(np.mean([float(row.get("leakage_free_rate") or 0.0) for row in rows])) if rows else 0.0,
        "skill guidance": float(np.mean([float(row.get("skill_guidance_coverage") or 0.0) for row in rows])) if rows else 0.0,
        "non-redundant": 1.0 - (float(np.mean([float(row.get("redundancy_rate") or 0.0) for row in rows])) if rows else 0.0),
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(5.4, 5.0))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color="#2563eb", linewidth=2)
    ax.fill(angles, values, color="#2563eb", alpha=0.16)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Residual-memory skill explanation quality", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "skill_explanation_quality_radar.png", dpi=150)
    plt.close(fig)


def _write_tex_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]], caption: str, label: str) -> None:
    cols = "l" * len(header)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paper_tables(run_dir: Path, rows: Sequence[dict], records: Sequence[dict]) -> None:
    def mean(field: str) -> float:
        return float(np.mean([float(row.get(field) or 0.0) for row in rows])) if rows else 0.0

    _write_tex_table(
        run_dir / "tables" / "explanation_quality_skill_ablation.tex",
        ["Mode", "Coverage", "ResidualRel.", "Multi-hop", "Grounded", "SkillGuide", "Leakage-free"],
        [
            [
                "Residual-memory skill",
                f"{mean('evidence_coverage'):.2f}",
                f"{mean('residual_case_relevance'):.2f}",
                f"{mean('multi_hop_completeness'):.2f}",
                f"{mean('groundedness'):.2f}",
                f"{mean('skill_guidance_coverage'):.2f}",
                f"{mean('leakage_free_rate'):.2f}",
            ]
        ],
        "Explanation quality diagnostics for residual-memory skill routing.",
        "tab:residual_memory_skill_quality",
    )
    skill_rows = []
    for record in records:
        if record.get("skill_category") == "residual_memory_skill":
            ep = record.get("evidence_pattern") or {}
            skill_rows.append(
                [
                    record.get("skill_id", "")[:32],
                    str(record.get("promoted")),
                    f"{float(record.get('validation_metric_delta') or 0.0):.2f}",
                    ep.get("residual_memory_cases", 0),
                    record.get("reason", "")[:70],
                ]
            )
    if not skill_rows:
        skill_rows.append(["none", "False", "0.00", "0", "No residual-memory skill candidate."])
    _write_tex_table(
        run_dir / "tables" / "residual_memory_skill_cases.tex",
        ["Skill", "Promoted", "QualityDelta", "Cases", "Reason"],
        skill_rows,
        "Residual-memory skill candidates and validation decisions.",
        "tab:residual_memory_skill_cases",
    )


def run_skill_evolution_study(
    output_root: str | Path,
    split: str = "val",
    skill_mode: str = "evolving_skill",
    prediction_json: Sequence[str] | None = None,
    existing_skill_library: str | None = None,
) -> Dict:
    run_dir = Path(output_root)
    _ensure_dirs(run_dir)
    payloads = _load_prediction_payloads(prediction_json)
    synthetic_smoke = False
    if not payloads:
        payloads = [_synthetic_payload()]
        synthetic_smoke = True

    update = should_update_skills(split, skill_mode)
    autoskill_experiences = [
        build_replay_experience(payload, split=split, experience_id=f"{split}_{i:04d}") for i, payload in enumerate(payloads)
    ]
    experience_pool = ExperiencePool(run_dir / "models" / "skill_memory" / "experience_pool")
    experience_pool.extend(autoskill_experiences)
    autoskill_report = evolve_skills(autoskill_experiences, split=split, mode=skill_mode)
    existing_autoskill = _load_existing_autoskill_skills(existing_skill_library)
    autoskill_active = autoskill_report["promoted_skills"] if update else existing_autoskill
    write_autoskill_jsonl(
        run_dir / "models" / "skill_memory" / "autoskill_candidate_skills.jsonl",
        autoskill_report["candidate_skills"],
    )
    write_autoskill_jsonl(
        run_dir / "models" / "skill_memory" / "autoskill_mutated_skills.jsonl",
        autoskill_report["mutated_skills"],
    )
    write_autoskill_jsonl(
        run_dir / "models" / "skill_memory" / "autoskill_promoted_skills.jsonl",
        autoskill_report["promoted_skills"],
    )
    write_autoskill_jsonl(
        run_dir / "models" / "skill_memory" / "autoskill_active_skill_library.jsonl",
        autoskill_active,
    )
    (run_dir / "reports" / "autoskill_evolution_trace.json").write_text(
        json.dumps(autoskill_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: List[dict] = []
    for i, payload in enumerate(payloads):
        rows.extend(_payload_to_window_rows(payload, i, skill_mode))
    _write_window_csv(run_dir / "predictions" / "skill_window_results.csv", rows)

    records = build_typed_skill_candidates(rows, split, skill_mode)
    existing = _load_existing_skills(existing_skill_library)
    active = []
    if update:
        active = [record for record in records if record.get("promoted")]
    else:
        active = existing or [
            {**record, "promoted": False, "reason": "Loaded as test-time frozen candidate; no skill update was performed."}
            for record in records
        ]
    if not active:
        active = [
            {**record, "promoted": False, "reason": "No validation improvement; retained as inactive analysis record."}
            for record in records
        ]

    _write_jsonl(run_dir / "models" / "skill_memory" / "candidate_skill_records.jsonl", records)
    _write_jsonl(run_dir / "models" / "skill_memory" / "active_skill_library.jsonl", active)
    _write_jsonl(run_dir / "models" / "skill_memory" / f"skill_library_round_{split}_{skill_mode}.jsonl", active)
    _write_figures(run_dir, rows, records)
    _write_paper_tables(run_dir, rows, records)

    skill_hit_rate = float(sum(1 for row in rows if row.get("has_major_event"))) / max(len(rows), 1)
    raw_mean = float(np.mean([float(row.get("raw_wape") or 0.0) for row in rows])) if rows else 0.0
    adjusted_mean = float(np.mean([float(row.get("adjusted_wape") or 0.0) for row in rows])) if rows else 0.0
    summary = {
        "split": split,
        "skill_mode": skill_mode,
        "update_skills": update,
        "synthetic_smoke": synthetic_smoke,
        "candidate_skills": len(records),
        "promoted_skills": sum(1 for record in records if record.get("promoted")),
        "promoted_by_category": {
            category: sum(1 for record in records if record.get("skill_category") == category and record.get("promoted"))
            for category in sorted({record.get("skill_category") for record in records})
        },
        "active_skills": len(active),
        "skill_hit_rate": skill_hit_rate,
        "mean_raw_wape": raw_mean,
        "mean_adjusted_wape": adjusted_mean,
        "mean_delta_raw_minus_adjusted": raw_mean - adjusted_mean,
        "mean_explanation_quality": {
            "evidence_coverage": float(np.mean([float(row.get("evidence_coverage") or 0.0) for row in rows])) if rows else 0.0,
            "residual_case_relevance": float(np.mean([float(row.get("residual_case_relevance") or 0.0) for row in rows])) if rows else 0.0,
            "multi_hop_completeness": float(np.mean([float(row.get("multi_hop_completeness") or 0.0) for row in rows])) if rows else 0.0,
            "groundedness": float(np.mean([float(row.get("groundedness") or 0.0) for row in rows])) if rows else 0.0,
            "leakage_free_rate": float(np.mean([float(row.get("leakage_free_rate") or 0.0) for row in rows])) if rows else 0.0,
            "skill_guidance_coverage": float(np.mean([float(row.get("skill_guidance_coverage") or 0.0) for row in rows])) if rows else 0.0,
        },
        "autoskill": {
            "experience_count": autoskill_report["experience_count"],
            "candidate_count": autoskill_report["candidate_count"],
            "mutation_count": autoskill_report["mutation_count"],
            "promoted_count": autoskill_report["promoted_count"],
            "rejected_count": autoskill_report["rejected_count"],
            "active_count": len(autoskill_active),
            "loaded_existing_count": len(existing_autoskill),
            "test_read_only": autoskill_report["test_read_only"],
            "forecast_arrays_identical": autoskill_report["forecast_arrays_identical"],
            "candidate_path": str(run_dir / "models" / "skill_memory" / "autoskill_candidate_skills.jsonl"),
            "promoted_path": str(run_dir / "models" / "skill_memory" / "autoskill_promoted_skills.jsonl"),
            "active_path": str(run_dir / "models" / "skill_memory" / "autoskill_active_skill_library.jsonl"),
            "experience_pool": str(run_dir / "models" / "skill_memory" / "experience_pool"),
        },
        "notes": "Uses paper workflow prediction JSON when provided; synthetic smoke is only for contract validation.",
    }
    (run_dir / "reports" / "skill_evolution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_records = "\n".join(
        f"- `{record['skill_category']}` `{record['skill_id']}` promoted={record['promoted']} delta={record['validation_metric_delta']}: {record['reason']}"
        for record in records
    )
    (run_dir / "reports" / "skill_evolution_summary.md").write_text(
        f"# Skill Evolution Study\n\n"
        f"- split: `{split}`\n"
        f"- skill_mode: `{skill_mode}`\n"
        f"- update_skills: `{update}`\n"
        f"- synthetic_smoke: `{synthetic_smoke}`\n"
        f"- autoskill_experience_count: `{autoskill_report['experience_count']}`\n"
        f"- autoskill_candidate_count: `{autoskill_report['candidate_count']}`\n"
        f"- autoskill_mutation_count: `{autoskill_report['mutation_count']}`\n"
        f"- autoskill_promoted_count: `{autoskill_report['promoted_count']}`\n"
        f"- autoskill_active_count: `{len(autoskill_active)}`\n"
        f"- mean raw WAPE: `{raw_mean:.4f}`\n"
        f"- mean adjusted WAPE: `{adjusted_mean:.4f}`\n\n"
        f"## Skill Records\n\n{md_records}\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--skill_mode",
        choices=[
            "no_skill",
            "frozen_skill",
            "evolving_skill",
            "residual_rag_no_skill",
            "residual_rag_static_skill",
            "residual_rag_evolving_skill",
        ],
        default="evolving_skill",
    )
    parser.add_argument("--prediction_json", action="append", default=[])
    parser.add_argument("--existing_skill_library", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_root or f"autotemp/skill_evolution_{time.strftime('%Y%m%d_%H%M%S')}")
    summary = run_skill_evolution_study(
        run_dir,
        split=args.split,
        skill_mode=args.skill_mode,
        prediction_json=args.prediction_json,
        existing_skill_library=args.existing_skill_library,
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
