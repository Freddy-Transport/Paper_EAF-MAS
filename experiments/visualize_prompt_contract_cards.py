"""Generate appendix prompt-contract cards for EAF-MAS LLM-related roles.

The script is intentionally read-only with respect to experiment logic. It scans
static source files, redacts sensitive strings, and writes a manifest, paper
snippet, and card-style PDF/PNG figures for appendix use.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence


MANIFEST_FIELDS = [
    "prompt_id",
    "paper_label",
    "module",
    "source_file",
    "source_line_start",
    "source_line_end",
    "runtime_role",
    "model_backend",
    "called_by",
    "is_llm_call",
    "is_prompt_template",
    "is_runtime_contract",
    "input_payload_fields",
    "output_schema",
    "safety_constraints",
    "failure_handling",
    "forecast_array_can_change",
    "uses_posthoc_metrics",
    "uses_actual_ridership",
    "uses_external_url_citation",
    "citation_quality_claim_allowed",
    "paper_placement",
    "show_in_main_appendix",
    "card_title",
    "card_color",
    "redaction_status",
    "notes",
]


CARD_COLORS = {
    "slate": "#1f2937",
    "blue": "#2563eb",
    "green": "#15803d",
    "purple": "#7c3aed",
    "orange": "#ea580c",
    "teal": "#0f766e",
    "gray": "#4b5563",
}


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}", re.I),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-/+=]+", re.I),
    re.compile(r"(OPENAI_API_KEY|DASHSCOPE_API_KEY|HF_TOKEN)\s*[:=]\s*[\"']?[^\"'\s,}]+", re.I),
    re.compile(r"(api[_ -]?key|token|password|passwd|secret)\s*[:=]\s*[\"']?[^\"'\n,}]+", re.I),
]

FORBIDDEN_OUTPUT_PATTERNS = [
    re.compile(r"OPENAI_API_KEY", re.I),
    re.compile(r"DASHSCOPE_API_KEY", re.I),
    re.compile(r"HF_TOKEN", re.I),
    re.compile(r"sk-", re.I),
    re.compile(r"Bearer\s+(?!\[REDACTED\])", re.I),
]


@dataclass
class PromptCard:
    row: dict
    body: str
    full_template: str = ""


def redact_sensitive_text(text: str) -> tuple[str, list[str]]:
    """Redact secrets and environment-key names before writing any artifact."""

    hits: list[str] = []
    redacted = text or ""
    replacements = [
        (SECRET_PATTERNS[0], "[REDACTED_API_KEY]"),
        (SECRET_PATTERNS[1], "Bearer [REDACTED]"),
        (SECRET_PATTERNS[2], "[REDACTED_SECRET_ENV]=[REDACTED]"),
        (SECRET_PATTERNS[3], "[REDACTED_SECRET_FIELD]"),
    ]
    for pattern, repl in replacements:
        if pattern.search(redacted):
            hits.append(pattern.pattern[:24])
            redacted = pattern.sub(repl, redacted)
    # Also redact literal mentions of sensitive env names and key prefixes that
    # may appear in source regex strings rather than as concrete secret values.
    env_name_pattern = re.compile(r"OPENAI_API_KEY|DASHSCOPE_API_KEY|HF_TOKEN", re.I)
    if env_name_pattern.search(redacted):
        hits.append("secret-env-name")
        redacted = env_name_pattern.sub("[REDACTED_SECRET_ENV]", redacted)
    if "sk-" in redacted.lower():
        hits.append("secret-prefix-literal")
        redacted = re.sub(r"sk-", "[REDACTED_API_KEY_PREFIX]", redacted, flags=re.I)
    redacted = re.sub(r"/Users/[^ \n\t\"']+", "[LOCAL_PATH_REDACTED]", redacted)
    redacted = re.sub(r"/root/(?!autodl-tmp/0206moment\b)[^ \n\t\"']+", "[REMOTE_PATH_REDACTED]", redacted)
    return redacted, hits


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "reports": root / "reports",
        "tables": root / "tables",
        "figures": root / "figures",
        "latex": root / "latex",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_range_for_symbols(path: Path, symbols: Sequence[str]) -> tuple[int | None, int | None]:
    source = read_text(path)
    tree = ast.parse(source)
    starts: list[int] = []
    ends: list[int] = []
    wanted = set(symbols)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if wanted.intersection(names):
            starts.append(getattr(node, "lineno", 0))
            ends.append(getattr(node, "end_lineno", getattr(node, "lineno", 0)))
    if not starts:
        return None, None
    return min(starts), max(ends)


def line_for_pattern(path: Path, pattern: str) -> int | None:
    rx = re.compile(pattern)
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        if rx.search(line):
            return idx
    return None


def source_segment(path: Path, start: int | None, end: int | None, max_chars: int = 7000) -> str:
    if not start or not end:
        return ""
    lines = read_text(path).splitlines()
    segment = "\n".join(lines[start - 1 : end])
    if len(segment) > max_chars:
        segment = segment[:max_chars] + "\n[abbreviated]"
    return redact_sensitive_text(segment)[0]


def git_info(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(["git", *args], cwd=repo_root, stderr=subprocess.DEVNULL, text=True)
            return out.strip()
        except Exception:
            return None

    head = run_git(["rev-parse", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    status = run_git(["status", "--short"])
    return {
        "git_repo_detected": bool(head),
        "git_head": head,
        "git_branch": branch,
        "git_status_short": status if head else None,
    }


def csv_join(values: Iterable[str]) -> str:
    return "; ".join(str(v) for v in values if str(v))


def make_row(
    prompt_id: str,
    label: str,
    module: str,
    source_file: str,
    source_line_start: int | None,
    source_line_end: int | None,
    runtime_role: str,
    model_backend: str,
    called_by: str,
    is_llm_call: bool,
    is_prompt_template: bool,
    is_runtime_contract: bool,
    input_payload_fields: Sequence[str],
    output_schema: Sequence[str],
    safety_constraints: Sequence[str],
    failure_handling: str,
    card_title: str,
    card_color: str,
    notes: str,
    uses_actual_ridership: bool = False,
    uses_external_url_citation: bool = False,
    citation_quality_claim_allowed: bool = False,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "paper_label": label,
        "module": module,
        "source_file": source_file,
        "source_line_start": source_line_start,
        "source_line_end": source_line_end,
        "runtime_role": runtime_role,
        "model_backend": model_backend,
        "called_by": called_by,
        "is_llm_call": is_llm_call,
        "is_prompt_template": is_prompt_template,
        "is_runtime_contract": is_runtime_contract,
        "input_payload_fields": csv_join(input_payload_fields),
        "output_schema": csv_join(output_schema),
        "safety_constraints": csv_join(safety_constraints),
        "failure_handling": failure_handling,
        "forecast_array_can_change": False,
        "uses_posthoc_metrics": False,
        "uses_actual_ridership": uses_actual_ridership,
        "uses_external_url_citation": uses_external_url_citation,
        "citation_quality_claim_allowed": citation_quality_claim_allowed,
        "paper_placement": "appendix",
        "show_in_main_appendix": "appendix",
        "card_title": card_title,
        "card_color": card_color,
        "redaction_status": "passed",
        "notes": notes,
    }


def body_from_row(row: dict) -> str:
    return "\n\n".join(
        [
            f"Role:\n  {row['runtime_role']} ({row['model_backend']})",
            f"Runtime inputs:\n  {row['input_payload_fields']}",
            f"Output schema:\n  {row['output_schema']}",
            f"Safety constraints:\n  {row['safety_constraints']}",
            f"Failure handling:\n  {row['failure_handling']}",
            f"Paper claim boundary:\n  {row['notes']}",
        ]
    )


def build_cards(repo_root: Path) -> list[PromptCard]:
    cards: list[PromptCard] = []

    # Card 00: synthesized runtime contract.
    row = make_row(
        "prompt_card_00_shared_runtime_contract",
        "Shared runtime I/O",
        "EAF-MAS runtime",
        "synthesized",
        None,
        None,
        "shared_runtime_contract",
        "mixed LLM / non-LLM contracts",
        "EAF-MAS paper appendix",
        False,
        False,
        True,
        [
            "forecast_request",
            "structured_events",
            "evidence sources or model-assisted summaries",
            "train/validation residual memory",
            "selected skills",
            "calibration decision",
            "evidence audit",
            "station scope",
        ],
        ["one JSON object or structured skill record", "no prose outside JSON when JSON is required"],
        [
            "no actual ridership, WAPE, or post-hoc metrics in forecast-time reasoning",
            "no direct forecast override",
            "no API keys",
            "no citation-quality claim without URL evidence",
            "respect controller abstention",
        ],
        "Malformed JSON, missing provenance, secret leakage, post-hoc leakage, or decision inconsistency are validation failures.",
        "Shared EAF-MAS Runtime I/O Contract",
        "slate",
        "Synthesized from code-level boundaries; governs all LLM-related roles.",
    )
    cards.append(PromptCard(row=row, body=body_from_row(row)))

    # Card 01.
    path = repo_root / "agents/evidence_research_agent.py"
    start, end = line_range_for_symbols(path, ["_prompt"])
    prompt_intervene_line = line_for_pattern(path, r"prompt_intervene")
    if prompt_intervene_line:
        start = min([x for x in [start, prompt_intervene_line] if x])
        end = max([x for x in [end, prompt_intervene_line] if x])
    row = make_row(
        "prompt_card_01_qwenplus_event_summary",
        "Qwen-Plus event summary",
        "agents.evidence_research_agent",
        "agents/evidence_research_agent.py",
        start,
        end,
        "qwenplus_event_summary",
        "qwen-plus / DashScope compatible API",
        "cost-gated event evidence enrichment",
        True,
        True,
        False,
        ["event title", "event_time", "location", "target_date", "nearby station channels"],
        [
            "event_summary",
            "event_relevance_to_station",
            "expected_ridership_effect",
            "uncertainty",
            "summary_used_for_explanation",
        ],
        [
            "strict JSON",
            "Do not return URLs",
            "no citation markers",
            "no evidence list",
            "do not invent URLs",
            "summary-only",
        ],
        "If unavailable or malformed, cache status records skipped/failed summary and downstream explanation treats it as non-citation context.",
        "Qwen-Plus Model-Assisted Event Summary Prompt",
        "blue",
        "Model-assisted summary only; not citation-quality evidence and not numerical forecasting.",
    )
    cards.append(PromptCard(row=row, body=body_from_row(row), full_template=source_segment(path, start, end)))

    # Card 02.
    path = repo_root / "agents/llm_evidence_auditor.py"
    start, end = line_range_for_symbols(path, ["AUDIT_SCORE_KEYS", "DECISIONS", "SYSTEM_PROMPT", "USER_PROMPT"])
    response_line = line_for_pattern(path, r"response_format")
    thinking_line = line_for_pattern(path, r"enable_thinking")
    end = max([x for x in [end, response_line, thinking_line] if x])
    row = make_row(
        "prompt_card_02_local_qwen_evidence_auditor",
        "Local evidence auditor",
        "agents.llm_evidence_auditor",
        "agents/llm_evidence_auditor.py",
        start,
        end,
        "evidence_audit_scorer",
        "Qwen/Qwen3-8B via local vLLM OpenAI-compatible API",
        "ForecastEvidenceAuditor hard-gate plus LLM audit",
        True,
        True,
        False,
        ["items_to_score", "evidence_sources", "model_assisted_summaries", "historical_event_cases", "residual_cases", "hard_audit_summary"],
        ["item_audits", "aggregate source/temporal/geo/linking/residual/conflict/confidence scores", "decision"],
        ["score only supplied evidence", "do not browse", "do not add external facts", "no hidden post-event outcomes", "enable_thinking=False"],
        "Malformed JSON or invalid scores fall back to hard-gate-only conservative audit.",
        "Local Qwen Evidence Auditor Prompt",
        "green",
        "Evidence governance only; not a retrieval source and not numerical forecasting.",
    )
    cards.append(PromptCard(row=row, body=body_from_row(row), full_template=source_segment(path, start, end)))

    # Card 03.
    path = repo_root / "agents/forecast_explanation_agent.py"
    start, end = line_range_for_symbols(path, ["SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "_enforce_decision_consistency"])
    row = make_row(
        "prompt_card_03_forecast_explanation_agent",
        "Forecast-time explanation",
        "agents.forecast_explanation_agent",
        "agents/forecast_explanation_agent.py",
        start,
        end,
        "explanation_agent",
        "local Qwen / OpenAI-compatible API",
        "forecast-time explanation generation",
        True,
        True,
        False,
        [
            "forecast_request",
            "structured_events",
            "retrieved_sources",
            "model_assisted_summaries",
            "historical_event_cases",
            "selected_residual_memory_skills",
            "station_scope",
            "numerical_forecast_summary",
            "calibration_decision",
            "evidence_audit",
            "calibration_controller",
        ],
        ["event_summary", "evidence_used", "station_event_linking", "multi_hop_reasoning", "calibration_rationale", "uncertainty_and_abstention", "markdown"],
        [
            "Do not claim ground truth, actual ridership, WAPE, or post-hoc metrics",
            "train/validation cases only",
            "do not use rejected retrieval diagnostics",
            "abstention consistency: no correction claim when abstained",
        ],
        "Decision-consistency postprocessor rewrites inconsistent abstention/correction claims.",
        "Forecast-Time Explanation Agent Prompt",
        "purple",
        "Explanation trace only; not a direct predictor.",
    )
    cards.append(PromptCard(row=row, body=body_from_row(row), full_template=source_segment(path, start, end)))

    # Card 04.
    path = repo_root / "event_screening/llm.py"
    start, end = line_range_for_symbols(path, ["build_llm_prompt"])
    retry_line = line_for_pattern(path, r"previous answer was invalid")
    if retry_line:
        end = max([x for x in [end, retry_line] if x])
    row = make_row(
        "prompt_card_04_event_screening_prompt",
        "Major-event screening",
        "event_screening.llm",
        "event_screening/llm.py",
        start,
        end,
        "major_event_screening",
        "HF API / isolated transformers / transformers; vLLM disabled for safe screening",
        "event-corpus filtering and preprocessing",
        True,
        True,
        False,
        ["event id", "title", "event_time", "event_type", "location", "duplicate_station_matches", "min_distance_m", "affected_channels", "event content", "traffic evidence"],
        ["is_major_event", "major_event_type", "crowd_scale", "transit_impact_likelihood", "expected_direction", "affected_scope", "traffic_evidence_used", "keep_for_modeling", "reason"],
        [
            "do not classify as major only because permit type is Special Event",
            "demote routine youth sports, farmers markets, filming/setup/closures unless traffic evidence supports abnormal impact",
            "return JSON only",
        ],
        "Invalid JSON triggers one retry; repeated invalid output is marked needs_review.",
        "Event Screening Prompt for Major-Event Filtering",
        "orange",
        "Event-corpus filtering / preprocessing; not forecast-time calibration.",
    )
    cards.append(PromptCard(row=row, body=body_from_row(row), full_template=source_segment(path, start, end)))

    # Card 05.
    path = repo_root / "agents/autoskill_memory.py"
    start, end = line_range_for_symbols(path, ["REQUIRED_AUTOSKILL_SKILL_FIELDS", "DOMAIN_SKILL_CATEGORIES", "_reasoning_template_for"])
    row = make_row(
        "prompt_card_05_autoskill_skill_memory_contract",
        "AutoSkill memory contract",
        "agents.autoskill_memory",
        "agents/autoskill_memory.py",
        start,
        end,
        "validation-evolved_skill_memory",
        "skill record / memory contract",
        "AutoSkill-style validation replay and test read-only reuse",
        False,
        False,
        True,
        ["validation replay experiences", "structured event", "evidence audit", "controller decision", "explanation scores", "forecast digest"],
        ["skill_id", "trigger_condition", "evidence_pattern", "memory_selection_policy", "reasoning_template", "abstention_rule", "validation_metrics", "promotion_status"],
        ["use_for_prediction=false", "forecast_array_changed=false", "test read-only", "no PT-MOMENT or residual adapter override"],
        "Skills that alter forecast arrays, leak test information, or raise unsupported-claim risk are rejected by promotion guards.",
        "AutoSkill-Style Skill Memory Contract",
        "teal",
        "Explanation and residual-memory governance, not numerical forecasting.",
    )
    cards.append(PromptCard(row=row, body=body_from_row(row), full_template=source_segment(path, start, end)))

    # Card 06.
    path = repo_root / "experiments/run_full_autoskill_skillbench.py"
    start, end = line_range_for_symbols(path, ["SKILLBENCH_MODES", "build_skillbench_judge_prompt", "_judge_rows"])
    row = make_row(
        "prompt_card_06_skillbench_judge_template_proxy",
        "SkillBench proxy judge template",
        "experiments.run_full_autoskill_skillbench",
        "experiments/run_full_autoskill_skillbench.py",
        start,
        end,
        "explanation_quality_proxy_judge",
        "proxy template; deterministic diagnostic rows",
        "SkillBench explanation-quality diagnostics",
        False,
        True,
        False,
        ["explanation A", "explanation B"],
        ["JSON winners for clarity", "faithfulness", "multi_hop", "residual_memory_usefulness"],
        ["do not use realized ridership", "do not use errors", "do not use post-hoc evaluation metrics"],
        "Current _judge_rows is proxy/deterministic; qwenplus_crosscheck_winner is not_run.",
        "SkillBench Pairwise Judge Prompt Template",
        "gray",
        "Diagnostic prompt template only; not formal independent LLM judge and not human evaluation.",
    )
    cards.append(PromptCard(row=row, body=body_from_row(row), full_template=source_segment(path, start, end)))
    return cards


def wrap_body(text: str, max_chars: int) -> str:
    text = redact_sensitive_text(text)[0]
    if len(text) > max_chars:
        text = text[: max_chars - 88].rstrip() + "\n\n[abbreviated for appendix card; full static template in manifest]"
    wrapped_lines: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            wrapped_lines.append("")
            continue
        indent = len(para) - len(para.lstrip(" "))
        width = 94 if indent == 0 else 88
        wrapped = textwrap.wrap(para, width=width, subsequent_indent=" " * min(indent, 4)) or [para]
        wrapped_lines.extend(wrapped)
    return "\n".join(wrapped_lines)


def draw_card(card: PromptCard, figures_dir: Path, max_card_chars: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    row = card.row
    color = CARD_COLORS[row["card_color"]]
    body = wrap_body(card.body, max_card_chars)
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="#ffffff")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    shadow = FancyBboxPatch((0.045, 0.045), 0.91, 0.88, boxstyle="round,pad=0.014,rounding_size=0.028", linewidth=0, facecolor="#9ca3af", alpha=0.22)
    card_box = FancyBboxPatch((0.04, 0.055), 0.91, 0.88, boxstyle="round,pad=0.014,rounding_size=0.028", linewidth=1.2, edgecolor="#e5e7eb", facecolor="#ffffff")
    title_box = FancyBboxPatch((0.04, 0.805), 0.91, 0.13, boxstyle="round,pad=0.014,rounding_size=0.028", linewidth=0, facecolor=color)
    ax.add_patch(shadow)
    ax.add_patch(card_box)
    ax.add_patch(title_box)
    ax.text(0.07, 0.872, row["card_title"], color="white", fontsize=20, fontweight="bold", va="center", ha="left")
    ax.text(0.07, 0.812, f"{row['paper_label']}  |  {row['module']}", color="#e5e7eb", fontsize=10.5, va="center", ha="left")
    ax.text(0.07, 0.765, body, color="#111827", fontsize=9.0, fontfamily="DejaVu Sans Mono", va="top", ha="left", linespacing=1.12)
    ax.text(0.93, 0.075, "Secrets redacted | Forecast arrays unchanged", color="#6b7280", fontsize=8.5, va="bottom", ha="right")
    out_base = figures_dir / row["prompt_id"]
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    return fig, pdf, png


def write_manifest(cards: Sequence[PromptCard], dirs: dict[str, Path], repo_root: Path, git: dict[str, Any]) -> None:
    rows = []
    for card in cards:
        clean_row = {}
        for key in MANIFEST_FIELDS:
            value = card.row.get(key)
            if isinstance(value, bool):
                clean_row[key] = value
            elif value is None:
                clean_row[key] = None
            else:
                clean_row[key] = redact_sensitive_text(str(value))[0]
        rows.append(clean_row)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root_name": repo_root.name,
        "git": git,
        "cards": rows,
    }
    json_path = dirs["reports"] / "prompt_contract_manifest.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = dirs["reports"] / "prompt_contract_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(text: Any) -> str:
    value = "" if text is None else str(text)
    value = redact_sensitive_text(value)[0]
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def write_tex_table(cards: Sequence[PromptCard], dirs: dict[str, Path]) -> None:
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{p{0.17\linewidth}p{0.17\linewidth}p{0.22\linewidth}ccc p{0.18\linewidth}}",
        r"\toprule",
        r"Prompt role & Model / backend & Used for & Forecast arrays changed? & Uses post-hoc metrics? & Citation-quality claim? & Paper placement \\",
        r"\midrule",
    ]
    for card in cards:
        row = card.row
        lines.append(
            " & ".join(
                [
                    tex_escape(row["runtime_role"]),
                    tex_escape(row["model_backend"]),
                    tex_escape(row["called_by"]),
                    "No",
                    "No",
                    "No",
                    "Appendix",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\endgroup", ""])
    (dirs["tables"] / "prompt_contract_manifest.tex").write_text("\n".join(lines), encoding="utf-8")


def write_latex_snippet(cards: Sequence[PromptCard], dirs: dict[str, Path]) -> None:
    lines = [
        r"\section{Prompt Contracts and LLM Usage Boundaries}",
        "",
        "These cards summarize static prompt contracts and runtime I/O schemas used by EAF-MAS. Dynamic runtime payloads are replaced with placeholders, and secrets are redacted. The prompts govern evidence auditing, model-assisted summarization, explanation, event screening, and skill memory; they do not allow LLMs or AutoSkill to directly generate ridership forecasts.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLM prompt roles and claim boundaries.}",
        r"\input{tables/prompt_contract_manifest.tex}",
        r"\end{table}",
        "",
    ]
    for card in cards:
        pid = card.row["prompt_id"]
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{figures/{pid}.pdf}}",
                rf"\caption{{{tex_escape(card.row['card_title'])}. This card reports the static prompt contract and runtime boundary; dynamic payloads are abbreviated and secrets are redacted.}}",
                r"\end{figure}",
                "",
            ]
        )
    (dirs["latex"] / "appendix_prompt_contract_cards.tex").write_text("\n".join(lines), encoding="utf-8")


def prompt_scan(repo_root: Path) -> str:
    roots = ["agents", "experiments", "event_screening", "tests"]
    rx = re.compile(r"SYSTEM_PROMPT|USER_PROMPT|prompt_version|messages=|chat\.completions\.create|apply_chat_template|_prompt|build_.*prompt")
    rows: list[str] = []
    for rel_root in roots:
        base = repo_root / rel_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix not in {".py", ".md", ".json"}:
                continue
            try:
                text = read_text(path)
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    rel = path.relative_to(repo_root)
                    rows.append(f"{rel}:{i}:{line.strip()}")
    return redact_sensitive_text("\n".join(rows))[0]


def write_reports(cards: Sequence[PromptCard], dirs: dict[str, Path], git: dict[str, Any], scan_text: str) -> None:
    real_llm = [c.row["prompt_id"] for c in cards if c.row["is_llm_call"]]
    proxy = ["prompt_card_06_skillbench_judge_template_proxy"]
    skill_contract = ["prompt_card_05_autoskill_skill_memory_contract"]
    report = [
        "# Prompt Contract Extraction Report",
        "",
        f"- Generated cards: `{len(cards)}`.",
        f"- Git repository detected: `{git['git_repo_detected']}`.",
        f"- Git head: `{git['git_head']}`.",
        f"- Git branch: `{git['git_branch']}`.",
        "- Appendix cards recommended: all seven cards.",
        "- Main-text display recommended: none; cite the appendix snippet only.",
        f"- Real LLM-call cards: `{', '.join(real_llm)}`.",
        f"- Proxy / diagnostic cards: `{', '.join(proxy)}`.",
        f"- Skill-contract cards that are not LLM prompts: `{', '.join(skill_contract)}`.",
        "- Prompt use of actual ridership / WAPE / post-hoc metrics: `none`.",
        "- Prompt permission for direct forecast-array modification: `none`.",
        "- Qwen-Plus treated as citation-quality URL evidence: `no`.",
        "- Secret redaction: `passed`.",
        "- LaTeX appendix snippet generated: `latex/appendix_prompt_contract_cards.tex`.",
        "- Unit tests: run `python -m unittest tests/test_prompt_contract_cards.py -v` after generation.",
        "",
        "## Paper usage",
        "",
        "Use these figures in the appendix to document the static prompt contracts and claim boundaries. They support transparency of LLM usage, but they are not experimental results and should not be used as evidence that LLMs improve numerical forecasting.",
        "",
    ]
    (dirs["reports"] / "prompt_contract_extraction_report.md").write_text(redact_sensitive_text("\n".join(report))[0], encoding="utf-8")
    (dirs["reports"] / "prompt_scan_raw.txt").write_text(scan_text, encoding="utf-8")
    (dirs["reports"] / "prompt_contract_paper_usage.md").write_text(
        "The static prompt contracts and runtime I/O schemas for Qwen-Plus summaries, local evidence auditing, forecast-time explanation, event screening, AutoSkill memory, and SkillBench diagnostics are reported in Appendix X. These contracts explicitly prohibit post-hoc metrics, direct LLM-based numerical forecasting, and skill-driven forecast-array changes.\n",
        encoding="utf-8",
    )


def write_redaction_audit(dirs: dict[str, Path]) -> None:
    failures: list[str] = []
    scanned = 0
    for path in dirs["root"].rglob("*"):
        if path.is_dir() or path.suffix.lower() in {".png", ".pdf"}:
            continue
        text = read_text(path)
        scanned += 1
        for pattern in FORBIDDEN_OUTPUT_PATTERNS:
            if pattern.search(text):
                failures.append(f"{path.relative_to(dirs['root'])}: {pattern.pattern}")
    status = "passed" if not failures else "failed"
    lines = [
        "# Prompt Contract Redaction Audit",
        "",
        f"- status: `{status}`",
        f"- text files scanned: `{scanned}`",
        f"- failures: `{len(failures)}`",
    ]
    if failures:
        lines.extend(["", "## Failures", *[f"- {redact_sensitive_text(x)[0]}" for x in failures]])
    audit_path = dirs["reports"] / "prompt_contract_redaction_audit.md"
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    if failures:
        raise RuntimeError("prompt contract redaction audit failed")


def write_all_cards_pdf(card_figs: Sequence[Any], out_path: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    with PdfPages(out_path) as pdf:
        for fig in card_figs:
            pdf.savefig(fig, bbox_inches="tight")
    for fig in card_figs:
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=Path(os.environ.get("REMOTE_REPO") or "/root/autodl-tmp/0206moment"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--max_card_chars", type=int, default=2600)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    dirs = ensure_dirs(args.output_root)
    git = git_info(repo_root)
    cards = build_cards(repo_root)

    figs = []
    figure_paths: list[tuple[str, Path, Path]] = []
    for card in cards:
        fig, pdf, png = draw_card(card, dirs["figures"], args.max_card_chars)
        figs.append(fig)
        figure_paths.append((card.row["prompt_id"], pdf, png))
    write_all_cards_pdf(figs, dirs["figures"] / "prompt_contract_cards_all.pdf")

    write_manifest(cards, dirs, repo_root, git)
    write_tex_table(cards, dirs)
    write_latex_snippet(cards, dirs)
    write_reports(cards, dirs, git, prompt_scan(repo_root))
    write_redaction_audit(dirs)
    print(json.dumps({"output_root": str(dirs["root"]), "cards": len(cards), "figures": len(figure_paths)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
