#!/usr/bin/env python3
"""Export paper-ready explanation tables and appendix snippets."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List


def tex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _case_type(name: str, raw_wape: float, adjusted_wape: float, mode: str) -> str:
    if "abstention" in name or mode == "rag_explain" or abs(raw_wape - adjusted_wape) < 1e-9:
        return "correct_abstention"
    if adjusted_wape < raw_wape:
        return "successful_correction"
    return "failure_or_weak_evidence"


def _extract_section(markdown: str, start_heading: str, next_heading_level: str = "## ") -> str:
    start = markdown.find(start_heading)
    if start < 0:
        return ""
    end = markdown.find(f"\n{next_heading_level}", start + len(start_heading))
    if end < 0:
        end = len(markdown)
    return markdown[start:end].strip()


def forecast_time_text_has_posthoc_metrics(markdown: str) -> bool:
    section = _extract_section(markdown, "## Forecast-Time Model Reasoning")
    if not section:
        section = _extract_section(markdown, "## Model Reasoning")
    if not section:
        return False
    return bool(re.search(r"\b(WAPE|MAE|RMSE|sMAPE|post-hoc|ground truth)\b", section, flags=re.IGNORECASE))


def _load_cases(run_root: Path) -> List[Dict]:
    cases = []
    for summary_path in sorted((run_root / "cases_vllm").glob("*/reports/run_summary.json")):
        case_dir = summary_path.parents[1]
        case_name = case_dir.name
        summary = _read_json(summary_path)
        md_paths = list((case_dir / "explanations").glob("*.md"))
        markdown = md_paths[0].read_text(encoding="utf-8") if md_paths else ""
        raw_wape = float((summary.get("raw_metrics") or {}).get("wape", 0.0) or 0.0)
        adjusted_wape = float((summary.get("adjusted_metrics") or {}).get("wape", raw_wape) or raw_wape)
        mode = str(summary.get("mode", ""))
        cases.append(
            {
                "case": case_name,
                "case_type": _case_type(case_name, raw_wape, adjusted_wape, mode),
                "mode": mode,
                "raw_wape": raw_wape,
                "adjusted_wape": adjusted_wape,
                "delta_wape": raw_wape - adjusted_wape,
                "accepted_evidence_count": int((summary.get("explanation_quality") or {}).get("accepted_evidence_count", 0) or 0),
                "citation_coverage": float((summary.get("explanation_quality") or {}).get("citation_coverage", 0.0) or 0.0),
                "llm_parsed": bool(summary.get("llm_explanation_parsed")),
                "self_check_status": (summary.get("self_check") or {}).get("status", ""),
                "markdown_path": str(md_paths[0]) if md_paths else "",
                "markdown": markdown,
                "forecast_time_posthoc_leak": forecast_time_text_has_posthoc_metrics(markdown),
                "source_assisted_context": int(
                    "### Level B: Source-Assisted Evidence" in markdown
                    and "No source-assisted non-citation evidence recorded" not in markdown
                ),
                "model_assisted_context": int(
                    "### Level C: Model-Assisted Summary" in markdown
                    and "No model-assisted evidence summary available" not in markdown
                ),
            }
        )
    return cases


def _write_case_table(path: Path, cases: List[Dict]) -> None:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Case & Mode & Raw WAPE & Adjusted WAPE & $\\Delta$ WAPE & Citations \\\\",
        "\\midrule",
    ]
    for row in cases:
        lines.append(
            f"{tex_escape(row['case_type'])} & {tex_escape(row['mode'])} & "
            f"{row['raw_wape']:.2f} & {row['adjusted_wape']:.2f} & {row['delta_wape']:.3f} & "
            f"{row['accepted_evidence_count']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_evidence_table(path: Path, cases: List[Dict]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Case & LLM Parsed & Citation Coverage & Accepted URLs & Source-assisted & Model summary & Self-check \\\\",
        "\\midrule",
    ]
    for row in cases:
        lines.append(
            f"{tex_escape(row['case'])} & {int(row['llm_parsed'])} & "
            f"{row['citation_coverage']:.2f} & {row['accepted_evidence_count']} & "
            f"{int(row['source_assisted_context'])} & {int(row['model_assisted_context'])} & "
            f"{tex_escape(row['self_check_status'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_appendix(path: Path, cases: List[Dict]) -> None:
    blocks = []
    for row in cases:
        markdown = row["markdown"]
        evidence = _extract_section(markdown, "## Event Evidence")
        reasoning = _extract_section(markdown, "## Forecast-Time Model Reasoning")
        metrics = _extract_section(markdown, "## Post-hoc Metrics")
        if row["accepted_evidence_count"] == 0 and "No accepted citation-quality external evidence" not in evidence:
            evidence += "\n\nNo accepted citation-quality external evidence was available."
        blocks.append(
            "% case_id: " + str(row["case"]) + "\n"
            "\\subsection{" + tex_escape(row["case"]) + "}\n"
            "\\paragraph{Case type.} " + tex_escape(row["case_type"]) + "\n\n"
            "\\paragraph{Evidence.}\n" + tex_escape(evidence[:1800]) + "\n\n"
            "\\paragraph{LLM reasoning.}\n" + tex_escape(reasoning[:1800]) + "\n\n"
            "\\paragraph{Post-hoc metrics.}\n" + tex_escape(metrics[:700]) + "\n"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _write_markdown_case_study(path: Path, cases: List[Dict]) -> None:
    lines = ["# Paper Explanation Case Study", ""]
    for row in cases:
        lines += [
            f"## {row['case']} ({row['case_type']})",
            "",
            f"- mode: `{row['mode']}`",
            f"- raw WAPE: `{row['raw_wape']:.3f}`",
            f"- adjusted WAPE: `{row['adjusted_wape']:.3f}`",
            f"- accepted citation URLs: `{row['accepted_evidence_count']}`",
            f"- forecast-time posthoc leakage detected: `{row['forecast_time_posthoc_leak']}`",
            "",
            _extract_section(row["markdown"], "## Forecast-Time Model Reasoning")[:2400],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_paper_assets(run_root: str | Path) -> dict:
    run_root = Path(run_root)
    out = run_root / "paper_assets"
    fig_out = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(run_root)
    _write_case_table(out / "explanation_case_table.tex", cases)
    _write_evidence_table(out / "evidence_quality_table.tex", cases)
    _write_appendix(out / "appendix_explanations.tex", cases)
    _write_markdown_case_study(out / "explanation_case_study.md", cases)

    copied_figures = []
    for case_dir in sorted((run_root / "cases_vllm").glob("*")):
        for fig in (case_dir / "figures").glob("*.png"):
            dest = fig_out / f"{case_dir.name}_{fig.name}"
            shutil.copy2(fig, dest)
            copied_figures.append(str(dest))

    summary = {
        "run_root": str(run_root),
        "case_count": len(cases),
        "cases": [{k: v for k, v in row.items() if k != "markdown"} for row in cases],
        "copied_figures": copied_figures,
        "outputs": {
            "explanation_case_table": str(out / "explanation_case_table.tex"),
            "evidence_quality_table": str(out / "evidence_quality_table.tex"),
            "appendix_explanations": str(out / "appendix_explanations.tex"),
            "explanation_case_study": str(out / "explanation_case_study.md"),
        },
    }
    (out / "paper_assets_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="autotemp/ccfa_formal_rolling_20260607_115014")
    args = parser.parse_args()
    print(json.dumps(export_paper_assets(args.run_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
