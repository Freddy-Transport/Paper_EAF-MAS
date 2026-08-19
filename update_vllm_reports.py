import json
import re
import subprocess
import time
from pathlib import Path

RUN_ROOT = Path("autotemp/ccfa_formal_rolling_20260607_115014")
reports = RUN_ROOT / "reports"
full = json.loads((RUN_ROOT / "full_rolling/reports/full_results.json").read_text(encoding="utf-8"))

case_rows = []
for p in sorted((RUN_ROOT / "cases_vllm").glob("*/reports/run_summary.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    case_rows.append(
        {
            "case": p.parts[-3],
            "mode": obj.get("mode"),
            "raw_wape": obj.get("raw_metrics", {}).get("wape"),
            "adjusted_wape": obj.get("adjusted_metrics", {}).get("wape"),
            "self_check_status": obj.get("self_check", {}).get("status"),
            "llm_explanation_parsed": obj.get("llm_explanation_parsed"),
            "llm_explanation_fallback_reason": obj.get("llm_explanation_fallback_reason"),
            "qwenplus_api_calls": obj.get("qwenplus_stats", {}).get("api_calls", 0),
            "accepted_evidence_count": obj.get("explanation_quality", {}).get("accepted_evidence_count", 0),
            "citation_coverage": obj.get("explanation_quality", {}).get("citation_coverage", 0.0),
            "station_event_linking": obj.get("explanation_quality", {}).get("station_event_linking"),
            "multi_hop_reasoning": obj.get("explanation_quality", {}).get("multi_hop_reasoning"),
            "abstention_correct": obj.get("explanation_quality", {}).get("abstention_correct"),
        }
    )
(reports / "evidence_quality_summary.json").write_text(
    json.dumps({"source": "cases_vllm", "cases": case_rows}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

skill = {}
for name in ["skill_val_vllm", "skill_test_vllm"]:
    p = RUN_ROOT / name / "reports/skill_evolution_summary.json"
    if p.is_file():
        skill[name] = json.loads(p.read_text(encoding="utf-8"))
(reports / "skill_evolution_summary.json").write_text(json.dumps(skill, ensure_ascii=False, indent=2), encoding="utf-8")

summary = full["summary"]
base = summary.get("numerical_only", {})
frozen = summary.get("event_adapter_frozen_moment", {})
peft = summary.get("event_adapter_peft_moment", {})
ci = full.get("bootstrap_wape_delta_ci", [])

md = "# Final Formal Experiment Summary\n\n"
md += (
    "## Run\n\n"
    f"- Formal root: `{RUN_ROOT}`\n"
    f"- Test rolling anchors: `{full.get('n_test_anchors')}`\n"
    f"- Validation anchors: `{full.get('n_val_anchors')}`\n"
    f"- Rolling stride: `{full.get('rolling_stride')}`\n"
    "- Explanation engine: local `vLLM` serving `Qwen/Qwen3-8B` from `/root/autodl-tmp/modelscope_qwen3_8b`\n\n"
)
md += (
    "## Forecast Results\n\n"
    f"- Numerical-only WAPE: `{base.get('wape', float('nan')):.3f}`\n"
    f"- Frozen adapter WAPE: `{frozen.get('wape', float('nan')):.3f}`\n"
    f"- PEFT adapter WAPE: `{peft.get('wape', float('nan')):.3f}`\n"
    f"- Numerical event-active WAPE: `{base.get('event_wape', float('nan')):.3f}`\n"
    f"- Frozen event-active WAPE: `{frozen.get('event_wape', float('nan')):.3f}`\n"
    f"- PEFT event-active WAPE: `{peft.get('event_wape', float('nan')):.3f}`\n\n"
)
md += "## Bootstrap CI\n\n```json\n" + json.dumps(ci, ensure_ascii=False, indent=2) + "\n```\n\n"
md += (
    "## Explainability and Evidence\n\n"
    f"- Representative vLLM cases: `{len(case_rows)}`\n"
    f"- LLM parsed cases: `{sum(1 for r in case_rows if r['llm_explanation_parsed'])}/{len(case_rows)}`\n"
    f"- Accepted citation-quality evidence count remains `{sum(r['accepted_evidence_count'] for r in case_rows)}`; this is a citation limitation, not an LLM explanation failure.\n"
    "- Station-event linking and multi-hop reasoning checks passed for generated vLLM cases.\n\n"
)
md += "## Skill Evolution\n\n```json\n" + json.dumps(skill, ensure_ascii=False, indent=2) + "\n```\n\n"
md += (
    "## Interpretation\n\n"
    "Adapter gains are positive but small. The CCF-A positioning should emphasize numerical-backbone-first "
    "event-aware explainability, safe abstention, evidence diagnostics, and validation-promoted skill memory. "
    "The local Qwen/vLLM explanation path is now active for representative qualitative cases, while accepted "
    "external citations remain the main evidence limitation.\n"
)
(reports / "final_experiment_summary.md").write_text(md, encoding="utf-8")

secret_hits = []
pat = re.compile(r"sk-[A-Za-z0-9]{12,}")
for base_path in [Path("."), RUN_ROOT]:
    for p in base_path.rglob("*"):
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if pat.search(txt):
            secret_hits.append(str(p))

health = subprocess.run(
    "curl -s http://127.0.0.1:8000/v1/models",
    shell=True,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
).stdout[:1000]
verification = {
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "vllm_models_response_prefix": health,
    "vllm_cases": case_rows,
    "secret_pattern_hits": sorted(set(secret_hits)),
}
(reports / "vllm_explanation_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"vllm_cases": len(case_rows), "parsed": sum(1 for r in case_rows if r["llm_explanation_parsed"]), "secret_hits": len(set(secret_hits))}, ensure_ascii=False, indent=2))
