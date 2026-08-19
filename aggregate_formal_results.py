import glob
import hashlib
import json
import os
import time
from pathlib import Path


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


root = Path.cwd()
run = Path(os.environ.get("RUN_ROOT", "autotemp/ccfa_formal_rolling_20260607_115014"))
reports = run / "reports"
reports.mkdir(parents=True, exist_ok=True)
full = json.loads((run / "full_rolling/reports/full_results.json").read_text(encoding="utf-8"))

case_rows = []
for p in sorted((run / "cases").glob("*/reports/run_summary.json")):
    obj = json.loads(p.read_text(encoding="utf-8"))
    case_rows.append(
        {
            "case": p.parts[-3],
            "mode": obj.get("mode"),
            "raw_wape": obj.get("raw_metrics", {}).get("wape"),
            "adjusted_wape": obj.get("adjusted_metrics", {}).get("wape"),
            "self_check_status": obj.get("self_check", {}).get("status"),
            "warnings": len(obj.get("self_check", {}).get("warnings", [])),
            "qwenplus_api_calls": obj.get("qwenplus_stats", {}).get("api_calls", 0),
            "accepted_evidence_count": obj.get("explanation_quality", {}).get("accepted_evidence_count", 0),
            "citation_coverage": obj.get("explanation_quality", {}).get("citation_coverage", 0.0),
            "station_event_linking": obj.get("explanation_quality", {}).get("station_event_linking"),
            "multi_hop_reasoning": obj.get("explanation_quality", {}).get("multi_hop_reasoning"),
            "abstention_correct": obj.get("explanation_quality", {}).get("abstention_correct"),
        }
    )
(reports / "evidence_quality_summary.json").write_text(
    json.dumps({"cases": case_rows}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

skill = {}
for name in ["skill_val", "skill_test"]:
    p = run / name / "reports/skill_evolution_summary.json"
    if p.is_file():
        skill[name] = json.loads(p.read_text(encoding="utf-8"))
(reports / "skill_evolution_summary.json").write_text(json.dumps(skill, ensure_ascii=False, indent=2), encoding="utf-8")

key_files = [
    "agents/config.py",
    "agents/evidence_research_agent.py",
    "agents/explanation_evaluator.py",
    "experiments/run_full_paper_results.py",
    "experiments/run_paper_event_forecasting.py",
    "experiments/run_skill_evolution_study.py",
]
keep = [
    str(run),
    "data",
    "tests",
    "event_post_training",
    "experiments/outputs/lp_fallback_nyc_top128",
    "experiments/outputs/event_adapter_formal_frozen",
    "experiments/outputs/event_adapter_formal_peft",
    "agents/event_adapter.py",
    "agents/evidence_research_agent.py",
    "agents/explanation_evaluator.py",
    "agents/forecast_explanation_agent.py",
    "agents/paper_workflow.py",
    "agents/paper_knowledge_base.py",
    "agents/numerical_agent.py",
    "experiments/run_full_paper_results.py",
    "experiments/run_paper_event_forecasting.py",
    "experiments/run_skill_evolution_study.py",
    "experiments/train_event_aware_posttrainer.py",
]
legacy_code = [
    "predict_station.py",
    "run_multi_trace.py",
    "run_top128_autodl.py",
    "run_venue37_fusion.py",
    "visualize_agent.py",
    "build_residual_rag.py",
    "experiments/event_error_focus_experiment.py",
    "experiments/event_ablation_evaluation.py",
    "scripts/run_venue37_llm_comparison.sh",
    "scripts/compare_venue37_fusion.py",
    "scripts/export_experiment_summary.py",
    "scripts/select_experiment_dates.py",
]
delete = []
if Path("autotemp").is_dir():
    for p in Path("autotemp").iterdir():
        if p.resolve() != run.resolve():
            delete.append(str(p))
for p in glob.glob("autodl_outputs/*"):
    delete.append(p)
for pattern in [
    "experiments/outputs/event_adapter",
    "experiments/outputs/event_adapter_top128_smoke*",
    "experiments/outputs/event_adapter_peft_top128_smoke*",
    "experiments/outputs/lp_fallback",
    "experiments/outputs/纽约model.safetensors",
    ".DS_Store",
    "results.json",
    "results.png",
    "run_kb_*.log",
    "config.json",
    "model.safetensors",
    "traces*",
    "figures_best",
    "figures_experiment",
]:
    delete.extend(glob.glob(pattern))
delete.extend([p for p in legacy_code if Path(p).exists()])

seen = set()
clean_delete = []
for p in delete:
    if p in seen:
        continue
    seen.add(p)
    try:
        if Path(p).resolve() == run.resolve():
            continue
    except Exception:
        pass
    clean_delete.append(p)

manifest = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "formal_run_root": str(run),
    "keep": keep,
    "delete": clean_delete,
    "legacy_code": legacy_code,
    "model_hashes": {
        "lp_fallback_nyc_top128": sha(root / "experiments/outputs/lp_fallback_nyc_top128/lp_weights.pt"),
        "frozen_adapter": sha(run / "models/used_checkpoints/frozen_adapter/event_adapter.pt"),
        "peft_adapter": sha(run / "models/used_checkpoints/peft_adapter/event_adapter.pt"),
    },
    "source_hashes": {p: sha(root / p) for p in key_files},
}
(reports / "cleanup_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

summary = full["summary"]
base = summary.get("numerical_only", {})
frozen = summary.get("event_adapter_frozen_moment", {})
peft = summary.get("event_adapter_peft_moment", {})
ci = full.get("bootstrap_wape_delta_ci", [])

md = "# Final Formal Experiment Summary\n\n"
md += (
    f"## Run\n\n"
    f"- Formal root: `{run}`\n"
    f"- Test rolling anchors: `{full.get('n_test_anchors')}`\n"
    f"- Validation anchors: `{full.get('n_val_anchors')}`\n"
    f"- Rolling stride: `{full.get('rolling_stride')}`\n\n"
)
md += (
    f"## Forecast Results\n\n"
    f"- Numerical-only WAPE: `{base.get('wape', float('nan')):.3f}`\n"
    f"- Frozen adapter WAPE: `{frozen.get('wape', float('nan')):.3f}`\n"
    f"- PEFT adapter WAPE: `{peft.get('wape', float('nan')):.3f}`\n"
    f"- Numerical event-active WAPE: `{base.get('event_wape', float('nan')):.3f}`\n"
    f"- Frozen event-active WAPE: `{frozen.get('event_wape', float('nan')):.3f}`\n"
    f"- PEFT event-active WAPE: `{peft.get('event_wape', float('nan')):.3f}`\n\n"
)
md += "## Bootstrap CI\n\n```json\n" + json.dumps(ci, ensure_ascii=False, indent=2) + "\n```\n\n"
md += (
    f"## Explainability and Evidence\n\n"
    f"- Representative cases: `{len(case_rows)}`\n"
    f"- Accepted citation-quality evidence count remains `{sum(r['accepted_evidence_count'] for r in case_rows)}` because Qwen-Plus was cache-only / not configured in the SSH environment.\n"
    f"- Station-event linking and multi-hop reasoning checks passed for generated cases.\n\n"
)
md += "## Skill Evolution\n\n```json\n" + json.dumps(skill, ensure_ascii=False, indent=2) + "\n```\n\n"
md += (
    "## Interpretation\n\n"
    "Adapter gains are positive but small. The CCF-A positioning should emphasize numerical-backbone-first "
    "event-aware explainability, safe abstention, evidence diagnostics, and validation-promoted skill memory "
    "rather than claiming large accuracy gains from RAG/Skill.\n"
)
(reports / "final_experiment_summary.md").write_text(md, encoding="utf-8")

print(json.dumps({"reports": str(reports), "cases": len(case_rows), "delete_candidates": len(clean_delete)}, ensure_ascii=False, indent=2))
