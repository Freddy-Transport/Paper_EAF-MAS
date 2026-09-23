#!/usr/bin/env python3
"""Archive superseded execution paths and their dependent callers; default dry run.

Only source files and the previously published architecture image are moved.
Data, checkpoints, prediction caches, and experiment outputs stay in place.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

EXCLUDE_PARTS = {"legacy", "private_runs", "outputs", "data", "autotemp", "__pycache__", ".git", "external", ".venv"}
FORMAL = {"experiments/run_formal_evaluation.py", "experiments/refit_residual_adapter.py",
          "experiments/evolve_validated_skills.py", "scripts/archive_superseded.py"}
SEEDS = {
    "agents/orchestrator.py", "agents/run.py", "agents/skill_dispatcher.py",
    "agents/skill_extractor.py", "agents/prediction_fusion.py", "agents/paper_workflow.py",
    "agents/event_agent.py", "agents/build_knowledge_base.py",
    "scripts/run_on_autodl.sh", "scripts/run_venue37_llm_comparison.sh", "scripts/export_experiment_summary.py",
    "run_paper_event_forecasting.py", "aggregate_formal_results.py", "update_vllm_reports.py",
    "execute_cleanup.py", "tests/test_autoskill_evolution.py", "tests/test_explanation_quality.py",
    "tests/test_evidence_auditor_controller.py", "assets/eafmas_architecture_figure1.png",
    "tests/test_prompt_contract_cards.py",
    "experiments/analyze_event_stratified_results.py", "experiments/assemble_paper_figures.py",
    "experiments/export_ccfa_fullstack_figures.py", "experiments/export_paper_explanation_assets.py",
    "experiments/profile_eafmas_runtime_cost.py", "experiments/refresh_saved_explanations.py",
    "experiments/visualize_prompt_contract_cards.py", "experiments/visualize_station_channel_gate_heatmap.py",
    "experiments/train_event_aware_posttrainer.py",
}


def inventory(root):
    files = {str(p.relative_to(root)): p for p in root.rglob("*.py")
             if not set(p.relative_to(root).parts) & EXCLUDE_PARTS}
    selected = {p: "superseded workflow or corresponding legacy test" for p in files if p in SEEDS
                or (p.startswith("experiments/run_") and p not in FORMAL)
                or p.startswith("experiments/visualization/")}
    for name in SEEDS:
        if (root/name).is_file(): selected.setdefault(name,"superseded artifact")
    while True:
        modules = {str(Path(x).with_suffix("")).replace("/", ".") for x in selected if x.endswith(".py")}
        found = {}
        for name, path in files.items():
            if name in selected or name in FORMAL: continue
            try: tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeError): continue
            refs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): refs.extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    refs.append(node.module or "")
                    refs.extend((node.module + "." if node.module else "") + n.name for n in node.names)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value in selected: refs.append(node.value[:-3].replace("/", "."))
            hits = sorted(set(refs) & modules)
            if hits: found[name] = "depends on archived path: " + ", ".join(hits)
        if not found: break
        selected.update(found)
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args=p.parse_args()
    root=args.root.resolve()
    selected=inventory(root)
    manifest=[]
    archive=root/"legacy/20260923_pre_audit_fix/paths"
    for name, reason in sorted(selected.items()):
        source=root/name; target=archive/name
        if args.apply and target.exists(): raise FileExistsError(target)
        manifest.append(dict(path=name, reason=reason, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    if args.apply:
        for row in manifest:
            source=root/row["path"]; target=archive/row["path"]
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(target))
        archive.parent.mkdir(parents=True,exist_ok=True)
        manifest_path=archive.parent/"archive_manifest.json"
        previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        manifest_path.write_text(json.dumps(previous+manifest,indent=2))
    print(json.dumps({"applied":args.apply,"count":len(manifest),"files":manifest},indent=2))


if __name__=="__main__": main()
