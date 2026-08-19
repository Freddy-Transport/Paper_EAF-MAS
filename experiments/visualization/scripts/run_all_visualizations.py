#!/usr/bin/env python3
"""Run the full EAF-MAS v2 visualization pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


STALE_REPORTS = (
    "missing_ablation_plan.md",
    "top_venue_visual_experiment_plan.md",
)


def clean_stale_reports(repo_root: str | Path) -> None:
    reports_dir = Path(repo_root) / "experiments" / "visualization" / "outputs" / "reports"
    for name in STALE_REPORTS:
        path = reports_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="experiments/visualization/config/viz_config.yaml")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--skip-appendix", action="store_true")
    parser.add_argument("--skip-case", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    clean_stale_reports(args.repo_root)

    commands = [
        ["00_scan_repo_outputs.py", "--repo-root", args.repo_root],
        ["01_build_processed_viz_data.py", "--repo-root", args.repo_root, "--config", args.config],
        ["02_make_main_figures.py", "--repo-root", args.repo_root, "--config", args.config],
    ]
    if args.strict:
        commands[1].append("--strict")
    if not args.skip_appendix:
        commands.append(["03_make_appendix_figures.py", "--repo-root", args.repo_root, "--config", args.config])
    if not args.skip_case:
        cmd = ["04_make_case_dashboard.py", "--repo-root", args.repo_root, "--config", args.config]
        if args.case_id:
            cmd.extend(["--case-id", args.case_id])
        commands.append(cmd)
    commands.append(["05_make_figure_manifest.py", "--repo-root", args.repo_root, "--config", args.config])

    for cmd in commands:
        full = [sys.executable, str(SCRIPT_DIR / cmd[0]), *cmd[1:]]
        print("RUN", " ".join(full))
        subprocess.run(full, check=True)


if __name__ == "__main__":
    main()
