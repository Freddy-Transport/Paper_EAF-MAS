#!/usr/bin/env python3
"""Scan repository outputs for visualization candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default="experiments/visualization/outputs")
    args = parser.parse_args()
    root = add_repo_root(args.repo_root)

    from experiments.visualization.eaf_viz.io import scan_repo_outputs, write_scan_report

    out = root / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    df = scan_repo_outputs(root)
    df.to_csv(out / "candidate_files.csv", index=False)
    write_scan_report(df, out / "repo_scan_report.md")
    print(f"saved {out / 'candidate_files.csv'}")
    print(f"saved {out / 'repo_scan_report.md'}")


if __name__ == "__main__":
    main()
