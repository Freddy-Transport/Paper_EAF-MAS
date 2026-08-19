#!/usr/bin/env python3
"""Merge visualization figure manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="experiments/visualization/config/viz_config.yaml")
    args = parser.parse_args()
    root = add_repo_root(args.repo_root)

    from experiments.visualization.eaf_viz.config import load_config, resolve_output_root

    cfg = load_config(root / args.config, repo_root=root)
    output_root = resolve_output_root(cfg, root)
    frames = []
    for name in ["figure_manifest_main.csv", "figure_manifest_appendix.csv", "figure_manifest_case_dashboard.csv"]:
        path = output_root / name
        if path.is_file():
            frames.append(pd.read_csv(path))
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged.to_csv(output_root / "figure_manifest.csv", index=False)
    print(f"saved {output_root / 'figure_manifest.csv'}")


if __name__ == "__main__":
    main()
