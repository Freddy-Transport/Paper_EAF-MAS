#!/usr/bin/env python3
"""Generate main-paper figures."""

from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import add_repo_root


TABLE_NAMES = [
    "case_metrics",
    "channel_metrics",
    "horizon_forecast",
    "event_audit",
    "residual_memory",
    "skill_memory",
    "formal_metrics",
    "station_metadata",
    "event_metadata",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="experiments/visualization/config/viz_config.yaml")
    args = parser.parse_args()
    root = add_repo_root(args.repo_root)

    from experiments.visualization.eaf_viz.config import load_config, resolve_output_root
    from experiments.visualization.eaf_viz.plots_main import make_main_figures
    from experiments.visualization.eaf_viz.plotting_utils import ensure_output_dirs

    cfg = load_config(root / args.config, repo_root=root)
    out_dirs = ensure_output_dirs(resolve_output_root(cfg, root))
    tables = _read_tables(out_dirs["processed"])
    manifest = make_main_figures(tables, out_dirs, cfg)
    manifest.write(out_dirs["root"] / "figure_manifest_main.csv")
    print(f"saved {out_dirs['root'] / 'figure_manifest_main.csv'}")


def _read_tables(processed):
    tables = {}
    for name in TABLE_NAMES:
        path = processed / f"{name}.csv"
        tables[name] = _read_csv_safe(path)
    return tables


def _read_csv_safe(path):
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
