#!/usr/bin/env python3
"""Generate appendix figures."""

from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import add_repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="experiments/visualization/config/viz_config.yaml")
    args = parser.parse_args()
    root = add_repo_root(args.repo_root)

    from experiments.visualization.eaf_viz.config import load_config, resolve_output_root
    from experiments.visualization.eaf_viz.plots_appendix import make_appendix_figures
    from experiments.visualization.eaf_viz.plotting_utils import ensure_output_dirs

    cfg = load_config(root / args.config, repo_root=root)
    out_dirs = ensure_output_dirs(resolve_output_root(cfg, root))
    tables = {"case_metrics": _read(out_dirs["processed"] / "case_metrics.csv")}
    manifest = make_appendix_figures(tables, out_dirs, cfg)
    manifest.write(out_dirs["root"] / "figure_manifest_appendix.csv")
    print(f"saved {out_dirs['root'] / 'figure_manifest_appendix.csv'}")


def _read(path):
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
