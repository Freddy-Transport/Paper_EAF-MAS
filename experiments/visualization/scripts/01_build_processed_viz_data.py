#!/usr/bin/env python3
"""Build standardized visualization tables from experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="experiments/visualization/config/viz_config.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = add_repo_root(args.repo_root)

    from experiments.visualization.eaf_viz.config import load_config, resolve_formal_metric_paths, resolve_input_roots, resolve_output_root
    from experiments.visualization.eaf_viz.case_selection import write_selected_visualization_cases
    from experiments.visualization.eaf_viz.plotting_utils import ensure_output_dirs
    from experiments.visualization.eaf_viz.schema import (
        build_processed_tables,
        load_event_metadata,
        load_station_metadata,
        write_processed_tables,
        write_schema_report,
    )

    cfg = load_config(root / args.config, repo_root=root)
    out_dirs = ensure_output_dirs(resolve_output_root(cfg, root))
    tables, report = build_processed_tables(
        resolve_input_roots(cfg, root),
        method_aliases=cfg.get("method_aliases"),
        neutral_threshold=float(cfg.get("neutral_wape_gain_threshold", 0.01)),
        formal_metric_paths=resolve_formal_metric_paths(cfg, root),
    )
    tables["station_metadata"] = load_station_metadata(root / cfg.get("channel_map", "data/nyc_top128_channel_map.json"))
    tables["event_metadata"] = load_event_metadata(root / cfg.get("events_json", "data/nyc_top128_station_events.json"))
    report["station_metadata"] = f"{len(tables['station_metadata'])} rows"
    report["event_metadata"] = f"{len(tables['event_metadata'])} rows"
    modes_present = sorted(tables["case_metrics"]["mode"].dropna().unique().tolist()) if not tables["case_metrics"].empty and "mode" in tables["case_metrics"] else []
    report["modes_present"] = modes_present
    expected_modes = ["rag_explain", "event_adapter_frozen_moment"]
    report["missing_expected_modes"] = [mode for mode in expected_modes if mode not in modes_present]
    missing_x_report = out_dirs["reports"] / "missing_eaf_mas_x_cases.md"
    if "rag_explain" not in modes_present:
        missing_x_report.write_text(
            "# Missing EAF-MAS-X matched cases\n\n"
            "The current visualization input roots contain no `rag_explain` prediction JSON files. "
            "EAF-MAS-X is therefore represented only as a paper-facing alias and should not be plotted "
            "as an empirical method until matched `rag_explain` cases are generated for the same forecast windows.\n\n"
            "Recommended rerun pattern:\n\n"
            "```bash\n"
            "python experiments/run_paper_event_forecasting.py --mode rag_explain --target_date <anchor> "
            "--station_scope event_venue28 --output_root <matched_output_root> --enable_online_retrieval "
            "--llm_base_url http://127.0.0.1:8000/v1 --llm_model Qwen/Qwen3-8B\n"
            "```\n",
            encoding="utf-8",
        )
    elif missing_x_report.exists():
        missing_x_report.unlink()
    if args.strict and tables["case_metrics"].empty:
        raise SystemExit("strict mode: no case_metrics rows built")
    write_processed_tables(tables, out_dirs["processed"])
    write_selected_visualization_cases(
        tables["case_metrics"],
        tables["horizon_forecast"],
        tables["channel_metrics"],
        out_dirs["reports"] / "selected_visualization_cases.json",
    )
    write_schema_report(report, out_dirs["root"] / "data_schema_report.md")
    print(f"saved {out_dirs['processed']}")
    print(f"saved {out_dirs['reports'] / 'selected_visualization_cases.json'}")
    print(f"saved {out_dirs['root'] / 'data_schema_report.md'}")


if __name__ == "__main__":
    main()
