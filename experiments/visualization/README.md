# EAF-MAS v2 Visualization Pipeline

This directory generates publication-ready figures from existing EAF-MAS v2 experiment outputs. It does not fabricate missing baselines or ablations.

## Method aliases

- `numerical_only` -> `PT-MOMENT`
- `rag_explain` -> `EAF-MAS-X`
- `event_adapter_frozen_moment` -> `EAF-MAS-C`
- `event_adapter_peft_moment` and `full_skill_agent` are disabled for current paper figures unless explicitly enabled in config.

## Run

```bash
python experiments/visualization/scripts/run_all_visualizations.py \
  --repo-root . \
  --config experiments/visualization/config/viz_config.yaml
```

Individual steps:

```bash
python experiments/visualization/scripts/00_scan_repo_outputs.py --repo-root .
python experiments/visualization/scripts/01_build_processed_viz_data.py --repo-root . --config experiments/visualization/config/viz_config.yaml
python experiments/visualization/scripts/02_make_main_figures.py --repo-root . --config experiments/visualization/config/viz_config.yaml
python experiments/visualization/scripts/03_make_appendix_figures.py --repo-root . --config experiments/visualization/config/viz_config.yaml
python experiments/visualization/scripts/04_make_case_dashboard.py --repo-root . --config experiments/visualization/config/viz_config.yaml
python experiments/visualization/scripts/05_make_figure_manifest.py --repo-root . --config experiments/visualization/config/viz_config.yaml
```

## Outputs

- Standardized tables: `outputs/processed_viz_data/`
- Main figures: `outputs/figures_main/`
- Appendix figures: `outputs/figures_appendix/`
- Tables: `outputs/tables/`
- Manifest: `outputs/figure_manifest.csv`
- Missing data report: `outputs/data_schema_report.md`

Fig. 1 is skipped by request and recorded as `skipped_by_request` in the manifest.

## Missing data policy

If fields or experiment variants are absent, the relevant figure is skipped and a `missing_*_plan.md` file is written. Mock fixture data is used only in unit tests and is never included in main figures.

## Leakage-safe visualization

Forecast-time evidence must remain separated from post-hoc metrics. Citation-quality sources, model-assisted summaries, structured future events, historical residual memory, and source-assisted non-citable context are tracked separately in processed tables.
