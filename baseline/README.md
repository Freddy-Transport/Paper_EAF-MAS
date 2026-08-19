# PT-MOMENT Baseline Workspace

This folder is an independent baseline workspace for comparing conventional and lightweight deep forecasting models against the paper-facing PT-MOMENT numerical forecasting agent.

The workspace uses the same NYC Top128 hourly ridership tensor and the same full-test protocol used by the EAF-MAS paper experiments:

- traffic CSV: `/root/autodl-tmp/0206moment/data/nyc_top128_station_hourly_flow.csv`
- channels: 128 station/channel columns
- lookback: 512 hours
- horizon: 192 hours
- train rows: 8640
- validation rows: 2880
- test anchors: 576 complete anchors

## Baselines

First-batch baselines:

- `seasonal_week`: repeats the same 192-hour block from one week earlier.
- `last_day`: repeats the same 192-hour block from one day earlier.
- `dlinear`: decomposition linear model.
- `nlinear`: normalized linear model.
- `patchtst`: compact PatchTST-style Transformer baseline.
- `itransformer`: compact iTransformer-style baseline.

The neural models are implemented locally with PyTorch to avoid dependency on external official repositories during the first reproducibility pass. If official-repository baselines are required later, keep them in a separate subfolder and record commit hashes.

## Quick Commands

Use the server Python explicitly because non-login SSH shells do not always expose `python` on `PATH`:

```bash
cd /root/autodl-tmp/0206moment/baseline
/root/miniconda3/bin/python run_baselines.py --config config.json --models seasonal_week,last_day
```

Neural smoke test:

```bash
/root/miniconda3/bin/python run_baselines.py --config config.json --models dlinear --max_epochs 1 --limit_train_batches 2 --device cpu
```

Full training should be launched only after GPU availability and runtime budget are confirmed.

## Outputs

Each run writes to `runs/<timestamp>/`:

- `metrics/baseline_summary.csv`
- `metrics/baseline_summary.json`
- `predictions/<model>.npz`
- `reports/run_manifest.json`
- `logs/<model>.log`

The baseline outputs are numerical-forecasting artifacts only. They do not use event text, Evidence-RAG, AutoSkill, Qwen-Plus, local LLM reasoning, or residual-adapter corrections.
