# Data, Model, and Experiment Artifacts

This repository contains the EAF-MAS source code, experiment entry points, baseline interfaces, tests, and documentation. The following research artifacts are intentionally excluded from Git history and will be released after paper acceptance.

## Planned Release

### Dataset package

- processed NYC Top128 hourly subway ridership;
- Top128 channel metadata and mappings;
- structured public-event records;
- venue-to-station and event-focused channel mappings;
- train, validation, and test split manifests.

### Model package

- PT-MOMENT Top128 checkpoint and configuration;
- frozen residual-adapter checkpoint and training manifest;
- train/validation PT-MOMENT prediction exports used to construct residual targets;
- model and artifact checksums.

### Experiment package

- full forecasting metric rows;
- Evidence Audit and AutoSkill SkillBench outputs;
- Qwen-Plus and local-Qwen explanation summaries permitted for release;
- tables, figures, manifests, and runtime diagnostics;
- commands and environment specifications required for reproduction.

## Repository Policy

The Git repository must not contain:

- `data/` datasets or derived split files;
- `autotemp/` experiment outputs;
- generated knowledge-base indexes, retrieval corpora, predictions, metric exports, reports, or paper figures;
- smoke-run directories, smoke checkpoints, or smoke training manifests;
- model weights such as `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, or framework-specific weight binaries;
- API keys, SSH keys, `.env` files, or private retrieval logs.

Until the artifact package is released, the experiment CLIs document the required interfaces and schemas, while unit tests use synthetic or temporary fixtures.
