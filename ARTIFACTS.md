# Data, Model, and Experiment Artifacts

This repository contains the EAF-MAS source code, experiment entry points, baseline interfaces, tests, and documentation. The following research artifacts are intentionally excluded from Git history and will be released after paper acceptance.

## Planned Release

### Dataset package

- processed NYC hourly subway ridership for 128 station channels;
- metadata and mappings for the 128-channel ridership tensor;
- structured public-event records;
- venue-to-station mappings and the 28-channel venue-associated subset;
- train, validation, and test split manifests.

### Model package

- PT-MOMENT checkpoint and configuration for the 128 station channels;
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

The README framework illustration (`assets/eafmas_architecture_figure1.png`) is included as project documentation.

The Git repository must not contain:

- `data/` datasets or derived split files;
- `autotemp/` experiment outputs;
- generated knowledge-base indexes, retrieval corpora, predictions, metric exports, reports, or result figures beyond the README framework illustration;
- smoke-run directories, smoke checkpoints, or smoke training manifests;
- model weights such as `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, or framework-specific weight binaries;
- API keys, SSH keys, `.env` files, or private retrieval logs.

Until the artifact package is released, the experiment CLIs document the required interfaces and schemas, while unit tests use synthetic or temporary fixtures.
