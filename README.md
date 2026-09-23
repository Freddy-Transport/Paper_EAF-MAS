# EAF-MAS: Selective Residual Calibration for Planned Special Events

Code repository accompanying **Urban Transit Demand Forecasting during Planned Special Events: Selective Residual Calibration**.

## Implementation audit status

The active evaluation entry points use real explanation outputs, measured
validation replay, case-backed numerical residual support, and cell-level
bounded calibration. CPU contract tests are separate from scientific
reproduction: revised full-data evaluation and residual-MLP refitting remain
pending. Previously generated scores are not embedded in this release.

Two audit items remain deferred: the PT-MOMENT-specific core is not provided
by the retained LP interface, and complete event-publication-time verification
remains unresolved. Input model identities are retained verbatim; LP artifacts
are not relabeled as PT-MOMENT results. The architecture description below
describes the manuscript, not proof of complete numerical reproduction.

Planned special events can produce sharp but spatially concentrated deviations from regular metro ridership patterns. EAF-MAS separates history-based baseline forecasting from the decision to revise an event-exposed prediction. PT-MOMENT generates the network-wide baseline, and selective residual calibration applies a bounded local adjustment when spatial relevance and historical residual support justify intervention. Otherwise, the baseline is retained. New York City subway ridership provides the empirical case study.

The two core methodological components are:

- **PT-MOMENT**: parameter-efficient adaptation of a pretrained time-series model through channel-conditioned prompting and lightweight cross-channel conditioning.
- **Evidence-gated bounded residual calibration**: forecast-time evidence assessment, channel-hour localization, and control of the relative correction applied to eligible units.

Evidence retrieval, auditing, and validation-evolved skill memory support evidence organization and traceable explanations. This repository contains the forecasting workflow, residual-adapter training pipeline, evidence and explanation components, numerical baseline interfaces, experiment entry points, and tests.

## Framework

Architecture assets containing example forecasts are withheld during submission.

PT-MOMENT first generates a history-based raw forecast. The evidence auditor assesses scheduled events, station-event relations, source availability, and historical residual support. The event-station-channel gate identifies eligible channel-hour units, and the calibration controller accepts a bounded residual adjustment or preserves the baseline. The explanation agent records the evidence, decision, correction scope, and uncertainty.

**EAF-MAS-X** retains the raw forecast and provides explanations. **EAF-MAS-C** additionally enables selective residual calibration for eligible units.

### Component roles

| Component | Responsibility | Output |
| --- | --- | --- |
| PT-MOMENT | Generates the history-based forecast for all 128 station channels | Baseline channel-hour forecasts |
| Evidence-RAG | Organizes structured records, historical cases, and retrieved or model-assisted evidence | Evidence bundle with provenance |
| Evidence auditor | Assesses source, temporal, geographic, semantic, and residual support | Audit scores and inclusion or exclusion reasons |
| Event-station-channel gate | Localizes event relevance to individual channel-hour units | Eligibility mask |
| Residual adapter | Uses the raw forecast, event features, and fixed historical residual support | Relative residual proposals |
| Calibration controller | Applies eligibility, conflict checks, and relative-correction bounds | Bounded local calibration or baseline preservation |
| Explanation agent | Summarizes evidence, historical analogues, the decision, and uncertainty | Structured decision trace |
| Skill memory | Reuses validation-promoted evidence-processing procedures | Explanation-oriented analogue ranking and organization |

The forecasting backbone and forecast head are frozen during residual-adapter training; only the residual MLP is optimized in that stage. PT-MOMENT adaptation is a separate training stage.

## Knowledge and Evidence Layers

EAF-MAS separates four forms of memory so their provenance and evaluation remain explicit:

1. **Structured event knowledge**: event time, venue, category, impact tier, confidence, duration, and station-channel matches.
2. **Historical residual memory**: train/validation-only PT-MOMENT residual analogues, including event type, day type, station rank, correction direction, dispersion, and effective sample size.
3. **External evidence memory**: retrieved records and Qwen-Plus model-assisted summaries, with source type, availability time, and provenance retained. Citation status depends on usable source provenance.
4. **Skill memory**: routing, evidence-audit, residual-analogue, and abstention procedures generated and promoted through validation replay, with the promoted library fixed during test evaluation.

Fixed numerical residual support supplies the gate and residual adapter. Skill-guided historical analogues are selected for explanation after the numerical decision. SkillBench evaluates evidence and explanation organization under invariant forecasts and controller outputs. Realized ridership and forecast errors belong to retrospective evaluation, while forecast-time reasoning uses information available when the forecast is issued.

## Dataset and Evaluation Scope

- **Ridership tensor**: 128 station channels with 12,287 hourly observations from 1 February 2022 to 28 June 2023.
- **Event corpus**: 33,027 station-event records linked through spatial and temporal relations.
- **Venue-associated subset**: 28 station channels associated with major event venues.
- **Chronological split**: 8,640 hourly observations for training, 2,880 for validation, and the remaining observations for held-out testing.
- **Rolling evaluation**: 576 rolling forecasts, each covering the next 192 hours. The forecast origin advances by one hour, so successive forecast windows overlap by design.

Here, each channel represents a station. The manuscript distinguishes three evaluation scopes:

| Evaluation scope | Population | Purpose |
| --- | --- | --- |
| Network-wide | All forecast hours of all 128 channels | General numerical forecasting accuracy |
| Venue-associated WAPE | All forecast hours of the 28 venue-associated channels | Forecasting quality around event venues |
| Event-active WAPE | Channel-hour units linked to an event and within its predefined effect window | Forecasting performance during active event exposure |

Station selection and normalization statistics are determined from the training split. Residual-adapter fitting, controller configuration, and skill promotion use training and validation data. The held-out rolling evaluation assesses forecasting, calibration gain, the shares of improved and degraded forecasts, worst regret, intervention scope, non-event preservation, evidence quality, explanation consistency, and runtime. SkillBench reports rule-audited explanation-quality proxies.

## Installation

Python 3.10+ is recommended. Install the research dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

CUDA, PyTorch, and vLLM builds are hardware-specific. Install a CUDA-compatible PyTorch/vLLM stack separately when running PT-MOMENT or local Qwen inference on GPU.

Datasets, model checkpoints, and generated experiment outputs are not included in this repository. They will be released after paper acceptance. See [`ARTIFACTS.md`](ARTIFACTS.md) for the planned release contents.

## LLM Services

### Local open-source explanation model

The explanation agent uses a local Qwen3-8B model served through an OpenAI-compatible vLLM endpoint. The workflow expects:

```text
http://127.0.0.1:8000/v1
served model name: Qwen/Qwen3-8B
```

Start vLLM with your local Qwen3-8B snapshot and verify `/v1/models` before running an explanation experiment.

### Qwen-Plus evidence enrichment

Qwen-Plus optionally enriches high-value event records with model-assisted summaries. Credentials are supplied through runtime environment variables:

```bash
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="<your-runtime-key>"
export LLM_MODEL="qwen-plus"
```

Do not commit API keys, `.env` files, raw credentials, or private retrieval logs.

## Experiment Entry Points

| Study | Entry point | Purpose |
| --- | --- | --- |
| Audited calibration and SkillBench | `experiments/run_formal_evaluation.py` | Fixed-input ablations, per-cell decision traces, stratified summaries and actual-output scoring |
| Validation skill evolution | `experiments/evolve_validated_skills.py` | Execute each candidate/mutation and measure validation output before promotion |
| Residual-MLP refit | `experiments/refit_residual_adapter.py` | Fit versioned historical-residual features from private fixed backbone predictions |
| Numerical baselines | `baseline/run_baselines.py` | Traditional and deep forecasting baseline interface |

Run `python <entry-point> --help` for the complete experiment-specific contract.

See [FORMAL_INTERFACE.md](FORMAL_INTERFACE.md) for required private manifests,
configuration fields, score definitions, and staged execution. Missing real
inputs or incompatible adapter schemas stop the formal run. No synthetic
fallback is available. Old execution paths and result-specific exporters are
archived on the private server and excluded from this release.

The dataset, checkpoints, and experiment artifacts required to reproduce the reported tables and figures will be published after paper acceptance.

## Repository Layout

```text
agents/                 Shared audited numerical path, evidence components, output scoring and skill memory
event_post_training/    Retained prediction-cache and historical schema utilities (not the revised fitting entry)
experiments/            Formal evaluation, validated skill evolution and residual refitting
baseline/               Official-interface wrappers for traditional/deep numerical baselines
tests/                  Synthetic unit and regression tests; no paper-result assertions
momentfm/               Retained upstream MOMENT implementation used by the numerical backbone
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Evaluation Focus

The rule-audited grounding/relevance metrics quantify reference traceability,
not semantic entailment or human-rated quality. Every summary includes effective
sample counts and generation failures. Unknown measurements are null, never
inferred perfect scores. Revised proxy definitions require fresh evaluation;
old and new scores are not directly interchangeable.

- **Baseline quality** across network-wide, venue-associated, and event-active demand, and across forecast horizons.
- **Accuracy-intervention scope trade-off**: whether improvement comes from selecting appropriate locations rather than modifying more predictions.
- **Relative-correction control** and deterioration risk, including preservation of non-event forecasts.
- **Event-stratified robustness** across sufficiently represented event types, venues, and station groups.
- **Evidence and explanation quality**, computational cost, and representative calibration or preservation decisions.

## Acknowledgements

This project builds on the open-source [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) implementation and uses its forecasting backbone within a new event-aware multi-agent pipeline. We also draw methodological inspiration from retrieval-augmented generation, ReAct-style tool reasoning, Reflexion, and AutoSkill-style experience-to-skill evolution.

The retained MOMENT implementation is distributed under its original MIT license; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No license grant for the remaining EAF-MAS research code is implied by that third-party notice.

If you use the retained MOMENT implementation, please cite the original work:

```bibtex
@inproceedings{goswami2024moment,
  title     = {MOMENT: A Family of Open Time-series Foundation Models},
  author    = {Goswami, Mononito and Szafer, Konrad and Choudhry, Arjun and Cai, Yifu and Li, Shuo and Dubrawski, Artur},
  booktitle = {International Conference on Machine Learning},
  year      = {2024}
}
```

The EAF-MAS paper citation will be added after publication.
