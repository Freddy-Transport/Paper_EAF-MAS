# Revised formal interface

## Status and private inputs

This is an implementation-integrity release, not a completed reproduction of
manuscript results. Supply real private inputs explicitly; source code contains
no published performance values, forecast curves, checkpoints or datasets.
Only source code and fictional fixtures are released. Old Git history is retained
and may contain previous result literals or the former architecture image.

`agents/audit_inputs.py` consumes a manifest with schema `eafmas.audit-inputs.v1`,
`synthetic: false`, `model_identity`, `source_revision`, `split_definition`, and
`files`. Entries for `train`, `val`, `test`, and `residual_memory` specify a private
JSONL `path`, file `sha256`, and row `count`. Missing provenance fails closed.
Hashes establish input identity, not retrospective proof of causal collection.
`split_definition` contains `train`, `val`, and `test`, each with `start` and
`end` timestamps defining a half-open interval. Intervals must be chronological
and disjoint; origins and request identifiers must be unique across splits.

Each forecast record contains `request_id`, `split`, `model_identity`,
`forecast_origin`, hourly `timestamps`, `channel_names`, `channel_meta`,
`raw_forecast`, admissible `events`, `evidence_sources`,
`model_assisted_summaries`, and `station_relations`. Evidence items require stable
IDs for scoring. Retrospective `outcomes`, a fixed `event_active_mask`,
`venue_channels`, and original `groups` are separate evaluation inputs; only
explicitly selected forecast-time fields enter prompts. No event availability
timestamps are inferred by this migration.

Residual records require `case_id`, train/val `split`, `support_end`,
`channel_name`, `event_type`, `median_correction`, `iqr`, `n_eff`, and
`prediction_source_sha256`. The complete support window must precede the current
origin, including during adapter training. This avoids using future residuals
inside a nominally train-only pool. Match current channel and event category;
unsupported units receive no numerical residual support.

## Numerical contract

Configuration requires `audit_thresholds` (`source_threshold`, `geo_threshold`,
`temporal_threshold`, `residual_threshold`), `gate_weights` for the five source,
geo, temporal, semantic and residual score keys, `secondary_threshold`,
`core_threshold`, `rho_full`, `rho_partial`, `random_seed`, and optional `device`.
Values must be recovered from documented training/validation configurations or
selected on validation before testing; there are no inferred test-specific defaults.
The rule auditor is the current explicit backend. Local LLM audit is not silently
substituted or represented as having run.

Feature schema `event_residual_support.v2` adds actual residual support, median
and log-count to event features. The refitted MLP exposes an identity-output
proposal `g_phi`; the controller owns clipping. Legacy tanh-output weights cannot
be reinterpreted as this model. This interface change requires a new residual-only
fit and independent results, while the forecasting backbone remains frozen.

The training configuration must explicitly provide `seed`, `epochs`, `patience`,
`batch_size`, `learning_rate`, `weight_decay`, `hidden_dim`, `dropout`,
`max_correction`, and target-denominator `epsilon`. AdamW and MSE fit the residual
MLP; validation loss selects the checkpoint. Training never loads test targets.

The controller applies a full channel-hour mask and bounds, preserves excluded
units, and returns `no_event_keep_raw`, `explain_only`, `abstain`, `apply`, or
`partial_apply`. The same immutable proposal tensor is shared by matched
controller variants. `no_bound` removes controller clipping; randomized gating
matches the actual full-controller corrected count and reports inability to
match rather than quietly changing population. Evaluation masks stay fixed.
Stratified summaries preserve input group assignments and report group counts.
The `no_event_gate`, `no_residual_support`, and `no_audit_guard` diagnostics
remove the corresponding hard eligibility check. Shared score components remain
fixed, so these variants may coincide when another check enforces the same
restriction. Gate traces report each actual mask rather than inventing reductions.
Adapter provenance binds the input-manifest digest, configuration digest and
model identity. Promoted libraries additionally bind the adapter checkpoint hash.

## Explanation and validation replay

The numerical stage finishes before fixed-library lookup. Skills operate only
on explanation analogues and organization. `evaluate_modes` hashes numerical
state and library before/after every mode. Use a locally hosted JSON-compatible
model; external transmission is disabled in this evaluation entry.

Generated JSON contains `claims` (text, evidence IDs, case IDs), `decision`,
`links` (from/to IDs and relation), and `uncertainty`. Link relations are
`event_station`, `station_time`, `historical_residual`, `residual_decision`;
`decision` is a reserved endpoint. Supply station/time IDs alongside evidence.
Every mode uses the same response schema and scorer. Only its skill organization
instructions and selected explanation analogues differ.

Scores are mode-blind rule proxies: evidence coverage counts actual references;
residual relevance checks cited train/validation cases against the same event's
channel and event type (with reference precision reported separately); multi-hop completeness
checks typed links; grounding checks traceable references; redundancy counts
repeated output claims; decision consistency compares the output to fixed state.
These are not semantic-entailment measurements. The leakage indicator covers
the input-field guard only, not a proof against every possible leakage source.
Failures and unverifiable values remain explicit, with metric denominators.

Candidate mutation changes policies only. `evolve_skills` requires an actual
validation replay callback for each original and mutation, measured digests,
replay IDs and valid counts. Existing hard promotion thresholds consume measured
scores; promotion is not itself a claim of improvement over no-skill. Baseline
comparisons require matching samples. Test-split evolution is rejected.

## Execution order

1. Audit existing private caches and map them to the manifest without changing
   origins, splits, event groups or model identity. Missing required information
   remains a blocker; do not populate it with synthetic values.
2. Run `experiments/refit_residual_adapter.py --manifest ... --config ... --output ...`.
3. Run `experiments/evolve_validated_skills.py --manifest ... --config ... --adapter ... --model ... --output ...`.
4. Run `experiments/run_formal_evaluation.py --manifest ... --config ... --adapter ... --output ...`;
   add `--explanations --library ... --model ...` for actual SkillBench outputs.

Every run requires a fresh private output directory. Preserve previous artifacts
and review differences before changing manuscript tables. GPU/model availability,
new manifest preparation, locked gate parameters and adapter refitting are
prerequisites, not automatically completed by passing CPU tests.
