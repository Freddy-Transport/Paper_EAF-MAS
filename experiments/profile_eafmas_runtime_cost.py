#!/usr/bin/env python3
"""Runtime, cost, and latency profiling for EAF-MAS.

The profiler measures deployability costs without changing model weights,
controller thresholds, AutoSkill memory, or forecast arrays.  Qwen-Plus and
local vLLM timing are optional and explicitly status-coded when disabled or
unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.autoskill_memory import apply_residual_memory_skill  # noqa: E402
from agents.evidence_research_agent import EvidenceResearchAgent  # noqa: E402
from experiments.run_autoskill_effectiveness_study import (  # noqa: E402
    _best_docs_for_events,
    _candidate_cases_for_events,
    _load_residual_docs,
)
from experiments.run_calibration_matched_ablation import (  # noqa: E402
    _audit_pass,
    _predict_adapter_correction,
    compute_controller_scores,
    filter_events_for_window,
    prepare_timed_events,
)
from experiments.run_full_autoskill_skillbench import (  # noqa: E402
    EventWindowIndex,
    build_full_hourly_anchor_plan,
    build_residual_memory_layers,
)
from experiments.run_ccfa_fullstack_final import (  # noqa: E402
    collect_unique_high_value_events,
    normalize_event_for_fullstack,
    read_events,
    require_qwenplus_live_env,
    validate_real_lp_adapter_manifest,
)


TIMING_COLUMNS = [
    "pt_moment_inference_ms",
    "event_feature_cube_ms",
    "adapter_correction_ms",
    "evidence_lookup_ms",
    "evidence_audit_ms",
    "residual_memory_retrieval_ms",
    "autoskill_retrieval_ms",
    "explanation_generation_ms",
    "qwenplus_summary_ms",
    "total_pipeline_ms",
]

STACK_STAGE_COLUMNS = [
    "pt_moment_inference_ms",
    "event_feature_cube_ms",
    "adapter_correction_ms",
    "evidence_lookup_ms",
    "evidence_audit_ms",
    "residual_memory_retrieval_ms",
    "autoskill_retrieval_ms",
    "explanation_generation_ms",
    "qwenplus_summary_ms",
]


def ensure_dirs(root: Path) -> None:
    for sub in ["tables", "figures", "reports", "logs", "external_evidence/qwenplus_runtime_cache"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _sync_cuda(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        return


@contextmanager
def timed_stage(device: str = "cpu"):
    _sync_cuda(device)
    start = time.perf_counter()
    yield lambda: _ms(start, time.perf_counter())
    _sync_cuda(device)


def cpu_memory_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return float("nan")


def gpu_memory(device: str) -> Tuple[float, float]:
    if not str(device).startswith("cuda"):
        return float("nan"), float("nan")
    try:
        import torch

        if not torch.cuda.is_available():
            return float("nan"), float("nan")
        idx = torch.device(device).index or 0
        return (
            float(torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0)),
            float(torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0)),
        )
    except Exception:
        return float("nan"), float("nan")


def select_profile_anchors(test_anchors: Sequence[dict], sample_anchors: int, full_split: bool, seed: int) -> List[dict]:
    rows = list(test_anchors)
    if full_split or int(sample_anchors) <= 0 or int(sample_anchors) >= len(rows):
        return rows
    rng = np.random.default_rng(int(seed))
    chosen = sorted(rng.choice(np.arange(len(rows)), size=int(sample_anchors), replace=False).tolist())
    return [rows[i] for i in chosen]


def _read_active_skills(run_root: str | Path) -> List[dict]:
    path = Path(run_root) / "autoskill_skillbench_full" / "models" / "skill_memory" / "autoskill_active_skill_library.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_or_build_memory_docs(
    run_root: str | Path,
    events: Sequence[dict],
    legacy_docs: Sequence[dict],
    traffic_csv: str | Path,
    train_rows: int,
    val_rows: int,
) -> List[dict]:
    path = Path(run_root) / "autoskill_skillbench_full" / "models" / "residual_memory" / "memory_train_val_v2.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    dates = []
    with Path(traffic_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        dates = [row[0] for row in reader if row]
    memory = build_residual_memory_layers(
        events,
        legacy_docs,
        train_start=dates[0],
        train_end=dates[train_rows - 1],
        val_end=dates[train_rows + val_rows - 1],
    )
    return memory["memory_train_val_v2"]


def health_check_vllm(base_url: str, model: str) -> dict:
    if not base_url:
        return {"enabled": False, "status": "missing_base_url", "model": model}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return {"enabled": True, "status": "ok", "model": model, "models_response_chars": len(body)}
    except Exception as exc:
        return {"enabled": True, "status": "failed", "model": model, "error": str(exc)}


def call_local_vllm(base_url: str, model: str, anchor: Mapping[str, Any], event_count: int, residual_cases: Sequence[dict], audit_scores: Mapping[str, float]) -> Tuple[float, str, str]:
    prompt = (
        "Return strict JSON with fields summary, decision_reason, and uncertainty. "
        "Explain forecast-time subway ridership reasoning without actual ridership or WAPE.\n"
        f"Anchor: {anchor.get('date')}; event_count={event_count}; "
        f"residual_cases={len(residual_cases)}; audit_scores={json.dumps(audit_scores, ensure_ascii=False)}"
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a forecast-time explanation agent. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        status = "ok"
        return _ms(start, time.perf_counter()), status, raw[:200]
    except Exception as exc:
        return _ms(start, time.perf_counter()), f"failed:{type(exc).__name__}", str(exc)[:200]


def _qwenplus_disabled_row(enabled: bool) -> dict:
    status = "disabled" if not enabled else "no_selected_event"
    return {
        "qwenplus_summary_ms": np.nan,
        "qwenplus_api_calls": 0,
        "qwenplus_cache_hits": 0,
        "qwenplus_cache_misses": 0,
        "qwenplus_status": status,
    }


def profile_qwenplus_anchor(
    *,
    enabled: bool,
    cache_dir: Path,
    anchor: Mapping[str, Any],
    events: Sequence[dict],
    horizon: int,
    model: str,
    min_score: float,
    force_refresh: bool,
) -> dict:
    if not enabled:
        return _qwenplus_disabled_row(False)
    selected = collect_unique_high_value_events([anchor], events, horizon=horizon, max_events=1)
    if not selected:
        return _qwenplus_disabled_row(True)
    agent = EvidenceResearchAgent(
        cache_dir=cache_dir,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
        model=model,
        force_refresh=bool(force_refresh),
    )
    start = time.perf_counter()
    try:
        result = agent.research_events(
            selected,
            target_date=str(anchor.get("date") or selected[0].get("event_time") or ""),
            station_names=[str(selected[0].get("channel_name") or selected[0].get("station_complex") or "event_venue28")],
            max_events=1,
            min_score=min_score,
            horizon_hours=horizon,
        )
        elapsed = _ms(start, time.perf_counter())
        stats = result.stats or {}
        if int(stats.get("api_calls", 0) or 0) > 0:
            status = "api_call"
        elif int(stats.get("cache_hits", 0) or 0) > 0:
            status = "cache_hit"
        else:
            status = "no_summary"
        return {
            "qwenplus_summary_ms": elapsed,
            "qwenplus_api_calls": int(stats.get("api_calls", 0) or 0),
            "qwenplus_cache_hits": int(stats.get("cache_hits", 0) or 0),
            "qwenplus_cache_misses": int(stats.get("cache_misses", 0) or 0),
            "qwenplus_status": status,
        }
    except Exception as exc:
        return {
            "qwenplus_summary_ms": _ms(start, time.perf_counter()),
            "qwenplus_api_calls": 0,
            "qwenplus_cache_hits": 0,
            "qwenplus_cache_misses": 1,
            "qwenplus_status": f"error:{type(exc).__name__}",
        }


def summarize_runtime(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for metric in TIMING_COLUMNS:
        values = pd.to_numeric(rows[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            out.append({"metric": metric, "n": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan})
            continue
        out.append(
            {
                "metric": metric,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(out)


def write_latex_summary(summary: pd.DataFrame, root: Path) -> None:
    cols = ["metric", "n", "mean", "median", "p90", "p95"]
    latex = summary[cols].to_latex(index=False, float_format=lambda x: f"{x:.2f}", escape=True)
    (root / "tables" / "runtime_cost_summary.tex").write_text(latex, encoding="utf-8")


def write_figures(rows: pd.DataFrame, root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for label, group in [("All", rows), ("Event", rows[rows["has_event"].astype(bool)]), ("No event", rows[~rows["has_event"].astype(bool)])]:
        if group.empty:
            groups.append((label, {col: 0.0 for col in STACK_STAGE_COLUMNS}))
        else:
            groups.append((label, {col: float(pd.to_numeric(group[col], errors="coerce").fillna(0.0).mean()) for col in STACK_STAGE_COLUMNS}))
    labels = [g[0] for g in groups]
    bottoms = np.zeros(len(groups), dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(STACK_STAGE_COLUMNS)))
    for color, col in zip(colors, STACK_STAGE_COLUMNS):
        values = np.array([g[1][col] for g in groups], dtype=float)
        ax.bar(labels, values, bottom=bottoms, label=col.replace("_ms", "").replace("_", " "), color=color)
        bottoms += values
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("EAF-MAS runtime breakdown by pipeline stage")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "runtime_cost_stacked_bar.png", dpi=220)
    fig.savefig(fig_dir / "runtime_cost_stacked_bar.pdf")
    plt.close(fig)

    event_values = pd.to_numeric(rows[rows["has_event"].astype(bool)]["total_pipeline_ms"], errors="coerce").dropna()
    non_values = pd.to_numeric(rows[~rows["has_event"].astype(bool)]["total_pipeline_ms"], errors="coerce").dropna()
    data = []
    tick_labels = []
    if len(event_values):
        data.append(event_values)
        tick_labels.append(f"Event\nn={len(event_values)}")
    if len(non_values):
        data.append(non_values)
        tick_labels.append(f"No event\nn={len(non_values)}")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    if data:
        ax.boxplot(data, labels=tick_labels, showfliers=False)
        ax.set_ylabel("Total pipeline latency (ms)")
    else:
        ax.text(0.5, 0.5, "No timing rows available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Event vs non-event runtime distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "runtime_cost_event_vs_nonevent_boxplot.png", dpi=220)
    fig.savefig(fig_dir / "runtime_cost_event_vs_nonevent_boxplot.pdf")
    plt.close(fig)


def write_interpretation(rows: pd.DataFrame, summary: pd.DataFrame, root: Path, manifest: Mapping[str, Any]) -> None:
    summary_idx = summary.set_index("metric")
    total_mean = float(summary_idx.loc["total_pipeline_ms", "mean"]) if "total_pipeline_ms" in summary_idx.index else float("nan")
    total_p95 = float(summary_idx.loc["total_pipeline_ms", "p95"]) if "total_pipeline_ms" in summary_idx.index else float("nan")
    stage_means = {
        col: float(summary_idx.loc[col, "mean"])
        for col in STACK_STAGE_COLUMNS
        if col in summary_idx.index and pd.notna(summary_idx.loc[col, "mean"])
    }
    bottleneck = max(stage_means.items(), key=lambda item: item[1])[0] if stage_means else "unknown"
    lines = [
        "# Runtime / Cost / Latency Interpretation",
        "",
        "This profiling run measures EAF-MAS deployability costs without training, threshold tuning, or test-time Skill updates.",
        "",
        "## Runtime Summary",
        f"- Profiled anchors: `{len(rows)}`.",
        f"- Mean total pipeline latency: `{total_mean:.2f}` ms.",
        f"- p95 total pipeline latency: `{total_p95:.2f}` ms.",
        f"- Largest mean stage: `{bottleneck}`.",
        "",
        "## Cost and Cache Notes",
        f"- Qwen-Plus live enabled: `{manifest['qwenplus']['live_enabled']}`.",
        f"- Qwen-Plus API calls: `{int(rows['qwenplus_api_calls'].sum())}`.",
        f"- Qwen-Plus cache hits: `{int(rows['qwenplus_cache_hits'].sum())}`.",
        f"- Qwen-Plus cache misses: `{int(rows['qwenplus_cache_misses'].sum())}`.",
        f"- Local vLLM status: `{manifest['local_vllm']['status']}`.",
        "",
        "## Paper Wording",
        "- Use these results to support measured deployability and module-level cost attribution.",
        "- Do not claim real-time operation unless the reported p95 latency meets the target deployment budget.",
        "- Qwen-Plus is an optional model-assisted summary component; PT-MOMENT, event features, adapter, and audit are the core numerical path.",
    ]
    (root / "reports" / "runtime_cost_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def hardware_manifest(device: str) -> dict:
    out = {
        "device": device,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_memory_mb": cpu_memory_mb(),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            idx = torch.device(device).index or 0
            out["gpu_name"] = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            out["gpu_total_memory_mb"] = float(props.total_memory / (1024.0 * 1024.0))
    except Exception as exc:
        out["torch_error"] = str(exc)
    return out


def finalize_outputs(root: Path, rows: pd.DataFrame, manifest: dict) -> dict:
    ensure_dirs(root)
    rows.to_csv(root / "tables" / "runtime_cost_rows.csv", index=False)
    summary = summarize_runtime(rows)
    summary.to_csv(root / "tables" / "runtime_cost_summary.csv", index=False)
    write_latex_summary(summary, root)
    write_figures(rows, root)
    manifest = dict(manifest)
    manifest.update(
        {
            "status": "completed",
            "created_at": _now_iso(),
            "row_count": int(len(rows)),
            "event_group_counts": {
                "event": int(rows["has_event"].astype(bool).sum()) if len(rows) else 0,
                "non_event": int((~rows["has_event"].astype(bool)).sum()) if len(rows) else 0,
            },
            "outputs": {
                "rows": "tables/runtime_cost_rows.csv",
                "summary": "tables/runtime_cost_summary.csv",
                "figures": [
                    "figures/runtime_cost_stacked_bar.png",
                    "figures/runtime_cost_event_vs_nonevent_boxplot.png",
                ],
            },
        }
    )
    (root / "reports" / "runtime_cost_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_interpretation(rows, summary, root, manifest)
    return {"summary": summary, "manifest": manifest}


def run_synthetic_smoke(
    output_root: str | Path,
    sample_anchors: int = 3,
    seed: int = 13,
    enable_qwenplus_live: bool = False,
    enable_local_vllm: bool = False,
    llm_model: str = "Qwen/Qwen3-8B",
) -> dict:
    root = Path(output_root)
    ensure_dirs(root)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(sample_anchors)):
        has_event = i % 2 == 0
        qwen = _qwenplus_disabled_row(enable_qwenplus_live)
        explanation_ms = float(rng.uniform(15, 25)) if enable_local_vllm else 0.0
        stage = {
            "pt_moment_inference_ms": float(rng.uniform(8, 12)),
            "event_feature_cube_ms": float(rng.uniform(1, 3)),
            "adapter_correction_ms": float(rng.uniform(0.4, 1.0)),
            "evidence_lookup_ms": float(rng.uniform(0.2, 0.6)),
            "evidence_audit_ms": float(rng.uniform(0.2, 0.6)),
            "residual_memory_retrieval_ms": float(rng.uniform(0.5, 1.5)),
            "autoskill_retrieval_ms": float(rng.uniform(0.2, 0.8)),
            "explanation_generation_ms": explanation_ms,
            "qwenplus_summary_ms": qwen["qwenplus_summary_ms"],
        }
        total = float(sum(v for v in stage.values() if pd.notna(v)))
        rows.append(
            {
                "anchor": 1000 + i,
                "date": f"2023-06-{1+i:02d} 10:00:00",
                "has_event": bool(has_event),
                "event_count": int(2 if has_event else 0),
                **stage,
                "total_pipeline_ms": total,
                "gpu_memory_allocated_mb": 0.0,
                "gpu_memory_reserved_mb": 0.0,
                "cpu_memory_mb": cpu_memory_mb(),
                **qwen,
                "local_vllm_model": llm_model,
                "qwenplus_model": os.environ.get("LLM_MODEL", "qwen-plus"),
                "cache_enabled": True,
                "explanation_status": "synthetic_ok" if enable_local_vllm else "disabled",
                "notes": "synthetic smoke row",
            }
        )
    manifest = {
        "synthetic": True,
        "run_root": "",
        "sample_protocol": {"sample_anchors": int(sample_anchors), "seed": int(seed), "full_split": False},
        "hardware": hardware_manifest("cpu"),
        "models": {"local_vllm_model": llm_model, "qwenplus_model": os.environ.get("LLM_MODEL", "qwen-plus")},
        "qwenplus": {"live_enabled": bool(enable_qwenplus_live), "status": "disabled" if not enable_qwenplus_live else "synthetic_no_live"},
        "local_vllm": {"enabled": bool(enable_local_vllm), "status": "synthetic_ok" if enable_local_vllm else "disabled"},
        "claim_boundary": "Synthetic rows validate output contracts only.",
    }
    return finalize_outputs(root, pd.DataFrame(rows), manifest)


def run_runtime_profile(args: argparse.Namespace) -> dict:
    import torch
    from sklearn.preprocessing import StandardScaler

    from agents.event_adapter import build_event_feature_cube, channel_meta_by_name, load_adapter, load_channel_map, load_events_json
    from agents.numerical_agent import NumericalPredictionAgent
    from event_post_training.config import resolve_lp_model_path
    from experiments.run_ccfa_fullstack_final import _inverse_transform_batch
    from experiments.run_full_paper_results import load_channel_names

    root = Path(args.output_root)
    ensure_dirs(root)
    if args.enable_qwenplus_live:
        qwen_env = require_qwenplus_live_env(True)
    else:
        qwen_env = {"enabled": False, "env_present": False, "model": os.environ.get("LLM_MODEL", "qwen-plus")}

    traffic_csv = Path(args.traffic_csv)
    df = pd.read_csv(traffic_csv, parse_dates=["date"])
    channel_names = load_channel_names(traffic_csv)
    val_anchors, test_anchors = build_full_hourly_anchor_plan(traffic_csv, horizon=args.horizon, train_rows=args.train_rows, val_rows=args.val_rows)
    selected_anchors = select_profile_anchors(test_anchors, args.sample_anchors, args.full_split, args.seed)

    raw_events = [normalize_event_for_fullstack(event) for event in read_events(args.events_json)]
    skill_event_index = EventWindowIndex(raw_events)
    adapter_events = load_events_json(args.events_json)
    timed_adapter_events = prepare_timed_events(adapter_events)
    channel_meta = channel_meta_by_name(load_channel_map(args.channel_map))
    legacy_docs = _load_residual_docs(args.residual_kb_preview)
    residual_docs = _load_or_build_memory_docs(args.run_root, raw_events, legacy_docs, traffic_csv, args.train_rows, args.val_rows)
    active_skills = _read_active_skills(args.run_root)
    residual_skill = next((skill for skill in active_skills if skill.get("skill_category") == "residual_memory_skill"), {})

    data_df = df.drop(columns=["date"]).infer_objects(copy=False).interpolate(method="cubic")
    data_np = data_df.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(data_np[: args.train_rows])
    data_norm = scaler.transform(data_np).astype(np.float32)

    lp_dir, lp_source = resolve_lp_model_path(Path(args.lp_model_path), PROJECT_ROOT)
    validate_real_lp_adapter_manifest(Path(args.adapter_path) / "training_manifest.json")
    agent = NumericalPredictionAgent(
        model_path=str(root / "__no_gca_model_for_runtime_profile__"),
        device=args.device,
        forecast_horizon=args.horizon,
        n_channels=len(channel_names),
        channel_names=list(channel_names),
        lp_model_path=str(lp_dir),
        lp_data_path=str(traffic_csv),
        allow_lp_training_fallback=False,
    )
    adapter, _adapter_config = load_adapter(args.adapter_path, device=args.device)
    vllm_status = health_check_vllm(args.llm_base_url, args.llm_model) if args.enable_local_vllm else {"enabled": False, "status": "disabled", "model": args.llm_model}
    seq_len = int(args.seq_len)
    rows = []

    for idx, anchor in enumerate(selected_anchors):
        anchor_start = time.perf_counter()
        anchor_id = int(anchor["anchor"])
        date_text = str(anchor["date"])
        timestamps = [pd.Timestamp(ts) for ts in df["date"].iloc[anchor_id : anchor_id + args.horizon].tolist()]
        row: Dict[str, Any] = {
            "anchor": anchor_id,
            "date": date_text,
            "local_vllm_model": args.llm_model,
            "qwenplus_model": os.environ.get("LLM_MODEL", "qwen-plus"),
            "cache_enabled": True,
            "notes": "",
        }

        lookup_start = time.perf_counter()
        window_events = skill_event_index.events_for_anchor(date_text, horizon=args.horizon, max_events=None)
        row["evidence_lookup_ms"] = _ms(lookup_start, time.perf_counter())
        row["has_event"] = bool(window_events)
        row["event_count"] = int(len(window_events))

        histories = np.stack([data_norm[anchor_id - seq_len : anchor_id].T], axis=0)
        x = torch.tensor(histories, dtype=torch.float32, device=args.device)
        input_mask = torch.ones(1, seq_len, dtype=torch.float32, device=args.device)
        _sync_cuda(args.device)
        start = time.perf_counter()
        with torch.no_grad():
            output = agent.model(x_enc=x, input_mask=input_mask)
            pred_norm = output.forecast.detach().cpu().numpy().astype(np.float32)
        raw = _inverse_transform_batch(scaler, pred_norm)[0].astype(np.float32)
        _sync_cuda(args.device)
        row["pt_moment_inference_ms"] = _ms(start, time.perf_counter())

        feature_start = time.perf_counter()
        filtered_events = filter_events_for_window(
            timed_adapter_events,
            pd.Timestamp(timestamps[0]),
            pd.Timestamp(timestamps[-1]),
            back_hours=args.event_prefilter_back_hours,
            forward_hours=args.event_prefilter_forward_hours,
        )
        features = build_event_feature_cube(raw, [str(ts) for ts in timestamps], channel_names, filtered_events, channel_meta)
        row["event_feature_cube_ms"] = _ms(feature_start, time.perf_counter())

        audit_start = time.perf_counter()
        scores = compute_controller_scores(features)
        _audit_pass(scores, args.controller_thresholds)
        row["evidence_audit_ms"] = _ms(audit_start, time.perf_counter())

        _sync_cuda(args.device)
        adapter_start = time.perf_counter()
        _predict_adapter_correction(adapter, features, args.device, force_event_active=None, disable_bound=False)
        _sync_cuda(args.device)
        row["adapter_correction_ms"] = _ms(adapter_start, time.perf_counter())

        residual_start = time.perf_counter()
        candidate_cases = _candidate_cases_for_events(window_events, residual_docs, max_cases=10)
        best_cases = _best_docs_for_events(window_events, residual_docs, max_cases=4)
        row["residual_memory_retrieval_ms"] = _ms(residual_start, time.perf_counter())

        skill_start = time.perf_counter()
        if residual_skill and candidate_cases:
            apply_residual_memory_skill(candidate_cases, residual_skill)
        row["autoskill_retrieval_ms"] = _ms(skill_start, time.perf_counter())

        audit_means = {
            key: float(np.nanmean(value)) if np.asarray(value).size else 0.0
            for key, value in scores.items()
            if key.endswith("_score")
        }
        if args.enable_local_vllm and vllm_status.get("status") == "ok":
            elapsed, status, note = call_local_vllm(args.llm_base_url, args.llm_model, anchor, len(window_events), best_cases, audit_means)
            row["explanation_generation_ms"] = elapsed
            row["explanation_status"] = status
            row["notes"] = note
        else:
            row["explanation_generation_ms"] = np.nan if args.enable_local_vllm else 0.0
            row["explanation_status"] = vllm_status.get("status", "disabled")

        qwen = profile_qwenplus_anchor(
            enabled=args.enable_qwenplus_live,
            cache_dir=root / "external_evidence" / "qwenplus_runtime_cache",
            anchor=anchor,
            events=raw_events,
            horizon=args.horizon,
            model=os.environ.get("LLM_MODEL", "qwen-plus"),
            min_score=args.qwenplus_min_score,
            force_refresh=args.qwenplus_force_refresh,
        )
        row.update(qwen)
        alloc, reserved = gpu_memory(args.device)
        row["gpu_memory_allocated_mb"] = alloc
        row["gpu_memory_reserved_mb"] = reserved
        row["cpu_memory_mb"] = cpu_memory_mb()
        row["total_pipeline_ms"] = _ms(anchor_start, time.perf_counter())
        rows.append(row)
        if (idx + 1) % max(1, int(args.progress_every)) == 0 or idx + 1 == len(selected_anchors):
            print(f"[runtime-profile] processed {idx + 1}/{len(selected_anchors)} anchors", flush=True)

    manifest = {
        "synthetic": False,
        "run_root": str(args.run_root),
        "run_root_read_only": True,
        "sample_protocol": {
            "sample_anchors": int(len(selected_anchors)),
            "requested_sample_anchors": int(args.sample_anchors),
            "seed": int(args.seed),
            "full_split": bool(args.full_split),
            "available_test_anchors": int(len(test_anchors)),
        },
        "hardware": hardware_manifest(args.device),
        "models": {
            "pt_moment_path": str(lp_dir),
            "lp_source": lp_source,
            "adapter_path": str(args.adapter_path),
            "local_vllm_model": args.llm_model,
            "qwenplus_model": os.environ.get("LLM_MODEL", "qwen-plus"),
        },
        "qwenplus": {
            "live_enabled": bool(args.enable_qwenplus_live),
            "env_present": bool(qwen_env.get("env_present")),
            "force_refresh": bool(args.qwenplus_force_refresh),
            "min_score": float(args.qwenplus_min_score),
            "api_calls": int(sum(int(row.get("qwenplus_api_calls", 0) or 0) for row in rows)),
            "cache_hits": int(sum(int(row.get("qwenplus_cache_hits", 0) or 0) for row in rows)),
            "cache_misses": int(sum(int(row.get("qwenplus_cache_misses", 0) or 0) for row in rows)),
        },
        "local_vllm": vllm_status,
        "claim_boundary": "Runtime profiling measures deployability costs only; no training, threshold tuning, skill promotion, or forecast-array mutation is performed.",
    }
    return finalize_outputs(root, pd.DataFrame(rows), manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Profile EAF-MAS runtime, cost, and latency.")
    parser.add_argument("--run_root", default=str(PROJECT_ROOT / "autotemp" / "ccfa_fullstack_summary_only_20260619_123041"))
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "autotemp" / f"revision_trc_tits_ccfa_{timestamp}" / "runtime_cost"))
    parser.add_argument("--traffic_csv", default=str(PROJECT_ROOT / "data" / "nyc_top128_station_hourly_flow.csv"))
    parser.add_argument("--events_json", default=str(PROJECT_ROOT / "data" / "nyc_top128_station_events.json"))
    parser.add_argument("--channel_map", default=str(PROJECT_ROOT / "data" / "nyc_top128_channel_map.json"))
    parser.add_argument("--lp_model_path", default=str(PROJECT_ROOT / "experiments" / "outputs" / "lp_fallback_nyc_top128"))
    parser.add_argument("--adapter_path", default=str(PROJECT_ROOT / "experiments" / "outputs" / "event_adapter_formal_frozen_real_lpmoment_20260612_084207"))
    parser.add_argument("--residual_kb_preview", default=str(PROJECT_ROOT / "agents" / "knowledge_base_residual_venue37" / "documents_preview.json"))
    parser.add_argument("--sample_anchors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--full_split", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_rows", type=int, default=12 * 30 * 24)
    parser.add_argument("--val_rows", type=int, default=4 * 30 * 24)
    parser.add_argument("--enable_local_vllm", action="store_true")
    parser.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--enable_qwenplus_live", action="store_true")
    parser.add_argument("--qwenplus_force_refresh", action="store_true")
    parser.add_argument("--qwenplus_min_score", type=float, default=0.75)
    parser.add_argument("--event_prefilter_back_hours", type=float, default=72.0)
    parser.add_argument("--event_prefilter_forward_hours", type=float, default=2.0)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--synthetic_smoke", action="store_true")
    args = parser.parse_args(argv)
    from experiments.run_calibration_matched_ablation import ControllerThresholds

    args.controller_thresholds = ControllerThresholds()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.synthetic_smoke:
        run_synthetic_smoke(
            args.output_root,
            sample_anchors=args.sample_anchors,
            seed=args.seed,
            enable_qwenplus_live=args.enable_qwenplus_live,
            enable_local_vllm=args.enable_local_vllm,
            llm_model=args.llm_model,
        )
    else:
        run_runtime_profile(args)


if __name__ == "__main__":
    main()
