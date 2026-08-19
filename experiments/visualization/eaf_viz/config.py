"""Configuration loading for the visualization pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "input_roots": ["autotemp/evidence_audited_cases_qwenplus_live_20260612_102653"],
    "output_root": "experiments/visualization/outputs",
    "method_aliases": {
        "numerical_only": "PT-MOMENT",
        "rag_explain": "EAF-MAS-X",
        "event_adapter_frozen_moment": "EAF-MAS-C",
    },
    "disabled_modes": ["event_adapter_peft_moment", "full_skill_agent"],
    "neutral_wape_gain_threshold": 0.01,
    "correction_bound_default": 0.05,
    "top_k_channels": 20,
    "skip_fig1": True,
    "formal_metric_rows": [],
}


def load_config(config_path: str | Path | None = None, repo_root: str | Path | None = None) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    path = Path(config_path) if config_path else None
    if path and path.is_file():
        loaded = _read_yaml_or_json(path)
        _deep_update(cfg, loaded)
    if repo_root:
        cfg["repo_root"] = str(Path(repo_root))
    else:
        cfg.setdefault("repo_root", ".")
    return cfg


def resolve_output_root(cfg: Dict[str, Any], repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root or cfg.get("repo_root", "."))
    out = Path(cfg.get("output_root", DEFAULT_CONFIG["output_root"]))
    return out if out.is_absolute() else root / out


def resolve_input_roots(cfg: Dict[str, Any], repo_root: str | Path | None = None) -> list[Path]:
    root = Path(repo_root or cfg.get("repo_root", "."))
    paths = []
    for item in cfg.get("input_roots", []):
        p = Path(item)
        paths.append(p if p.is_absolute() else root / p)
    return paths


def resolve_formal_metric_paths(cfg: Dict[str, Any], repo_root: str | Path | None = None) -> list[Path]:
    root = Path(repo_root or cfg.get("repo_root", "."))
    paths = []
    for item in cfg.get("formal_metric_rows", []):
        p = Path(item)
        paths.append(p if p.is_absolute() else root / p)
    return paths


def _read_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return _minimal_yaml_parse(text)


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """Parse the small config shape used by this pipeline when PyYAML is absent."""
    out: Dict[str, Any] = {}
    current_key = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            if value == "":
                out[key] = {}
            elif value.lower() in {"true", "false"}:
                out[key] = value.lower() == "true"
            elif value.startswith("["):
                out[key] = json.loads(value.replace("'", '"'))
            else:
                out[key] = value.strip('"').strip("'")
        elif current_key and line.strip().startswith("- "):
            if not isinstance(out.get(current_key), list):
                out[current_key] = []
            out[current_key].append(line.strip()[2:].strip('"').strip("'"))
        elif current_key and ":" in line:
            subkey, value = line.strip().split(":", 1)
            if not isinstance(out.get(current_key), dict):
                out[current_key] = {}
            out[current_key][subkey.strip()] = value.strip().strip('"').strip("'")
    return out


def _deep_update(base: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
