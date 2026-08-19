"""Official baseline adapter registry for the NYC Top128 protocol.

This module records how Table 5/Table 6 baseline rows should be tied to
upstream official repositories. It intentionally does not train models. The
existing local PyTorch implementations in ``run_baselines.py`` remain fallback
smoke-test implementations; paper-facing official runs should first bootstrap
the upstream repositories and record their commit hashes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOCAL_FALLBACK_MODELS = {"seasonal_week", "last_day"}
OFFICIAL_MODEL_TO_SOURCE = {
    "dlinear": "ltsf_linear",
    "nlinear": "ltsf_linear",
    "patchtst": "patchtst",
    "itransformer": "itransformer",
}


@dataclass(frozen=True)
class OfficialSource:
    source_id: str
    models: List[str]
    official_repo: str
    repo_name: str
    local_path: str
    commit: Optional[str]
    license: str
    adapter_status: str
    paper_role: str

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "OfficialSource":
        return cls(
            source_id=row["source_id"],
            models=list(row["models"]),
            official_repo=row["official_repo"],
            repo_name=row["repo_name"],
            local_path=row["local_path"],
            commit=row.get("commit"),
            license=row.get("license", "see upstream repository"),
            adapter_status=row.get("adapter_status", "unknown"),
            paper_role=row.get("paper_role", ""),
        )


def load_registry(path: str | Path = "official_sources.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_sources(path: str | Path = "official_sources.json") -> Iterable[OfficialSource]:
    registry = load_registry(path)
    for row in registry.get("sources", []):
        yield OfficialSource.from_dict(row)


def source_for_model(model: str, path: str | Path = "official_sources.json") -> Optional[OfficialSource]:
    source_id = OFFICIAL_MODEL_TO_SOURCE.get(model)
    if source_id is None:
        return None
    for source in iter_sources(path):
        if source.source_id == source_id:
            return source
    raise KeyError(f"Registry is missing source_id={source_id!r} for model={model!r}")


def git_head(repo_path: str | Path) -> Optional[str]:
    repo_path = Path(repo_path)
    if not (repo_path / ".git").exists():
        return None
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True)
    except Exception:
        return None
    return out.strip()


def official_execution_plan(
    model: str,
    config_path: str = "config.json",
    registry_path: str = "official_sources.json",
) -> Dict[str, Any]:
    """Return a non-executing plan for a paper-facing official baseline run."""
    if model in LOCAL_FALLBACK_MODELS:
        return {
            "model": model,
            "implementation": "local_deterministic",
            "status": "ready",
            "notes": "Naive baselines are deterministic and do not require an upstream repository.",
        }
    source = source_for_model(model, registry_path)
    if source is None:
        raise ValueError(f"Unsupported model for official adapter planning: {model}")
    repo_path = Path(source.local_path)
    return {
        "model": model,
        "implementation": "official_repository_adapter",
        "source_id": source.source_id,
        "official_repo": source.official_repo,
        "local_path": source.local_path,
        "repo_present": repo_path.exists(),
        "resolved_commit": git_head(repo_path),
        "config_path": config_path,
        "protocol": "NYC Top128, lookback=512, horizon=192, train=8640, val=2880, test=576 anchors",
        "status": "interface_ready_not_executed",
        "notes": (
            "Clone the official repository with bootstrap_official_baselines.py, "
            "adapt its dataset loader to the exported Top128 arrays, then run on "
            "the same split. This function does not train or evaluate models."
        ),
    }


def write_plan(
    models: Iterable[str],
    output_path: str | Path,
    config_path: str = "config.json",
    registry_path: str = "official_sources.json",
) -> None:
    plans = [
        official_execution_plan(model, config_path=config_path, registry_path=registry_path)
        for model in models
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"plans": plans}, f, indent=2)
