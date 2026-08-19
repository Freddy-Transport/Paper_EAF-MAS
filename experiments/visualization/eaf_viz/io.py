"""Repository scanning and file loading helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd


SCAN_KEYWORDS = [
    "metrics",
    "forecast",
    "raw",
    "adjusted",
    "event_adapter",
    "rag_explain",
    "calibration",
    "evidence",
    "residual",
    "memory",
    "case",
]


def scan_repo_outputs(repo_root: str | Path, keywords: Iterable[str] | None = None) -> pd.DataFrame:
    root = Path(repo_root)
    keys = [k.lower() for k in (keywords or SCAN_KEYWORDS)]
    rows: List[dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_text = str(rel).lower()
        matched = [k for k in keys if k in rel_text]
        if not matched and path.suffix.lower() not in {".json", ".csv", ".md"}:
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(rel),
                "file_type": path.suffix.lower().lstrip(".") or "unknown",
                "size": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" "),
                "matched_keywords": ";".join(matched),
            }
        )
    return pd.DataFrame(rows).sort_values(["file_type", "path"]) if rows else pd.DataFrame()


def write_scan_report(df: pd.DataFrame, out_md: str | Path) -> None:
    out = Path(out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Repository Output Scan", ""]
    if df.empty:
        lines.append("No candidate files found.")
    else:
        lines.append(f"- candidate files: {len(df)}")
        lines.append("")
        by_type = df.groupby("file_type").size().sort_values(ascending=False)
        lines.append("## By file type")
        for kind, count in by_type.items():
            lines.append(f"- `{kind}`: {count}")
        lines.append("")
        lines.append("## Largest candidates")
        for _, row in df.sort_values("size", ascending=False).head(20).iterrows():
            lines.append(f"- `{row['path']}` ({row['size']} bytes; {row['matched_keywords']})")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_prediction_jsons(input_roots: Iterable[str | Path]) -> List[Path]:
    paths: List[Path] = []
    for root in input_roots:
        p = Path(root)
        if not p.exists():
            continue
        for item in p.rglob("predictions/*.json"):
            paths.append(item)
    return sorted(set(paths))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
