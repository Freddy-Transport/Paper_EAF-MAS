"""Shared plotting and manifest utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass
class FigureManifest:
    rows: List[dict] = field(default_factory=list)

    def add(
        self,
        figure_id: str,
        title: str,
        input_files: Iterable[str],
        output_pdf: str,
        output_png: str,
        required_fields: Iterable[str],
        status: str,
        notes: str = "",
    ) -> None:
        self.rows.append(
            {
                "figure_id": figure_id,
                "title": title,
                "input_files": ";".join(str(x) for x in input_files),
                "output_pdf": str(output_pdf or ""),
                "output_png": str(output_png or ""),
                "required_fields": ";".join(str(x) for x in required_fields),
                "status": status,
                "notes": notes,
            }
        )

    def extend(self, other: "FigureManifest") -> None:
        self.rows.extend(other.rows)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "figure_id",
            "title",
            "input_files",
            "output_pdf",
            "output_png",
            "required_fields",
            "status",
            "notes",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


def ensure_output_dirs(output_root: str | Path) -> dict:
    root = Path(output_root)
    dirs = {
        "root": root,
        "processed": root / "processed_viz_data",
        "figures_main": root / "figures_main",
        "figures_appendix": root / "figures_appendix",
        "tables": root / "tables",
        "reports": root / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig, out_base: str | Path, dpi: int = 300) -> tuple[Path, Path]:
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    print(f"saved {pdf}")
    print(f"saved {png}")
    return pdf, png


def missing_note(path: str | Path, title: str, reason: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{reason}\n", encoding="utf-8")
