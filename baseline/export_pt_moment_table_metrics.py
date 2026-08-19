"""Export available PT-MOMENT values for paper Table 5 and Table 6.

This utility reads existing paper artifacts only. It does not run forecasts or
train baselines. Missing fields are left as ``To be filled`` so the manuscript
does not invent unavailable MAE/RMSE/sMAPE or official-baseline metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict


def read_backbone_comparison(path: str | Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        model_name = row.get("", row.get("model", ""))
        if model_name == "pt_moment":
            return {
                "top128_wape": f"{float(row['top128']):.3f}",
                "event_venue_wape": f"{float(row['event_venue28']):.3f}",
                "event_active_wape": f"{float(row['event_active']):.3f}",
            }
    return {}


def read_horizon_wape(path: str | Path, mode: str = "pt_moment") -> Dict[str, str]:
    by_idx: Dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("mode") == mode:
                by_idx[int(row["horizon_idx"])] = float(row["wape"])
    return {
        "H24": f"{by_idx[23]:.3f}" if 23 in by_idx else "To be filled",
        "H48": f"{by_idx[47]:.3f}" if 47 in by_idx else "To be filled",
        "H96": f"{by_idx[95]:.3f}" if 95 in by_idx else "To be filled",
        "H192": f"{by_idx[191]:.3f}" if 191 in by_idx else "To be filled",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbone_comparison_csv",
        default="../autotemp/ccfa_fullstack_summary_only_20260619_123041/tables/table_backbone_adapter_comparison.csv",
    )
    p.add_argument(
        "--per_horizon_csv",
        default="../autotemp/ccfa_fullstack_final_20260618_100628/tables/per_horizon_wape.csv",
    )
    p.add_argument("--output", default="pt_moment_table_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = {
        "table5_available_pt_moment_metrics": read_backbone_comparison(args.backbone_comparison_csv),
        "table6_available_pt_moment_horizon_wape": read_horizon_wape(args.per_horizon_csv),
        "notes": "Existing artifact export only; no baseline training or prompt ablation is run.",
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
