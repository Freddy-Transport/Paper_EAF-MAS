import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class GateCoverageDiagnosticsTests(unittest.TestCase):
    def _load_trace(self, root: Path) -> pd.DataFrame:
        parquet_path = root / "predictions" / "gate_trace_rows.parquet"
        csv_path = root / "predictions" / "gate_trace_rows.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return pd.read_csv(csv_path)

    def test_synthetic_smoke_writes_required_outputs(self):
        from experiments.run_gate_coverage_diagnostics import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, trace_format="parquet", seed=7)

            required = [
                "predictions/gate_trace_rows.parquet",
                "tables/gate_funnel_summary.csv",
                "tables/gate_funnel_summary.tex",
                "tables/gate_reason_code_summary.csv",
                "tables/gate_reason_code_summary.tex",
                "tables/gate_by_event_tier_summary.csv",
                "tables/gate_by_event_tier_summary.tex",
                "figures/gate_coverage_funnel.png",
                "figures/gate_coverage_funnel.pdf",
                "figures/gate_reason_code_bar.png",
                "figures/gate_reason_code_bar.pdf",
                "figures/gate_by_tier_heatmap.png",
                "figures/gate_by_tier_heatmap.pdf",
                "reports/gate_coverage_manifest.json",
                "reports/gate_coverage_interpretation.md",
            ]
            for rel in required:
                self.assertTrue((root / rel).is_file(), rel)

    def test_funnel_counts_are_monotonic_and_corrections_are_bounded_by_allowed_path(self):
        from experiments.run_gate_coverage_diagnostics import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, trace_format="parquet", seed=11)
            funnel = pd.read_csv(root / "tables" / "gate_funnel_summary.csv").set_index("stage")

            stages = [
                "all_cells",
                "event_candidate_cells",
                "station_channel_matched_cells",
                "residual_supported_cells",
                "audit_passed_cells",
                "controller_allowed_cells",
                "corrected_cells",
                "clipped_cells",
            ]
            counts = [int(funnel.loc[stage, "count"]) for stage in stages]
            self.assertEqual(counts, sorted(counts, reverse=True))
            self.assertLessEqual(
                int(funnel.loc["corrected_cells", "count"]),
                int(funnel.loc["controller_allowed_cells", "count"]),
            )
            self.assertLessEqual(
                int(funnel.loc["clipped_cells", "count"]),
                int(funnel.loc["corrected_cells", "count"]),
            )

    def test_trace_invariants_and_reason_codes(self):
        from experiments.run_gate_coverage_diagnostics import run_synthetic_smoke

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_synthetic_smoke(root, trace_format="parquet", seed=13)
            trace = self._load_trace(root)
            manifest = json.loads((root / "reports" / "gate_coverage_manifest.json").read_text())

            self.assertIn("non_event_correction_ratio", manifest)
            self.assertLessEqual(float(manifest["non_event_correction_ratio"]), 1e-9)
            self.assertFalse(trace["forecast_array_changed_by_skill"].any())
            self.assertFalse(trace["reason_code"].isna().any())
            self.assertTrue((trace["reason_code"].astype(str).str.len() > 0).all())

    def test_default_trace_format_requires_parquet_and_csv_fallback_is_explicit(self):
        from experiments.run_gate_coverage_diagnostics import parse_args

        default_args = parse_args(["--synthetic_smoke"])
        self.assertEqual("parquet", default_args.trace_format)
        csv_args = parse_args(["--synthetic_smoke", "--trace_format", "csv"])
        self.assertEqual("csv", csv_args.trace_format)


if __name__ == "__main__":
    unittest.main()
