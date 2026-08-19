import unittest


class ExplanationWindowSelectorTests(unittest.TestCase):
    def test_selector_returns_diverse_case_types_and_prefers_default_anchor(self):
        from experiments.select_explanation_windows import select_windows

        rows = [
            {
                "date": "2023-06-19 10:00:00",
                "event_type": "Plaza Event",
                "impact_tier": "A",
                "has_major_event": True,
                "raw_wape": 17.6,
                "adjusted_wape": 17.5,
                "wape_delta_raw_minus_adjusted": 0.1,
                "accepted_evidence_count": 0,
                "residual_memory_cases": 2,
                "abstain": False,
                "adjusted_channels": "N060",
            },
            {
                "date": "2023-06-20 10:00:00",
                "event_type": "Concert",
                "impact_tier": "A",
                "has_major_event": True,
                "wape_delta_raw_minus_adjusted": -0.4,
                "accepted_evidence_count": 0,
                "residual_memory_cases": 0,
                "abstain": False,
                "adjusted_channels": "N070",
            },
            {
                "date": "2023-06-21 10:00:00",
                "event_type": "none",
                "has_major_event": False,
                "wape_delta_raw_minus_adjusted": 0.0,
                "accepted_evidence_count": 0,
                "residual_memory_cases": 0,
                "abstain": True,
                "adjusted_channels": "",
            },
            {
                "date": "2023-06-22 10:00:00",
                "event_type": "Street Festival",
                "impact_tier": "B",
                "has_major_event": True,
                "wape_delta_raw_minus_adjusted": 0.0,
                "accepted_evidence_count": 0,
                "residual_memory_cases": 3,
                "abstain": True,
                "adjusted_channels": "",
            },
        ]

        selected = select_windows(rows, max_windows=4, default_anchor="2023-06-19 10:00:00")
        labels = {row["case_type"] for row in selected}

        self.assertEqual(selected[0]["date"], "2023-06-19 10:00:00")
        self.assertIn("event_intervention", labels)
        self.assertIn("weak_evidence_failure", labels)
        self.assertIn("safe_abstention", labels)
        self.assertIn("historical_memory_case", labels)

    def test_selector_does_not_promote_near_zero_delta_as_event_intervention(self):
        from experiments.select_explanation_windows import select_windows

        rows = [
            {
                "date": "2023-05-09 10:00:00",
                "event_type": "Plaza Event",
                "impact_tier": "A",
                "has_major_event": True,
                "raw_wape": 15.939786,
                "adjusted_wape": 15.939784,
                "event_window_wape_gain": 0.000023,
                "adjusted_channel_wape_gain": 0.000010,
                "included_unit_wape_gain": 0.000006,
                "accepted_evidence_count": 1,
                "residual_memory_cases": 2,
                "abstain": False,
                "adjusted_channels": "N060",
            },
            {
                "date": "2023-06-17 10:00:00",
                "event_type": "Concert",
                "impact_tier": "A",
                "has_major_event": True,
                "raw_wape": 18.079610,
                "adjusted_wape": 17.987849,
                "event_window_wape_gain": 0.282633,
                "adjusted_channel_wape_gain": 0.16,
                "included_unit_wape_gain": 0.12,
                "accepted_evidence_count": 1,
                "residual_memory_cases": 2,
                "abstain": False,
                "adjusted_channels": "N070",
            },
        ]

        selected = select_windows(rows, max_windows=1, default_anchor="2023-05-09 10:00:00")

        self.assertEqual(selected[0]["case_type"], "event_intervention")
        self.assertEqual(selected[0]["date"], "2023-06-17 10:00:00")
        self.assertGreaterEqual(float(selected[0]["event_window_wape_gain"]), 0.05)

    def test_rows_from_full_metric_csv_pair_numerical_and_adapter(self):
        import csv
        import tempfile
        from pathlib import Path

        from experiments.select_explanation_windows import _rows_from_csv

        fields = ["split", "anchor", "date", "mode", "scope", "wape", "event_wape", "event_n"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "full_metric_rows.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "test",
                        "anchor": "12032",
                        "date": "2023-06-17 10:00:00",
                        "mode": "numerical_only",
                        "scope": "event_venue28",
                        "wape": "18.079609",
                        "event_wape": "14.970915",
                        "event_n": "1128",
                    }
                )
                writer.writerow(
                    {
                        "split": "test",
                        "anchor": "12032",
                        "date": "2023-06-17 10:00:00",
                        "mode": "event_adapter_frozen_moment",
                        "scope": "event_venue28",
                        "wape": "17.987849",
                        "event_wape": "14.688282",
                        "event_n": "1128",
                    }
                )

            rows = _rows_from_csv(str(path))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "event_adapter_frozen_moment")
        self.assertAlmostEqual(float(rows[0]["wape_delta_raw_minus_adjusted"]), 0.091760, places=5)
        self.assertAlmostEqual(float(rows[0]["event_window_wape_gain"]), 0.282633, places=5)
        self.assertTrue(rows[0]["has_major_event"])


if __name__ == "__main__":
    unittest.main()
