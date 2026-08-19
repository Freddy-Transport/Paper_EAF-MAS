import csv
import json
import os
import tempfile
import unittest
from pathlib import Path


class FullAutoSkillSkillBenchTests(unittest.TestCase):
    def _write_traffic(self, root: Path, n_rows: int = 18) -> Path:
        path = root / "traffic.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "station_a", "station_b"])
            for i in range(n_rows):
                writer.writerow([f"2023-01-01 {i:02d}:00:00", 10 + i, 20 + i])
        return path

    def _write_events(self, root: Path) -> Path:
        path = root / "events.json"
        events = [
            {
                "title": "Train window plaza event",
                "event_time": "2023-01-01 02:00:00",
                "event_type": "Plaza Partner Event",
                "impact_tier": "C",
                "station_rank": 40,
            },
            {
                "title": "Validation sports event",
                "event_time": "2023-01-01 07:00:00",
                "event_type": "Sport-Adult",
                "impact_tier": "B",
                "station_rank": 20,
            },
            {
                "title": "Test event must not enter memory",
                "event_time": "2023-01-01 14:00:00",
                "event_type": "Street Festival",
                "impact_tier": "A",
                "station_rank": 10,
            },
        ]
        path.write_text(json.dumps(events), encoding="utf-8")
        return path

    def _write_residual_kb(self, root: Path) -> Path:
        path = root / "residual.json"
        docs = [
            {
                "text": "plaza residual",
                "event_type": "Plaza Partner Event",
                "impact_tier": "C",
                "rank_group": "top64",
                "day_type": "weekend",
                "median_correction": 0.01,
                "iqr": 0.02,
                "n_eff": 5,
                "split": "train",
            },
            {
                "text": "sports residual",
                "event_type": "Sport-Adult",
                "impact_tier": "B",
                "rank_group": "top32",
                "day_type": "weekend",
                "median_correction": -0.02,
                "iqr": 0.03,
                "n_eff": 6,
                "split": "train",
            },
        ]
        path.write_text(json.dumps(docs), encoding="utf-8")
        return path

    def test_full_hourly_anchor_builder_counts_and_no_cross_split(self):
        from experiments.run_full_autoskill_skillbench import build_full_hourly_anchor_plan

        with tempfile.TemporaryDirectory() as td:
            traffic = self._write_traffic(Path(td), n_rows=18)
            val, test = build_full_hourly_anchor_plan(
                traffic,
                horizon=3,
                train_rows=5,
                val_rows=6,
            )

        self.assertEqual(len(val), 4)
        self.assertEqual(len(test), 5)
        self.assertEqual(val[0]["row_idx"], 5)
        self.assertEqual(val[-1]["row_idx"], 8)
        self.assertEqual(test[0]["row_idx"], 11)
        self.assertEqual(test[-1]["row_idx"], 15)
        self.assertTrue(all(a["split"] == "val" for a in val))
        self.assertTrue(all(a["split"] == "test" for a in test))

    def test_memory_layers_exclude_future_splits(self):
        from experiments.run_autoskill_effectiveness_study import _load_events, _load_residual_docs
        from experiments.run_full_autoskill_skillbench import build_residual_memory_layers

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            events = _load_events(self._write_events(root))
            residual = _load_residual_docs(self._write_residual_kb(root))
            layers = build_residual_memory_layers(
                events,
                residual,
                train_start="2023-01-01 00:00:00",
                train_end="2023-01-01 04:00:00",
                val_end="2023-01-01 10:00:00",
            )

        train_types = {doc["event_type"] for doc in layers["memory_train_v2"]}
        train_val_types = {doc["event_type"] for doc in layers["memory_train_val_v2"]}
        self.assertIn("Plaza Partner Event", train_types)
        self.assertNotIn("Sport-Adult", train_types)
        self.assertIn("Sport-Adult", train_val_types)
        self.assertNotIn("Street Festival", train_val_types)
        self.assertTrue(all(doc["split"] in {"train", "train_memory"} for doc in layers["memory_train_v2"]))
        self.assertTrue(all(doc["split"] != "test" for doc in layers["memory_train_val_v2"]))

    def test_qwenplus_live_requires_runtime_env(self):
        from experiments.run_full_autoskill_skillbench import require_qwenplus_env

        old = {k: os.environ.pop(k, None) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL")}
        try:
            with self.assertRaisesRegex(RuntimeError, "Qwen-Plus live"):
                require_qwenplus_env(enabled=True)
        finally:
            for key, value in old.items():
                if value is not None:
                    os.environ[key] = value

    def test_event_window_index_matches_horizon_filter(self):
        from experiments.run_full_autoskill_skillbench import EventWindowIndex

        events = [
            {"title": "before", "event_time": "2023-01-01 02:00:00", "event_type": "A", "impact_tier": "A"},
            {"title": "inside", "event_time": "2023-01-01 05:00:00", "event_type": "B", "impact_tier": "B"},
            {"title": "edge_out", "event_time": "2023-01-01 08:00:00", "event_type": "C", "impact_tier": "C"},
        ]
        index = EventWindowIndex(events)

        rows = index.events_for_anchor("2023-01-01 04:00:00", horizon=4)

        self.assertEqual([row["title"] for row in rows], ["inside"])

    def test_small_skillbench_outputs_six_modes_and_keeps_forecast_arrays_identical(self):
        from experiments.run_full_autoskill_skillbench import SKILLBENCH_MODES, run_full_autoskill_skillbench

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary = run_full_autoskill_skillbench(
                output_root=root / "out",
                traffic_csv=self._write_traffic(root, n_rows=18),
                events_json=self._write_events(root),
                residual_kb_preview=self._write_residual_kb(root),
                horizon=3,
                train_rows=5,
                val_rows=6,
                expected_val_anchors=4,
                expected_test_anchors=5,
                enable_qwenplus_live=False,
                enable_local_vllm_explanations=False,
            )

            self.assertEqual(summary["val_anchor_count"], 4)
            self.assertEqual(summary["test_anchor_count"], 5)
            self.assertEqual(summary["modes"], SKILLBENCH_MODES)
            self.assertTrue(summary["forecast_arrays_identical"])
            self.assertTrue(summary["skill_lifecycle"]["test_read_only"])
            self.assertEqual(summary["skill_lifecycle"]["test_promoted_count"], 0)
            self.assertGreaterEqual(summary["mean_quality_by_mode"]["full_autoskill_memory"]["groundedness"], summary["mean_quality_by_mode"]["no_skill"]["groundedness"])
            self.assertLessEqual(summary["mean_quality_by_mode"]["full_autoskill_memory"]["unsupported_claim_rate"], summary["mean_quality_by_mode"]["no_skill"]["unsupported_claim_rate"])
            self.assertEqual(summary["mean_quality_by_mode"]["full_autoskill_memory"]["leakage_free_rate"], 1.0)
            self.assertGreater(summary["residual_memory_organization"]["delta"], 0.0)
            self.assertTrue((root / "out" / "figures" / "fig_explanation_quality_delta_forest_ci.png").is_file())
            self.assertTrue((root / "out" / "tables" / "explanation_quality_ablation.tex").is_file())

    def test_abstention_stress_rows_include_conflict_diversity(self):
        from experiments.run_full_autoskill_skillbench import build_abstention_stress_rows

        def payload(date, source, residual, has_event=True, allowed=False, severe=None):
            return {
                "request": {"date": date},
                "evidence": {
                    "has_major_event": has_event,
                    "evidence_audit": {
                        "source_validity_score": source,
                        "geo_consistency_score": 0.30 if severe == "geo_conflict" else 0.80,
                        "temporal_alignment_score": 0.30 if severe == "temporal_conflict" else 0.90,
                        "residual_support_score": residual,
                        "severe_conflict_flags": [severe] if severe else [],
                    },
                },
                "decision": {"controller_allowed": allowed, "abstain": not allowed},
            }

        rows = build_abstention_stress_rows(
            [
                payload("weak-source", 0.20, 0.80, allowed=True),
                payload("weak-residual", 0.90, 0.10),
                payload("no-event", 1.00, 0.80, has_event=False),
                payload("geo-conflict", 0.90, 0.80, severe="geo_conflict"),
                payload("temporal-conflict", 0.90, 0.80, severe="temporal_conflict"),
            ],
            active_skills=[{"skill_category": "abstention_skill"}],
        )

        stress_types = {row["stress_type"] for row in rows}
        self.assertGreaterEqual(len(stress_types), 5)
        self.assertTrue(any(row["weak_source"] for row in rows))
        self.assertTrue(any(row["weak_residual"] for row in rows))
        self.assertTrue(any(row["no_major_event"] for row in rows))
        self.assertTrue(any(row["severe_conflict"] for row in rows))
        self.assertFalse(all(row["skill_abstention_correct"] == 1.0 for row in rows))

    def test_judge_prompt_does_not_include_posthoc_metrics(self):
        from experiments.run_full_autoskill_skillbench import build_skillbench_judge_prompt

        prompt = build_skillbench_judge_prompt(
            "A forecast-time explanation",
            "A skill-guided forecast-time explanation",
            posthoc_metrics={"actual": 100, "wape": 1.2},
        )

        lowered = prompt.lower()
        self.assertNotIn("actual", lowered)
        self.assertNotIn("wape", lowered)
        self.assertIn("forecast-time", lowered)


if __name__ == "__main__":
    unittest.main()
