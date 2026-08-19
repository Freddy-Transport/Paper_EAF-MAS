import csv
import json
import os
import tempfile
import unittest
from pathlib import Path


class CcfaFullStackFinalTests(unittest.TestCase):
    def _traffic(self, root: Path, n_rows: int = 18) -> Path:
        path = root / "traffic.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "station_a", "station_b"])
            for i in range(n_rows):
                writer.writerow([f"2023-01-01 {i:02d}:00:00", 10 + i, 20 + i])
        return path

    def _events(self, root: Path) -> Path:
        path = root / "events.json"
        path.write_text(json.dumps({"events": [
            {
                "title": "Times Square Plaza Festival",
                "event_time": "2023-01-01 08:00:00",
                "event_type": "Street Festival",
                "location": "Times Square | 49 St",
                "channel_name": "A013__49_St",
                "station_rank": 20,
            },
            {
                "title": "Routine field setup",
                "event_time": "2023-01-01 09:00:00",
                "event_type": "Special Event",
                "location": "Central Park",
                "channel_name": "N044__81_St",
                "station_rank": 100,
            },
        ]}), encoding="utf-8")
        return path

    def test_normalize_event_infers_tier_and_relevance_without_mutating_source(self):
        from experiments.run_ccfa_fullstack_final import normalize_event_for_fullstack

        raw = {"title": "Times Square Street Festival", "event_type": "Street Festival", "channel_name": "A013__49_St", "station_rank": 42}
        normalized = normalize_event_for_fullstack(raw)

        self.assertEqual(normalized["impact_tier"], "A")
        self.assertGreaterEqual(normalized["relevance_score"], 0.75)
        self.assertEqual(raw.get("impact_tier"), None)

    def test_normalize_event_demotes_testing_site_and_routine_closure(self):
        from experiments.run_ccfa_fullstack_final import normalize_event_for_fullstack

        testing = normalize_event_for_fullstack({"title": "Covid -19 testing site", "event_type": "Special Event", "channel_name": "R533__111_St"})
        closure = normalize_event_for_fullstack({"title": "Routine winter closure", "event_type": "Special Event", "channel_name": "N044__81_St"})
        lawn = normalize_event_for_fullstack({"title": "Lawn Closure East Green", "event_type": "Special Event", "location": "Central Park", "channel_name": "N044__81_St"})

        self.assertNotIn(testing["impact_tier"], {"A", "B"})
        self.assertNotIn(closure["impact_tier"], {"A", "B"})
        self.assertNotIn(lawn["impact_tier"], {"A", "B"})

    def test_adapter_manifest_validation_rejects_fallback_or_head_training(self):
        from experiments.run_ccfa_fullstack_final import validate_real_lp_adapter_manifest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good = root / "good.json"
            good.write_text(json.dumps({
                "real_lp_moment_predictions": True,
                "deterministic_fallback_used": False,
                "moment_head_training": False,
                "prediction_sources": ["moment_predictions_csv"],
            }), encoding="utf-8")
            result = validate_real_lp_adapter_manifest(good)
            self.assertTrue(result["valid"])

            bad = root / "bad.json"
            bad.write_text(json.dumps({
                "real_lp_moment_predictions": False,
                "deterministic_fallback_used": True,
                "moment_head_training": False,
                "prediction_sources": ["deterministic_lag_cache"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "real LP-MOMENT"):
                validate_real_lp_adapter_manifest(bad)

    def test_qwenplus_candidate_gate_excludes_farmers_market_and_youth_sport(self):
        from experiments.run_ccfa_fullstack_final import is_qwenplus_live_candidate, normalize_event_for_fullstack

        market = normalize_event_for_fullstack({"title": "Neighborhood Greenmarket", "event_type": "Farmers Market", "channel_name": "A013__49_St"})
        youth = normalize_event_for_fullstack({"title": "Soccer - Youth", "event_type": "Sport - Youth", "channel_name": "R530__111_St"})
        parade = normalize_event_for_fullstack({"title": "Times Square Parade", "event_type": "Parade", "channel_name": "A013__49_St"})

        self.assertFalse(is_qwenplus_live_candidate(market))
        self.assertFalse(is_qwenplus_live_candidate(youth))
        self.assertTrue(is_qwenplus_live_candidate(parade))

    def test_physical_event_key_deduplicates_same_event_across_station_channels(self):
        from experiments.run_ccfa_fullstack_final import event_physical_key

        a = {"title": "Columbia Greenmarket", "event_time": "2023-01-29 08:00:00", "venue_name": "Broadway between W 113 and W 116", "location": "... | 49 St"}
        b = {"title": "Columbia Greenmarket", "event_time": "2023-01-29 08:00:00", "venue_name": "Broadway between W 113 and W 116", "location": "... | 50 St"}

        self.assertEqual(event_physical_key(a), event_physical_key(b))

    def test_qwenplus_live_plan_requires_runtime_env_and_redacts_secret(self):
        from experiments.run_ccfa_fullstack_final import collect_qwenplus_live_evidence

        old = {k: os.environ.pop(k, None) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL")}
        try:
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(RuntimeError, "Qwen-Plus live"):
                    collect_qwenplus_live_evidence(
                        output_root=Path(td),
                        anchors=[{"date": "2023-01-01 04:00:00"}],
                        events=[{"title": "Times Square Festival", "event_time": "2023-01-01 08:00:00", "event_type": "Street Festival", "channel_name": "A013__49_St"}],
                        horizon=8,
                        enabled=True,
                        max_live_events=1,
                    )
        finally:
            for key, value in old.items():
                if value is not None:
                    os.environ[key] = value

    def test_qwenplus_live_with_fake_client_records_api_call_without_secret(self):
        from experiments.run_ccfa_fullstack_final import collect_qwenplus_live_evidence

        class FakeClient:
            calls = 0
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        FakeClient.calls += 1
                        response = type("Response", (), {})()
                        msg = type("Msg", (), {})()
                        choice = type("Choice", (), {})()
                        msg.content = json.dumps({
                            "event_summary": "Times Square Festival is a public event near 49 St.",
                            "evidence": [{"title": "Times Square Festival", "url": "https://example.org/event", "snippet": "Times Square Festival near 49 St on Jan 1."}],
                        })
                        choice.message = msg
                        response.choices = [choice]
                        return response

        old = {k: os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LLM_MODEL")}
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        os.environ["OPENAI_API_KEY"] = "sk-unit-test-secret"
        os.environ["LLM_MODEL"] = "qwen-plus"
        try:
            with tempfile.TemporaryDirectory() as td:
                summary = collect_qwenplus_live_evidence(
                    output_root=Path(td),
                    anchors=[{"date": "2023-01-01 04:00:00"}],
                    events=[{"title": "Times Square Festival", "event_time": "2023-01-01 08:00:00", "event_type": "Street Festival", "channel_name": "A013__49_St", "station_rank": 10}],
                    horizon=8,
                    enabled=True,
                    max_live_events=1,
                    client_factory=lambda **_: FakeClient(),
                )
                self.assertEqual(summary["stats"]["api_calls"], 1)
                text = "\n".join(p.read_text(encoding="utf-8") for p in Path(td).rglob("*.json"))
                self.assertNotIn("sk-unit-test-secret", text)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_secret_scan_does_not_flag_plain_risk_url_as_api_key(self):
        from experiments.run_ccfa_fullstack_final import _secret_scan_text

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "evidence.json").write_text("https://engineering.nyu.edu/finance-and-risk-engineering", encoding="utf-8")
            result = _secret_scan_text(root)

        self.assertTrue(result["passed"])

    def test_fullstack_smoke_uses_full_anchor_counts_and_writes_manifest(self):
        from experiments.run_ccfa_fullstack_final import run_ccfa_fullstack_final

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = run_ccfa_fullstack_final(
                output_root=root / "out",
                traffic_csv=self._traffic(root, n_rows=18),
                events_json=self._events(root),
                residual_kb_preview=None,
                adapter_manifest=None,
                horizon=3,
                train_rows=5,
                val_rows=6,
                expected_val_anchors=4,
                expected_test_anchors=5,
                run_numeric=False,
                run_skillbench=False,
                enable_qwenplus_live=False,
                enable_local_vllm=False,
            )
            self.assertEqual(manifest["anchors"]["val_count"], 4)
            self.assertEqual(manifest["anchors"]["test_count"], 5)
            self.assertTrue((root / "out" / "reports" / "ccfa_fullstack_manifest.json").is_file())
            self.assertEqual(manifest["claim_boundary"]["moment_head_training"], False)

    def test_ccfa_figure_exporter_writes_paper_evidence_chain(self):
        from experiments.export_ccfa_fullstack_figures import export_figures

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for sub in [
                "predictions",
                "tables",
                "reports",
                "autoskill_skillbench_full/reports",
                "autoskill_skillbench_full/figures",
            ]:
                (root / sub).mkdir(parents=True, exist_ok=True)
            with (root / "predictions" / "full_metric_rows.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "mode",
                        "anchor",
                        "date",
                        "top128_wape",
                        "event_venue28_wape",
                        "event_active_wape",
                        "non_event_wape",
                        "event_active_n",
                    ],
                )
                writer.writeheader()
                for i in range(4):
                    event_n = 10 + i
                    writer.writerow({"mode": "last_week", "anchor": i, "date": f"2023-01-0{i+1} 00:00:00", "top128_wape": 10 + i, "event_venue28_wape": 11 + i, "event_active_wape": 12 + i, "non_event_wape": 13 + i, "event_active_n": event_n})
                    writer.writerow({"mode": "pt_moment", "anchor": i, "date": f"2023-01-0{i+1} 00:00:00", "top128_wape": 9 + i, "event_venue28_wape": 10 + i, "event_active_wape": 11 + i, "non_event_wape": 12 + i, "event_active_n": event_n})
                    writer.writerow({"mode": "event_adapter_frozen", "anchor": i, "date": f"2023-01-0{i+1} 00:00:00", "top128_wape": 8.9 + i, "event_venue28_wape": 9.8 + i, "event_active_wape": 10.5 + i, "non_event_wape": 12 + i, "event_active_n": event_n})
            with (root / "tables" / "per_horizon_wape.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["mode", "horizon_idx", "wape"])
                writer.writeheader()
                for mode in ["last_week", "pt_moment", "event_adapter_frozen"]:
                    for h in range(1, 193):
                        writer.writerow({"mode": mode, "horizon_idx": h, "wape": 10 + h / 100 + (0 if mode == "event_adapter_frozen" else 0.2)})
            (root / "reports" / "qwenplus_live_evidence_summary.json").write_text(json.dumps({
                "unique_high_value_physical_events": 5,
                "stats": {"selected_events": 5, "api_calls": 5, "cache_hits": 0, "cache_misses": 5, "summary_count": 5},
                "events": [{"model_assisted_summary_count": 1, "summary_used_for_explanation": True} for _ in range(5)],
            }), encoding="utf-8")
            (root / "autoskill_skillbench_full" / "reports" / "autoskill_skillbench_summary.json").write_text(json.dumps({
                "val_anchor_count": 8,
                "test_anchor_count": 4,
                "skill_lifecycle": {"val_experience_count": 8, "val_candidate_count": 2, "val_mutation_count": 2, "val_promoted_count": 1, "test_active_count": 1, "test_read_only": True},
                "forecast_arrays_identical": True,
                "mean_quality_by_mode": {"no_skill": {"multi_hop_completeness": 0.5, "unsupported_claim_rate": 0.2}, "full_autoskill_memory": {"multi_hop_completeness": 0.7, "unsupported_claim_rate": 0.1}},
                "full_autoskill_delta_ci": [{"metric": "multi_hop_completeness", "mean_delta": 0.2, "ci_low": 0.1, "ci_high": 0.3}],
                "residual_memory_coverage": {"legacy_hit_at_3": 0.8, "coverage_v2_hit_at_3": 1.0, "legacy_empty_rate": 0.1, "coverage_v2_empty_rate": 0.0},
                "residual_memory_organization": {"baseline_relevance_mean": 0.4, "skill_selected_relevance_mean": 0.7, "delta": 0.3},
                "abstention_quality": {"case_count": 4, "abstention_correctness": 1.0, "unsupported_correction_claim_rate": 0.0},
            }), encoding="utf-8")

            manifest = export_figures(root)

            expected = {
                "figA_full_split_prediction_landscape.png",
                "figB_backbone_adapter_comparison.png",
                "figC_event_active_adapter_gain.png",
                "figD_correction_safety_sparsity.png",
                "figE_model_assisted_summary_coverage.png",
                "figF_autoskill_lifecycle.png",
                "figG_skill_explanation_quality_delta.png",
                "figH_residual_memory_organization.png",
                "figI_safe_abstention_matrix.png",
                "figJ_case_card_pair.png",
            }
            generated = {Path(p).name for p in manifest["generated_figures"]}
            self.assertTrue(expected.issubset(generated))
            self.assertEqual(manifest["n_test_anchors"], 4)
            self.assertIn("Figure G", manifest["paper_placement"])
            self.assertGreater(manifest["event_active_gain_mean"], 0)
            self.assertNotIn("LP-MOMENT", json.dumps(manifest))
            self.assertNotIn("last_week", json.dumps(manifest).lower())
            self.assertNotIn("accepted_evidence", json.dumps(manifest).lower())
            self.assertNotIn("rejected_evidence", json.dumps(manifest).lower())
            baseline_table = (root / "tables" / "table_backbone_adapter_comparison.csv").read_text(encoding="utf-8").lower()
            self.assertNotIn("last_week", baseline_table)
            self.assertTrue((root / "reports" / "figure_provenance_audit.csv").is_file())
            self.assertTrue((root / "reports" / "figure_provenance_audit.json").is_file())
            self.assertTrue((root / "figures_main_final" / "figA_full_split_prediction_landscape.png").is_file())
            self.assertTrue((root / "figures_appendix_final").is_dir())
            final_names = {path.name for path in (root / "figures_main_final").glob("*.png")}
            self.assertNotIn("fig2_dataset_event_station_split.png", final_names)
            self.assertNotIn("figJ_case_card_pair_placeholder.png", final_names)
            provenance = json.loads((root / "reports" / "figure_provenance_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(all("source_type" in row and "paper_action" in row for row in provenance["figures"]))
            self.assertIn("legacy figures are excluded from figures_main_final", provenance["summary"]["policy"])

    def test_full_test_evidence_audit_rows_use_summary_only_events(self):
        from experiments.assemble_paper_figures import build_full_test_evidence_audit_rows

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "reports").mkdir()
            (root / "reports" / "qwenplus_live_evidence_summary.json").write_text(json.dumps({
                "unique_high_value_physical_events": 2,
                "events": [
                    {
                        "physical_event_key": "times square festival|2023-01-01 08:00:00|times square",
                        "title": "Times Square Festival",
                        "event_time": "2023-01-01 08:00:00",
                        "first_anchor_date": "2023-01-01 04:00:00",
                        "impact_tier": "A",
                        "source_type": "model-assisted",
                        "model_assisted_summary_count": 1,
                        "summary_used_for_explanation": True,
                    },
                    {
                        "physical_event_key": "garden concert|2023-01-03 20:00:00|madison square garden",
                        "title": "Garden Concert",
                        "event_time": "2023-01-03 20:00:00",
                        "first_anchor_date": "2023-01-03 10:00:00",
                        "impact_tier": "B",
                        "source_type": "model-assisted",
                        "model_assisted_summary_count": 1,
                        "summary_used_for_explanation": True,
                    },
                ],
            }), encoding="utf-8")
            events_json = root / "events.json"
            events_json.write_text(json.dumps({"events": [
                {
                    "title": "Times Square Festival",
                    "event_time": "2023-01-01 08:00:00",
                    "event_type": "Street Festival",
                    "source": "nyc_open_data_permitted_events",
                    "channel_name": "A013__49_St",
                    "station_rank": 20,
                    "distance_m": 240.0,
                }
            ]}), encoding="utf-8")
            memory = root / "memory.json"
            memory.write_text(json.dumps([
                {
                    "event_type": "Street Festival",
                    "impact_tier": "A",
                    "rank_group": "top32",
                    "day_type": "weekend",
                    "n_eff": 40,
                    "median_correction": 0.03,
                }
            ]), encoding="utf-8")

            rows = build_full_test_evidence_audit_rows(root, events_json, memory, horizon_hours=192)

            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["inside_horizon"])
            self.assertGreater(rows[0]["geo_consistency_score"], rows[1]["geo_consistency_score"])
            self.assertGreater(rows[0]["residual_support_score"], rows[1]["residual_support_score"])
            self.assertIn("structured_event_match_missing", rows[1]["conflict_flags"])
            self.assertNotIn("accepted", json.dumps(rows).lower())
            self.assertNotIn("citation", json.dumps(rows).lower())

    def test_full_test_evidence_audit_flags_out_of_horizon_event(self):
        from experiments.assemble_paper_figures import build_full_test_evidence_audit_rows

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "reports").mkdir()
            (root / "reports" / "qwenplus_live_evidence_summary.json").write_text(json.dumps({
                "unique_high_value_physical_events": 1,
                "events": [{
                    "physical_event_key": "late event|2023-01-20 08:00:00|times square",
                    "title": "Late Event",
                    "event_time": "2023-01-20 08:00:00",
                    "first_anchor_date": "2023-01-01 04:00:00",
                    "impact_tier": "A",
                    "model_assisted_summary_count": 1,
                    "summary_used_for_explanation": True,
                }],
            }), encoding="utf-8")

            rows = build_full_test_evidence_audit_rows(root, events_json_path=root / "missing.json", residual_memory_path=root / "missing_memory.json", horizon_hours=192)

            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["inside_horizon"])
            self.assertEqual(rows[0]["temporal_alignment_score"], 0.2)
            self.assertIn("outside_forecast_horizon", rows[0]["conflict_flags"])


if __name__ == "__main__":
    unittest.main()
