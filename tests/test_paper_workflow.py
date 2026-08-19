import tempfile
import unittest
import json
import argparse
from pathlib import Path


class PaperWorkflowTests(unittest.TestCase):
    def test_times_square_display_name_is_paper_readable(self):
        from agents.paper_workflow import display_station_name

        raw = "N060__Times_Sq_42_St_N_Q_R_W_S_1_2_3_7_42_St_A_C_E_Bryant_Pk_B_D_F_M_5_A"
        compact = display_station_name(raw)
        self.assertEqual(compact, "Times Sq-42 St / 42 St / Bryant Pk / 5 Av")
        self.assertNotIn("N060__", compact)
        self.assertLessEqual(len(compact), 72)

    def test_qwenplus_live_policy_rejects_cache_only_or_missing_key(self):
        from experiments.run_paper_event_forecasting import validate_qwenplus_live_policy
        from unittest.mock import patch

        args = argparse.Namespace(
            require_qwenplus_live=True,
            enable_qwenplus_search=True,
            qwenplus_cache_only=True,
        )
        with self.assertRaisesRegex(RuntimeError, "qwenplus_cache_only"):
            validate_qwenplus_live_policy(args)

        args.qwenplus_cache_only = False
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                validate_qwenplus_live_policy(args)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-unit-test-secret"}, clear=True):
            validate_qwenplus_live_policy(args)

    def test_local_forecast_metric_groups_are_non_empty(self):
        from experiments.run_paper_event_forecasting import compute_local_forecast_metrics

        raw = [[100.0, 100.0, 100.0, 100.0], [50.0, 50.0, 50.0, 50.0]]
        adjusted = [[100.0, 110.0, 110.0, 100.0], [50.0, 50.0, 50.0, 50.0]]
        actual = [[100.0, 112.0, 112.0, 100.0], [50.0, 50.0, 50.0, 50.0]]
        out = compute_local_forecast_metrics(
            actual,
            raw,
            adjusted,
            ["N060__Times_Sq_42_St", "A013__49_St"],
            [
                "2023-06-19 10:00:00",
                "2023-06-19 11:00:00",
                "2023-06-19 12:00:00",
                "2023-06-19 13:00:00",
            ],
            [{"title": "Event", "event_time": "2023-06-19 12:00:00"}],
            ["N060__Times_Sq_42_St"],
            [{"station_channel": "N060__Times_Sq_42_St"}],
            controller_allowed=True,
        )

        self.assertEqual(out["event_window_metrics"]["status"], "ok")
        self.assertEqual(out["adjusted_channel_metrics"]["status"], "ok")
        self.assertEqual(out["included_unit_metrics"]["status"], "ok")
        self.assertGreater(out["event_window_metrics"]["delta_raw_minus_adjusted"]["wape"], 0)

        abstain = compute_local_forecast_metrics(
            actual,
            raw,
            raw,
            ["N060__Times_Sq_42_St", "A013__49_St"],
            ["2023-06-19 10:00:00"] * 4,
            [{"title": "Event", "event_time": "2023-06-19 12:00:00"}],
            [],
            [],
            controller_allowed=False,
        )
        self.assertEqual(abstain["adjusted_channel_metrics"]["status"], "not_applicable_abstained")

    def test_focused_station_query_resolves_venue_and_builds_short_window(self):
        from experiments.run_paper_event_forecasting import (
            build_focused_prediction,
            filter_events_for_focus_window,
            focused_context_for_llm,
            resolve_focus_channels,
            resolve_focus_time_window,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            channel_map = root / "channel_map.json"
            venue_map = root / "venue_station_map.json"
            channel_map.write_text(
                json.dumps(
                    {
                        "channels": [
                            {
                                "rank": 1,
                                "channel_name": "N060__Times_Sq_42_St",
                                "station_complex_id": "N060",
                                "station_complex": "Times Sq-42 St",
                                "station_name_clean": "Times Sq-42 St",
                            },
                            {
                                "rank": 2,
                                "channel_name": "N070__34_St_Penn_Station",
                                "station_complex_id": "N070",
                                "station_complex": "34 St-Penn Station",
                                "station_name_clean": "34 St-Penn Station",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            venue_map.write_text(
                json.dumps(
                    {
                        "venues": [
                            {
                                "id": "msg",
                                "aliases": ["madison square garden", "msg"],
                                "station_complex_ids": ["N070"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            channels = resolve_focus_channels(
                ["N060__Times_Sq_42_St", "N070__34_St_Penn_Station"],
                channel_map,
                venue_map,
                focus_channels=[],
                station_queries=["Madison Square Garden"],
                fallback_channels=[],
            )
            self.assertEqual([row["channel_name"] for row in channels], ["N070__34_St_Penn_Station"])
            self.assertEqual(channels[0]["matched_by"], "venue_alias")

            timestamps = [
                "2023-05-04 17:00:00",
                "2023-05-04 18:00:00",
                "2023-05-04 19:00:00",
                "2023-05-04 20:00:00",
                "2023-05-04 21:00:00",
            ]
            window = resolve_focus_time_window(
                timestamps,
                focus_start="",
                focus_end="",
                focus_center="2023-05-04 19:00:00",
                hours_before=1,
                hours_after=1,
                structured_events=[],
                fallback_anchor="2023-05-04 17:00:00",
            )
            focused = build_focused_prediction(
                raw_forecast=[[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
                adjusted_forecast=[[10, 11, 12, 13, 14], [20, 22, 23, 24, 24]],
                ground_truth=[[9, 10, 12, 15, 16], [20, 21, 24, 25, 24]],
                channel_names=["N060__Times_Sq_42_St", "N070__34_St_Penn_Station"],
                timestamps=timestamps,
                resolved_channels=channels,
                focus_window=window,
                controller_allowed=True,
            )

            self.assertEqual(len(focused["focused_forecast_rows"]), 3)
            self.assertEqual(focused["focused_forecast_rows"][0]["timestamp"], "2023-05-04 18:00:00")
            self.assertEqual(focused["focused_forecast_metrics"]["status"], "ok")
            llm_context = focused_context_for_llm(focused)
            self.assertIn("forecast_time_rows", llm_context)
            self.assertNotIn("actual", json.dumps(llm_context))
            self.assertNotIn("wape", json.dumps(llm_context).lower())

            focus_events = filter_events_for_focus_window(
                [
                    {"title": "Before window", "event_time": "2023-05-04 16:00:00", "impact_tier": "A"},
                    {"title": "Inside window", "event_time": "2023-05-04 19:00:00", "impact_tier": "A"},
                    {"title": "After window", "event_time": "2023-05-04 22:00:00", "impact_tier": "A"},
                ],
                window,
            )
            self.assertEqual([row["title"] for row in focus_events], ["Inside window"])

    def test_strict_numerical_agent_refuses_missing_weights(self):
        from agents.numerical_agent import NumericalPredictionAgent

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(FileNotFoundError):
                NumericalPredictionAgent(
                    model_path=str(root / "missing_gca"),
                    lp_model_path=str(root / "missing_lp"),
                    allow_lp_training_fallback=False,
                    device="cpu",
                )

    def test_cached_retrieval_records_disabled_status(self):
        from agents.event_retrieval_agent import EventRetrievalAgent

        with tempfile.TemporaryDirectory() as td:
            retriever = EventRetrievalAgent(Path(td), enabled=False)
            rows = retriever.search("NYC events test", max_results=2)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].status, "disabled")
            self.assertTrue(Path(rows[0].cache_path).is_file())

    def test_event_retrieval_builds_event_scoped_cache(self):
        from agents.event_retrieval_agent import EventRetrievalAgent

        with tempfile.TemporaryDirectory() as td:
            retriever = EventRetrievalAgent(Path(td), enabled=False)
            rows = retriever.retrieve_for_events(
                [{"title": "Concert at Madison Square Garden", "event_time": "2023-05-18 20:00:00", "location": "MSG"}],
                "2023-05-18 10:00:00",
                ["N070__34_St_Penn_Station"],
            )

            self.assertEqual(len(rows), 1)
            self.assertIn("Concert at Madison Square Garden", rows[0].query)
            self.assertTrue(rows[0].event_key)
            self.assertTrue(Path(rows[0].cache_path).is_file())

    def test_llm_explanation_json_contract_and_fallback(self):
        from agents.forecast_explanation_agent import ForecastExplanationAgent

        raw = """{
          "event_summary": "Major event near Times Square",
          "evidence_used": ["source A"],
          "station_event_linking": "Venue is close to N060.",
          "multi_hop_reasoning": "event -> crowd -> station-hour demand -> correction",
          "calibration_rationale": "Bounded correction applied.",
          "uncertainty_and_abstention": "Abstain if evidence weak.",
          "markdown": "## Forecast-time Evidence\\n- source A\\n\\n## Multi-hop Reasoning\\nreason"
        }"""
        parsed = ForecastExplanationAgent._parse(raw)
        self.assertTrue(parsed.parsed)
        self.assertIn("Multi-hop Reasoning", parsed.markdown)

        fallback = ForecastExplanationAgent._fallback(
            {
                "structured_events": [{"title": "Event"}],
                "retrieved_sources": [{"title": "cached"}],
                "calibration_decision": {"reason": "adapter applied"},
            },
            "unit test",
        )
        self.assertFalse(fallback.parsed)
        self.assertIn("Calibration rationale", fallback.markdown)

    def test_llm_explanation_enforces_abstain_consistency(self):
        from agents.forecast_explanation_agent import ForecastExplanationAgent

        raw = """{
          "event_summary": "Major event near Herald Square",
          "evidence_used": ["local residual case"],
          "station_event_linking": "Venue is close to nearby stations.",
          "multi_hop_reasoning": "event -> crowd -> demand",
          "calibration_rationale": "The system adjusted the forecast by 0.5%.",
          "uncertainty_and_abstention": "low uncertainty",
          "markdown": "## Forecast-time Evidence\\n- local residual case\\n\\n## Calibration Decision\\nThe system adjusted the forecast by 0.5%.\\n\\n## Uncertainty\\nlow"
        }"""
        parsed = ForecastExplanationAgent._parse(raw)
        fixed = ForecastExplanationAgent._enforce_decision_consistency(
            parsed,
            {"abstain": True, "adjusted_channels": [], "reason": "rag_explain leaves forecast unchanged"},
        )

        self.assertIn("No numerical event calibration was applied", fixed.markdown)
        self.assertIn("final forecast remains the raw PT-MOMENT", fixed.calibration_rationale)
        self.assertNotIn("adjusted the forecast by 0.5%", fixed.markdown)

    def test_qwen3_request_disables_thinking_when_supported(self):
        from agents.forecast_explanation_agent import ForecastExplanationAgent

        qwen3_agent = ForecastExplanationAgent(model="Qwen/Qwen3-8B")
        other_agent = ForecastExplanationAgent(model="Qwen/Qwen2.5-7B-Instruct")

        self.assertEqual(
            qwen3_agent._completion_extra_body(),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertGreaterEqual(qwen3_agent._completion_max_tokens(), 1800)
        self.assertIsNone(other_agent._completion_extra_body())
        self.assertEqual(other_agent._completion_max_tokens(), 1400)

    def test_paper_workflow_writes_core_artifacts(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            ForecastResultSpec,
            NumericalForecastSpec,
            build_explanation_markdown,
            compute_metrics,
            write_forecast_artifacts,
        )

        with tempfile.TemporaryDirectory() as td:
            request = ForecastRequestSpec(
                date="2023-05-06",
                horizon=3,
                station_scope="event_venue28",
                mode="numerical_only",
            )
            numerical = NumericalForecastSpec(
                raw_forecast=[[10.0, 12.0, 11.0]],
                channel_names=["N060__Times_Sq_42_St"],
                timestamps=["2023-05-06 00:00:00", "2023-05-06 01:00:00", "2023-05-06 02:00:00"],
                ground_truth=[[11.0, 12.0, 10.0]],
                model_type="Linear Probing (cached)",
            )
            evidence = EventEvidenceSpec(
                sources=[{"title": "cached source", "url": "https://example.com", "snippet": "evidence"}],
                structured_events=[],
                has_major_event=False,
            )
            decision = CalibrationDecisionSpec(mode="numerical_only", abstain=True, reason="baseline")
            metrics = compute_metrics(numerical.ground_truth, numerical.raw_forecast)
            explanation = build_explanation_markdown(
                request,
                evidence,
                decision,
                metrics,
                "baseline",
                llm_markdown="## Forecast-time Evidence\n- evidence\n\n## Multi-hop Reasoning\nreason",
            )
            result = ForecastResultSpec(
                request=request,
                numerical=numerical,
                evidence=evidence,
                decision=decision,
                adjusted_forecast=numerical.raw_forecast,
                explanation_markdown=explanation,
                metrics=metrics,
            )

            write_forecast_artifacts(result, td, "smoke")

            self.assertTrue((Path(td) / "predictions" / "smoke.json").is_file())
            self.assertTrue((Path(td) / "explanations" / "smoke.md").is_file())
            self.assertTrue((Path(td) / "figures" / "smoke_hourly_curves.png").is_file())
            self.assertIn("wape", metrics)
            self.assertIn("Post-hoc Metrics", explanation)

    def test_focused_prediction_artifacts_and_markdown_are_written(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            ForecastResultSpec,
            NumericalForecastSpec,
            build_explanation_markdown,
            write_forecast_artifacts,
        )

        focused = {
            "focused_request": {"include_focused_prediction": True},
            "focused_resolved_channels": [
                {"channel_name": "N070__34_St_Penn_Station", "matched_by": "venue_alias"}
            ],
            "focused_time_window": {
                "start": "2023-05-04 15:00:00",
                "center": "2023-05-04 19:00:00",
                "end": "2023-05-04 23:00:00",
            },
            "focused_forecast_rows": [
                {
                    "timestamp": "2023-05-04 19:00:00",
                    "channel_name": "N070__34_St_Penn_Station",
                    "raw_forecast": 100.0,
                    "adjusted_forecast": 104.0,
                    "correction": 4.0,
                    "controller_allowed": True,
                    "event_relevance_reason": "venue query maps to station complex N070",
                    "actual": 106.0,
                    "raw_error": 6.0,
                    "adjusted_error": 2.0,
                }
            ],
            "focused_forecast_metrics": {
                "status": "ok",
                "raw": {"wape": 5.0, "mae": 6.0},
                "adjusted": {"wape": 2.0, "mae": 2.0},
            },
        }
        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-05-04 10:00:00", 192, "event_venue28", "event_adapter_frozen_moment"),
            EventEvidenceSpec(has_major_event=True),
            CalibrationDecisionSpec(mode="event_adapter_frozen_moment", controller_allowed=True),
            metrics={"wape": 10.0},
            model_explanation="focused case",
            focused_prediction=focused,
        )
        self.assertIn("## 8. Explanation", markdown)
        self.assertIn("### Factual evidence", markdown)
        self.assertIn("### Risk control", markdown)
        self.assertNotIn("## 8. Operational Answer for the Focus Window", markdown)
        self.assertNotIn("### Focused correction summary", markdown)
        self.assertNotIn("### Operator question", markdown)
        self.assertNotIn("### Local ridership outlook", markdown)
        self.assertNotIn("### Post-hoc local evaluation", markdown)
        self.assertNotIn("| time | station | LP-MOMENT raw | event-aware adjusted | correction | decision |", markdown)
        self.assertNotIn("traffic operator asked for", markdown.lower())

        with tempfile.TemporaryDirectory() as td:
            result = ForecastResultSpec(
                request=ForecastRequestSpec("2023-05-04 10:00:00", 192, "event_venue28", "event_adapter_frozen_moment"),
                numerical=NumericalForecastSpec(
                    raw_forecast=[[100.0]],
                    channel_names=["N070__34_St_Penn_Station"],
                    timestamps=["2023-05-04 19:00:00"],
                    ground_truth=[[106.0]],
                ),
                evidence=EventEvidenceSpec(has_major_event=True),
                decision=CalibrationDecisionSpec(mode="event_adapter_frozen_moment", controller_allowed=True),
                adjusted_forecast=[[104.0]],
                explanation_markdown=markdown,
                focused_prediction=focused,
            )
            write_forecast_artifacts(result, td, "focused")
            self.assertTrue((Path(td) / "tables" / "focused_focused_forecast_rows.csv").is_file())
            self.assertTrue((Path(td) / "tables" / "focused_focused_forecast_rows.json").is_file())
            self.assertTrue((Path(td) / "figures" / "focused_focused_station_time_forecast.png").is_file())
            table_json = json.loads((Path(td) / "tables" / "focused_focused_forecast_rows.json").read_text())
            self.assertEqual(len(table_json["focused_forecast_rows"]), 1)

    def test_event_evidence_uses_summary_only_without_citation_sections(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
        )

        request = ForecastRequestSpec(
            date="2023-06-19 10:00:00",
            horizon=192,
            station_scope="event_venue28",
            mode="rag_explain",
        )
        evidence = EventEvidenceSpec(
            sources=[
                {
                    "title": "Weak TSQ LIVE source",
                    "url": "",
                    "snippet": "A broad Times Square summer concerts page without the exact date.",
                    "accepted": True,
                    "status": "source_time_unknown",
                    "source_agent": "qwen_plus_native_search",
                }
            ],
            rejected_sources=[
                {
                    "title": "Unrelated download site",
                    "url": "https://download.example/bad",
                    "snippet": "software mirror",
                    "accepted": False,
                    "rejected_reason": "low_value_domain",
                }
            ],
            structured_events=[
                {
                    "title": "TSQ LIVE: 45/46 Plaza Programming",
                    "event_time": "2023-06-20 17:00:00",
                    "impact_tier": "A",
                    "channel_name": "N060__Times_Sq_42_St",
                }
            ],
            model_assisted_summaries=[
                {
                    "summary": "Qwen-Plus summarized the TSQ LIVE plaza programming as a free open-air Times Square event.",
                    "source_agent": "qwen_plus",
                }
            ],
            has_major_event=True,
        )

        markdown = build_explanation_markdown(
            request,
            evidence,
            CalibrationDecisionSpec(mode="rag_explain", abstain=True, reason="explain only"),
            metrics={},
            model_explanation="explain only",
        )

        self.assertIn("### Forecast Window", markdown)
        self.assertIn("Horizon end: 2023-06-27 10:00:00", markdown)
        self.assertIn("### Model-Assisted Summary (Non-citable)", markdown)
        self.assertIn("Qwen-Plus summarized", markdown)
        self.assertNotIn("### Verified External Evidence", markdown)
        self.assertNotIn("### Source-Assisted Evidence", markdown)
        self.assertNotIn("### Accepted External Evidence", markdown)
        self.assertNotIn("accepted citation", markdown.lower())
        self.assertNotIn("rejected source", markdown.lower())
        self.assertNotIn("Weak TSQ LIVE source", markdown)
        self.assertNotIn("source_time_unknown", markdown)
        self.assertNotIn("Rejected Retrieval Diagnostics", markdown)
        self.assertNotIn("Unrelated download site", markdown)

    def test_explanation_section_is_human_readable_and_leakage_free(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
        )

        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-06-19 10:00:00", 192, "event_venue28", "event_adapter_frozen_moment"),
            EventEvidenceSpec(
                model_assisted_summaries=[{"summary": "Qwen-Plus summary supports the event context."}],
                has_major_event=True,
            ),
            CalibrationDecisionSpec(
                mode="event_adapter_frozen_moment",
                controller_allowed=True,
                correction_bound=0.05,
                adjusted_channels=["N060__Times_Sq_42_St"],
            ),
            metrics={"wape": 10.0},
            raw_metrics={"wape": 11.0},
            model_explanation="Factual evidence: structured event and Qwen-Plus summary.\n\nResidual analogues: train/validation memory supports a small positive correction.\n\nCalibration rationale: correction is bounded.\n\nUncertainty: residual spread remains non-trivial.\n\nRisk control: excluded channels keep PT-MOMENT raw.",
        )

        section8 = markdown.split("## 8. Explanation", 1)[1].split("## Post-hoc Metrics", 1)[0]
        for label in ["Factual evidence", "Residual analogues", "Calibration rationale", "Uncertainty", "Risk control"]:
            self.assertIn(label, section8)
        for forbidden in ["WAPE", " actual ", "ground truth", "post-hoc"]:
            self.assertNotIn(forbidden.lower(), section8.lower())

    def test_rejected_retrieval_diagnostics_are_not_rendered_in_markdown(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
        )

        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-06-19 10:00:00", 192, "event_venue28", "rag_explain"),
            EventEvidenceSpec(
                sources=[
                    {
                        "title": "Bilibili unrelated clip",
                        "url": "https://www.bilibili.com/video/bad",
                        "accepted": False,
                        "rejected_reason": "low_relevance_domain",
                    }
                ],
                rejected_sources=[
                    {
                        "title": "Speed test page",
                        "url": "https://speed.example",
                        "accepted": False,
                        "rejected_reason": "not_event_evidence",
                    }
                ],
                has_major_event=True,
            ),
            CalibrationDecisionSpec(mode="rag_explain", abstain=True),
            metrics={},
            model_explanation="No verified source; rely on structured event memory.",
        )

        self.assertNotIn("Rejected Retrieval Diagnostics", markdown)
        self.assertNotIn("Bilibili unrelated clip", markdown)
        self.assertNotIn("Speed test page", markdown)

    def test_structured_events_are_deduplicated_and_public_events_rank_first(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
        )

        events = [
            {
                "title": "Violife Cheese",
                "event_time": "2023-06-22 08:00:00",
                "impact_tier": "A",
                "location": "SPRING STREET between MERCER STREET and BROADWAY | 49 St",
                "channel_name": "A013__49_St",
            },
            {
                "title": "Violife Cheese",
                "event_time": "2023-06-22 08:00:00",
                "impact_tier": "A",
                "location": "SPRING STREET between MERCER STREET and BROADWAY | Times Sq-42 St",
                "channel_name": "N060__Times_Sq_42_St",
            },
            {
                "title": "TSQ LIVE: 45/46 Plaza Programming",
                "event_time": "2023-06-20 17:00:00",
                "impact_tier": "A",
                "location": "Broadway Pedestrian Plaza Times Square",
                "channel_name": "N060__Times_Sq_42_St",
            },
        ]
        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-06-19", 192, "event_venue28", "rag_explain"),
            EventEvidenceSpec(structured_events=events, has_major_event=True),
            CalibrationDecisionSpec(mode="rag_explain", abstain=True),
            metrics={},
            model_explanation="explain only",
        )

        structured = markdown.split("### Structured Future Events within Horizon", 1)[1].split("### Model-Assisted Summary", 1)[0]
        first_line = next(line for line in structured.splitlines() if line.startswith("- "))
        self.assertIn("TSQ LIVE", first_line)
        self.assertEqual(structured.count("Violife Cheese"), 1)
        self.assertIn("station_count=2", structured)

    def test_historical_event_cases_are_extracted_and_rendered(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
            extract_historical_event_cases,
        )

        context = (
            "[Model Correction RAG] Event type: Special Event | Station rank: top64 | Day: weekday\n"
            "When MOMENT predicts during this event category, actual ridership is typically 0.9% above model prediction "
            "(median correction needed = +0.9%).\n"
            "Example instances: \"2022 Times Square Valentine Heart\" (2022-02-21, N056, bl_resid=-99.3%); "
            "\"Holiday tree lighting\" (2022-12-09, R154, bl_resid=+71.0%)"
        )
        cases = extract_historical_event_cases(context)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["split"], "train_val_memory")
        self.assertIn("Times Square Valentine", cases[0]["historical_event_title"])
        self.assertEqual(cases[0]["residual_direction"], "decrease")

        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-06-19 10:00:00", 192, "event_venue28", "rag_explain"),
            EventEvidenceSpec(historical_event_cases=cases, local_residual_cases=[context], has_major_event=True),
            CalibrationDecisionSpec(mode="rag_explain", abstain=True),
            metrics={},
            model_explanation="explain only",
        )

        self.assertIn("### Historical Event Memory from Train/Validation", markdown)
        self.assertIn("2022 Times Square Valentine Heart", markdown)
        self.assertIn("median LP-MOMENT correction=+0.9%", markdown)

    def test_residual_memory_skill_guidance_is_rendered_when_attached(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
        )

        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-06-19 10:00:00", 192, "event_venue28", "rag_explain"),
            EventEvidenceSpec(
                selected_residual_memory_skills=[
                    {
                        "skill_id": "residual_memory_skill_street_event_a",
                        "skill_category": "residual_memory_skill",
                        "trigger_condition": {"event_type": "Street Event", "impact_tier": "A"},
                        "explanation_action": {
                            "policy": "organize_residual_memory_for_explanation",
                            "select_cases_by": ["event_type", "station_rank", "residual_direction"],
                        },
                        "reason": "Promoted because residual memory improved explanation coverage.",
                    }
                ],
                has_major_event=True,
            ),
            CalibrationDecisionSpec(mode="rag_explain", abstain=True),
            metrics={},
            model_explanation="explain only",
        )

        self.assertIn("### Residual-Memory Skill Guidance", markdown)
        self.assertIn("residual_memory_skill_street_event_a", markdown)
        self.assertIn("organize_residual_memory_for_explanation", markdown)

    def test_adjusted_forecast_section_renders_local_metrics_not_empty_dicts(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
        )

        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-06-19 10:00:00", 192, "event_venue28", "event_adapter_frozen_moment"),
            EventEvidenceSpec(has_major_event=True),
            CalibrationDecisionSpec(
                mode="event_adapter_frozen_moment",
                abstain=False,
                controller_allowed=True,
                correction_bound=0.05,
            ),
            metrics={"wape": 12.0, "mae": 5.0},
            raw_metrics={"wape": 15.0, "mae": 6.0},
            event_window_metrics={
                "status": "ok",
                "raw": {"wape": 20.0, "mae": 8.0},
                "adjusted": {"wape": 16.0, "mae": 6.0},
                "delta_raw_minus_adjusted": {"wape": 4.0, "mae": 2.0},
                "cell_count": 12,
            },
            adjusted_channel_metrics={"status": "not_applicable_abstained"},
            included_unit_metrics={
                "status": "ok",
                "raw": {"wape": 18.0, "mae": 7.0},
                "adjusted": {"wape": 17.0, "mae": 6.5},
                "delta_raw_minus_adjusted": {"wape": 1.0, "mae": 0.5},
                "cell_count": 6,
            },
            model_explanation="bounded correction",
        )

        self.assertIn("Overall 192h Raw WAPE: 15.000", markdown)
        self.assertIn("Overall 192h Adjusted WAPE: 12.000", markdown)
        self.assertIn("Event-window subset: raw WAPE=20.000, adjusted WAPE=16.000", markdown)
        self.assertIn("Adjusted-channel subset: not_applicable_abstained", markdown)
        self.assertIn("Included-unit subset: raw WAPE=18.000, adjusted WAPE=17.000", markdown)
        self.assertNotIn("Event-window WAPE: {}", markdown)
        self.assertNotIn("Adjusted-channel WAPE: {}", markdown)

    def test_tiny_metric_delta_is_not_rendered_as_zero(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            build_explanation_markdown,
        )

        markdown = build_explanation_markdown(
            ForecastRequestSpec("2023-05-09 10:00:00", 192, "event_venue28", "event_adapter_frozen_moment"),
            EventEvidenceSpec(has_major_event=True),
            CalibrationDecisionSpec(mode="event_adapter_frozen_moment", controller_allowed=True),
            metrics={"wape": 15.939783},
            raw_metrics={"wape": 15.939786},
            event_window_metrics={
                "status": "ok",
                "raw": {"wape": 15.939786},
                "adjusted": {"wape": 15.939783},
                "delta_raw_minus_adjusted": {"wape": 0.0000235935},
                "cell_count": 9,
            },
            model_explanation="bounded correction",
        )

        self.assertIn("delta(raw-adjusted) WAPE=<0.001", markdown)
        self.assertNotIn("delta(raw-adjusted) WAPE=0.000", markdown)

    def test_event_time_gate_keeps_future_events_inside_horizon_only(self):
        from experiments.run_paper_event_forecasting import event_within_forecast_window

        event = type("Event", (), {"event_time": "2023-06-22 08:00:00"})()
        late = type("Event", (), {"event_time": "2023-06-30 08:00:00"})()

        self.assertTrue(event_within_forecast_window(event, "2023-06-19 10:00:00", 192))
        self.assertFalse(event_within_forecast_window(late, "2023-06-19 10:00:00", 192))

    def test_residual_cases_are_deduplicated_for_markdown(self):
        from agents.paper_workflow import deduplicate_residual_cases

        cases = [
            "[correction_rag=1 score=1.210 type=Street Festival day=weekday rank=top64 correction=+0.5%]\n"
            "[Model Correction RAG] Event type: Street Festival | Station rank: top64 | Day: weekday\n"
            "Statistics (N=9): p25=-1.8%, median=+0.5%, p75=+0.8%, std=3.6%",
            "[correction_rag=1 score=1.229 type=Street Festival day=weekday rank=top64 correction=+0.5%]\n"
            "[Model Correction RAG] Event type: Street Festival | Station rank: top64 | Day: weekday\n"
            "Statistics (N=9): p25=-1.8%, median=+0.5%, p75=+0.8%, std=3.6%",
            "[correction_rag=1 score=1.157 type=Sidewalk Sale day=weekday rank=top64 correction=+0.4%]\n"
            "[Model Correction RAG] Event type: Sidewalk Sale | Station rank: top64 | Day: weekday\n"
            "Statistics (N=5): p25=-1.0%, median=+0.4%, p75=+0.9%, std=1.2%",
        ]

        deduped = deduplicate_residual_cases(cases)

        self.assertEqual(len(deduped), 2)
        self.assertIn("merged_count=2", deduped[0])
        self.assertIn("Street Festival", deduped[0])
        self.assertIn("Sidewalk Sale", deduped[1])

    def test_residual_case_summary_removes_repeated_baseline_text(self):
        from agents.paper_workflow import deduplicate_residual_cases

        case = (
            "[correction_rag=1 score=1.229 type=Street Festival day=weekday rank=top64 correction=+0.5%]\n"
            "Statistics (N=9): p25=-1.8%, median=+0.5%, p75=+0.8%, std=3.6% "
            "Baseline excess (actual vs weekly pattern): +5.1% — note: model already captures most of this. "
            "Baseline excess (actual vs weekly pattern): +5.1% — note: model already captures most of this."
        )

        summary = deduplicate_residual_cases([case])[0]

        self.assertEqual(summary.count("Baseline excess"), 1)
        self.assertEqual(summary.count("Statistics"), 1)

    def test_plot_raw_adjusted_overlap_is_annotated(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            ForecastResultSpec,
            NumericalForecastSpec,
            write_forecast_artifacts,
        )

        with tempfile.TemporaryDirectory() as td:
            result = ForecastResultSpec(
                request=ForecastRequestSpec(date="2023-05-18 10:00:00", horizon=3, station_scope="event_venue28", mode="rag_explain"),
                numerical=NumericalForecastSpec(
                    raw_forecast=[[10.0, 12.0, 11.0]],
                    channel_names=["N060__Times_Sq_42_St"],
                    timestamps=["a", "b", "c"],
                    ground_truth=[[11.0, 13.0, 12.0]],
                ),
                evidence=EventEvidenceSpec(structured_events=[{"title": "YELL", "event_time": "2023-05-18 11:00:00"}]),
                decision=CalibrationDecisionSpec(mode="rag_explain", abstain=True),
                adjusted_forecast=[[10.0, 12.0, 11.0]],
                explanation_markdown="x",
            )

            write_forecast_artifacts(result, td, "overlap")
            meta = (Path(td) / "figures" / "overlap_hourly_curves.meta.json").read_text(encoding="utf-8")

            self.assertIn("raw == adjusted", meta)
            self.assertIn("no correction applied", meta)

    def test_plot_delta_meta_and_focused_figure_for_nonzero_correction(self):
        from agents.paper_workflow import (
            CalibrationDecisionSpec,
            EventEvidenceSpec,
            ForecastRequestSpec,
            ForecastResultSpec,
            NumericalForecastSpec,
            write_forecast_artifacts,
        )

        with tempfile.TemporaryDirectory() as td:
            result = ForecastResultSpec(
                request=ForecastRequestSpec(date="2023-05-18 10:00:00", horizon=3, station_scope="event_venue28", mode="event_adapter_frozen_moment"),
                numerical=NumericalForecastSpec(
                    raw_forecast=[[10.0, 12.0, 11.0], [30.0, 31.0, 32.0]],
                    channel_names=["N060__Times_Sq_42_St", "A013__49_St"],
                    timestamps=["a", "b", "c"],
                    ground_truth=[[11.0, 13.0, 12.0], [30.0, 31.0, 32.0]],
                ),
                evidence=EventEvidenceSpec(structured_events=[{"title": "YELL", "event_time": "2023-05-18 11:00:00"}]),
                decision=CalibrationDecisionSpec(
                    mode="event_adapter_frozen_moment",
                    abstain=False,
                    controller_allowed=True,
                    adjusted_channels=["N060__Times_Sq_42_St"],
                ),
                adjusted_forecast=[[10.0, 13.0, 12.0], [30.0, 31.0, 32.0]],
                explanation_markdown="x",
            )

            write_forecast_artifacts(result, td, "delta")
            meta = json.loads((Path(td) / "figures" / "delta_hourly_curves.meta.json").read_text(encoding="utf-8"))
            focused = Path(td) / "figures" / "delta_focused_adjusted_channels.png"

            self.assertGreater(meta["max_abs_adjusted_minus_raw"], 0.0)
            self.assertGreater(meta["active_correction_cells"], 0)
            self.assertIn("bounded correction area (adjusted - raw)", meta["figure_checks"])
            self.assertTrue(focused.is_file())


if __name__ == "__main__":
    unittest.main()
