import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


class MajorEventScreeningTests(unittest.TestCase):
    def test_deduplicate_physical_events_merges_station_duplicates(self):
        from event_screening.screener import deduplicate_physical_events

        events = [
            {
                "title": "Yankees vs Mets",
                "event_time": "2023-06-10 19:00:00",
                "location": "Yankee Stadium",
                "event_type": "Special Event",
                "station_complex_id": "N203",
                "channel_name": "N203__161_St_Yankee_Stadium",
                "content": "matched_venue=Yankee Stadium; distance_to_station=120.0m",
            },
            {
                "title": "Yankees vs Mets",
                "event_time": "2023-06-10 19:00:00",
                "location": "Yankee Stadium",
                "event_type": "Special Event",
                "station_complex_id": "R262",
                "channel_name": "R262__167_St",
                "content": "matched_venue=Yankee Stadium; distance_to_station=620.0m",
            },
        ]

        physical = deduplicate_physical_events(events)

        self.assertEqual(len(physical), 1)
        self.assertEqual(physical[0].duplicate_count, 2)
        self.assertEqual(set(physical[0].affected_channels), {"N203__161_St_Yankee_Stadium", "R262__167_St"})

    def test_rule_prefilter_demotes_noise_and_keeps_major_events(self):
        from event_screening.screener import deduplicate_physical_events, rule_prefilter_event

        rows = [
            {"title": "Soccer - Non Regulation", "event_time": "2023-06-01 10:00:00", "location": "Park field", "event_type": "Sport - Youth", "content": "youth soccer permit"},
            {"title": "Columbia Greenmarket Thursday and Sunday", "event_time": "2023-06-01 10:00:00", "location": "Street", "event_type": "Farmers Market", "content": "greenmarket"},
            {"title": "Permitted Film Event", "event_time": "2023-06-01 10:00:00", "location": "Midtown", "event_type": "Shooting Permit", "content": "filming permit setup"},
            {"title": "Great Lawn Winter Closure", "event_time": "2023-06-01 10:00:00", "location": "Central Park", "event_type": "Special Event", "content": "lawn closure"},
            {"title": "Yankees vs Red Sox", "event_time": "2023-06-01 19:00:00", "location": "Yankee Stadium", "event_type": "Special Event", "content": "matched_venue=Yankee Stadium"},
            {"title": "Thanksgiving Day Parade", "event_time": "2023-11-23 09:00:00", "location": "Times Square", "event_type": "Parade", "content": "major parade"},
            {"title": "Station Closure", "event_time": "2023-06-01 10:00:00", "location": "Station", "event_type": "Service Disruption", "content": "station closure service disruption"},
        ]
        physical = deduplicate_physical_events(rows)
        by_title = {p.title: rule_prefilter_event(p) for p in physical}

        for title in ["Soccer - Non Regulation", "Columbia Greenmarket Thursday and Sunday", "Permitted Film Event", "Great Lawn Winter Closure"]:
            self.assertFalse(by_title[title].high_priority, title)
            self.assertIn("noise", by_title[title].reason.lower())
        for title in ["Yankees vs Red Sox", "Thanksgiving Day Parade", "Station Closure"]:
            self.assertTrue(by_title[title].high_priority, title)
            self.assertGreaterEqual(by_title[title].rule_score, 0.65)

    def test_traffic_evidence_uses_history_and_flags_low_volume(self):
        from event_screening.traffic import compute_traffic_evidence

        dates = pd.date_range("2023-01-01", periods=24 * 28, freq="h")
        values = np.full(len(dates), 100.0, dtype=float)
        # Same hour/day historical baseline is 100; event window jumps to 180.
        values[24 * 21 + 10: 24 * 21 + 13] = 180.0
        values[24 * 22 + 10: 24 * 22 + 13] = 5.0
        df = pd.DataFrame({"date": dates, "station_a": values})

        evidence = compute_traffic_evidence(df, ["station_a"], "2023-01-22 10:00:00", horizon_hours=3, train_end_idx=24 * 28)
        low = compute_traffic_evidence(df, ["station_a"], "2023-01-23 10:00:00", horizon_hours=3, train_end_idx=24 * 28)

        self.assertEqual(evidence["n_channels"], 1)
        self.assertGreater(evidence["delta_pct"], 0.5)
        self.assertGreater(evidence["z_score"], 2.0)
        self.assertFalse(evidence["low_volume_warning"])
        self.assertTrue(low["low_volume_warning"])

    def test_llm_schema_parses_valid_json_and_marks_invalid_review(self):
        from event_screening.llm import parse_llm_decision

        valid = '{"is_major_event": true, "major_event_type": "sports", "crowd_scale": "large", "transit_impact_likelihood": 0.82, "expected_direction": "increase", "affected_scope": "station", "traffic_evidence_used": true, "keep_for_modeling": true, "reason": "Yankee Stadium game with traffic spike"}'
        invalid = '{"is_major_event": true, "major_event_type": "tiny_party"}'
        wrapped = 'Reasoning text before JSON. ```json\n{"is_major_event": true, "major_event_type": "concert", "crowd_scale": "mega", "transit_impact_likelihood": 0.9, "expected_direction": "increase", "affected_scope": "multi_station", "traffic_evidence_used": true, "keep_for_modeling": true, "reason": "MSG concert with strong traffic evidence"}\n``` extra text'

        parsed = parse_llm_decision(valid)
        fallback = parse_llm_decision(invalid)
        wrapped_parsed = parse_llm_decision(wrapped)

        self.assertTrue(parsed.keep_for_modeling)
        self.assertEqual(parsed.major_event_type, "sports")
        self.assertTrue(wrapped_parsed.keep_for_modeling)
        self.assertEqual(wrapped_parsed.major_event_type, "concert")
        self.assertTrue(fallback.needs_review)
        self.assertFalse(fallback.keep_for_modeling)

    def test_prompt_warns_against_special_event_permit_bias(self):
        from event_screening.llm import build_llm_prompt
        from event_screening.screener import deduplicate_physical_events

        event = deduplicate_physical_events([{"title": "Party", "event_time": "2023-06-03 12:00:00", "event_type": "Special Event", "location": "Park", "content": "birthday party permit"}])[0]
        prompt = build_llm_prompt(event, {"z_score": 0.1, "delta_pct": 0.02, "actual_sum": 100.0})

        self.assertIn("Do not classify an event as major only because", prompt)
        self.assertIn("Special Event", prompt)
        self.assertIn("JSON", prompt)

    def test_hf_api_runner_without_token_is_unavailable(self):
        from event_screening.llm import HFApiRunner, RunnerUnavailable

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RunnerUnavailable):
                HFApiRunner("Qwen/Qwen2.5-3B-Instruct")

    def test_isolated_transformers_paths_do_not_touch_conda(self):
        from event_screening.llm import IsolatedTransformersRunner

        runner = IsolatedTransformersRunner(
            "Qwen/Qwen2.5-3B-Instruct",
            env_dir="/root/autodl-tmp/llm_screening_env",
            hf_cache="/root/autodl-tmp/hf_cache",
            require_ready=False,
        )
        commands = "\n".join(runner.install_commands())

        self.assertIn("/root/autodl-tmp/llm_screening_env", commands)
        self.assertIn("/root/autodl-tmp/hf_cache", commands)
        self.assertNotIn("/root/miniconda3", commands)

    def test_data_output_overwrite_only_for_llm_runners(self):
        from experiments.screen_major_events import should_write_data_output

        self.assertFalse(should_write_data_output("heuristic", requested=True, allow_heuristic=False))
        self.assertTrue(should_write_data_output("heuristic", requested=True, allow_heuristic=True))
        self.assertTrue(should_write_data_output("hf_api", requested=True, allow_heuristic=False))
        self.assertTrue(should_write_data_output("isolated_transformers", requested=True, allow_heuristic=False))
        self.assertFalse(should_write_data_output("hf_api", requested=False, allow_heuristic=False))


if __name__ == "__main__":
    unittest.main()
