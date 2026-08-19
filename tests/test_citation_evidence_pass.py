import json
import tempfile
import unittest
from pathlib import Path


class CitationEvidencePassTests(unittest.TestCase):
    def test_high_value_selector_prefers_public_programming_over_brand_permit(self):
        from experiments.run_citation_evidence_pass import _high_value_events

        prediction = {
            "evidence": {
                "structured_events": [
                    {
                        "title": "Violife Cheese",
                        "event_time": "2023-06-22 08:00:00",
                        "location": "SPRING STREET between MERCER STREET and BROADWAY",
                        "impact_tier": "A",
                        "event_category": "other",
                        "should_adjust_forecast": True,
                    },
                    {
                        "title": "TSQ LIVE: 45/46 Plaza Programming",
                        "event_time": "2023-06-20 17:00:00",
                        "location": "Broadway Pedestrian Plaza Times Square",
                        "impact_tier": "A",
                        "event_category": "other",
                        "should_adjust_forecast": True,
                    },
                ]
            }
        }

        selected = _high_value_events(prediction, max_events=1)

        self.assertEqual(len(selected), 1)
        self.assertIn("TSQ LIVE", selected[0]["title"])

    def test_missing_runtime_env_writes_skipped_status_without_secret(self):
        from experiments.run_citation_evidence_pass import run_citation_pass

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case_dir = root / "cases_vllm" / "case_a" / "predictions"
            case_dir.mkdir(parents=True)
            (case_dir / "case.json").write_text(
                json.dumps(
                    {
                        "request": {"date": "2023-06-19 10:00:00"},
                        "evidence": {
                            "structured_events": [
                                {
                                    "title": "TSQ LIVE",
                                    "event_time": "2023-06-20 17:00:00",
                                    "location": "Times Square",
                                    "impact_tier": "A",
                                    "should_adjust_forecast": True,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_citation_pass(root, env={})
            status_path = root / "reports" / "citation_evidence_status.json"
            payload = status_path.read_text(encoding="utf-8")

            self.assertEqual(result["status"], "skipped")
            self.assertIn("OPENAI_API_KEY", result["required_env_present"])
            self.assertNotIn("sk-", payload)


if __name__ == "__main__":
    unittest.main()
