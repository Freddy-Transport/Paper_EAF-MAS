import json
import tempfile
import unittest
from pathlib import Path


class RefreshSavedExplanationsTests(unittest.TestCase):
    def test_refresh_merges_model_assisted_summary_without_citation_sections(self):
        from experiments.refresh_saved_explanations import refresh_saved_explanations

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_dir = root / "cases_vllm" / "case_a" / "predictions"
            pred_dir.mkdir(parents=True)
            prediction = pred_dir / "case.json"
            prediction.write_text(
                json.dumps(
                    {
                        "request": {
                            "date": "2023-06-19 10:00:00",
                            "horizon": 192,
                            "station_scope": "event_venue28",
                            "mode": "rag_explain",
                            "retrieval_policy": "online_cached",
                        },
                        "evidence": {
                            "sources": [
                                {
                                    "title": "Weak source",
                                    "url": "https://example.com/weak",
                                    "accepted": False,
                                    "rejected_reason": "missing_date_match",
                                }
                            ],
                            "structured_events": [{"title": "TSQ LIVE", "event_time": "2023-06-20"}],
                            "local_residual_cases": [],
                            "model_assisted_summaries": [],
                            "has_major_event": True,
                        },
                        "decision": {"mode": "rag_explain", "abstain": True, "reason": "explain only"},
                        "metrics": {},
                        "explanation_markdown": "## Forecast-Time Model Reasoning\nExisting reasoning.",
                    }
                ),
                encoding="utf-8",
            )
            reports = root / "reports"
            reports.mkdir()
            (reports / "citation_evidence_status.json").write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_prediction": str(prediction),
                                "model_assisted_summaries": [
                                    {"summary": "Model-assisted TSQ LIVE event context.", "source_agent": "qwen_plus"}
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = refresh_saved_explanations(root)
            markdown = (root / "cases_vllm" / "case_a" / "explanations" / "case.md").read_text(encoding="utf-8")

            self.assertEqual(result["refreshed_count"], 1)
            self.assertNotIn("### Verified External Evidence", markdown)
            self.assertNotIn("### Source-Assisted Evidence", markdown)
            self.assertIn("### Model-Assisted Summary (Non-citable)", markdown)
            self.assertIn("Model-assisted TSQ LIVE event context", markdown)


if __name__ == "__main__":
    unittest.main()
