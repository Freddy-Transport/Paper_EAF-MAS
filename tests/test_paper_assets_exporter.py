import json
import tempfile
import unittest
from pathlib import Path


class PaperAssetsExporterTests(unittest.TestCase):
    def _write_case(self, root: Path, name: str, mode: str, raw_wape: float, adjusted_wape: float, markdown: str):
        case_dir = root / "cases_vllm" / name
        (case_dir / "explanations").mkdir(parents=True)
        (case_dir / "logs").mkdir(parents=True)
        (case_dir / "reports").mkdir(parents=True)
        (case_dir / "explanations" / f"{name}.md").write_text(markdown, encoding="utf-8")
        (case_dir / "logs" / f"llm_explanation_{mode}.json").write_text(
            json.dumps(
                {
                    "result": {
                        "event_summary": f"{name} event",
                        "station_event_linking": "venue to station link",
                        "multi_hop_reasoning": "event to demand to calibration",
                        "calibration_rationale": "bounded calibration",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (case_dir / "reports" / "run_summary.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "raw_metrics": {"wape": raw_wape},
                    "adjusted_metrics": {"wape": adjusted_wape},
                    "has_major_event": True,
                    "adjusted_channels": ["A013__49_St"] if mode != "rag_explain" else [],
                    "explanation_quality": {
                        "accepted_evidence_count": 1 if "https://" in markdown else 0,
                        "citation_coverage": 1.0 if "https://" in markdown else 0.0,
                        "station_event_linking": True,
                        "multi_hop_reasoning": True,
                        "calibration_decision_consistent": True,
                    },
                    "artifacts": {"explanation_md": str(case_dir / "explanations" / f"{name}.md")},
                }
            ),
            encoding="utf-8",
        )

    def test_exporter_writes_tables_and_appendix_with_three_case_types(self):
        from experiments.export_paper_explanation_assets import export_paper_assets

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "full_rolling" / "tables").mkdir(parents=True)
            (root / "reports").mkdir()
            (root / "reports" / "evidence_quality_summary.json").write_text(
                json.dumps({"source": "cases_vllm", "cases": []}),
                encoding="utf-8",
            )
            self._write_case(
                root,
                "adapter_success",
                "event_adapter_frozen_moment",
                20.0,
                18.0,
                "## Event Evidence\n### Accepted External Evidence\n- [Official Event](https://example.org/event)\n\n## Forecast-Time Model Reasoning\n### Forecast-time Evidence\nEvidence only.\n\n## Post-hoc Metrics\n```json\n{\"wape\": 18.0}\n```",
            )
            self._write_case(
                root,
                "correct_abstention",
                "rag_explain",
                19.0,
                19.0,
                "## Event Evidence\nNo accepted citation-quality external evidence was available.\n\n## Forecast-Time Model Reasoning\n### Calibration Decision\nNo numerical event calibration was applied.",
            )
            self._write_case(
                root,
                "adapter_failure",
                "event_adapter_frozen_moment",
                18.0,
                19.0,
                "## Event Evidence\nNo accepted citation-quality external evidence was available.\n\n## Forecast-Time Model Reasoning\n### Uncertainty\nEvidence was weak.",
            )

            summary = export_paper_assets(root)

            self.assertEqual(summary["case_count"], 3)
            self.assertTrue((root / "paper_assets" / "explanation_case_table.tex").is_file())
            self.assertTrue((root / "paper_assets" / "evidence_quality_table.tex").is_file())
            evidence_table = (root / "paper_assets" / "evidence_quality_table.tex").read_text(encoding="utf-8")
            appendix = (root / "paper_assets" / "appendix_explanations.tex").read_text(encoding="utf-8")
            self.assertIn("Source-assisted", evidence_table)
            self.assertIn("Model summary", evidence_table)
            self.assertIn("adapter_success", appendix)
            self.assertIn("correct_abstention", appendix)
            self.assertIn("adapter_failure", appendix)

    def test_forecast_time_section_excludes_posthoc_wape(self):
        from experiments.export_paper_explanation_assets import forecast_time_text_has_posthoc_metrics

        markdown = """## Forecast-Time Model Reasoning
### Forecast-time Evidence
The event is near the station.

## Post-hoc Metrics
```json
{"wape": 17.5}
```"""
        self.assertFalse(forecast_time_text_has_posthoc_metrics(markdown))

        bad = """## Forecast-Time Model Reasoning
The WAPE was 17.5, so the event reasoning is good.

## Post-hoc Metrics
{}"""
        self.assertTrue(forecast_time_text_has_posthoc_metrics(bad))


if __name__ == "__main__":
    unittest.main()
