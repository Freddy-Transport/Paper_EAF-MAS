import json
import tempfile
import unittest
from pathlib import Path


CASE_MD = """# Event-Aware Forecast Explanation

## 1. Forecast Request
- Anchor time: 2023-06-19 10:00:00
- Forecast horizon: 192 hours
- Mode: event_adapter_frozen_moment
- Station scope: event_venue28
- Calibration enabled: True

## 2. Raw Forecast
- Raw model: PT-MOMENT
- Raw metrics: Overall WAPE 17.6

## 3. Forecast-time Evidence Audit
### Structured Future Events within Horizon
- event_name: TSQ LIVE Plaza Programming; event_time: 2023-06-20 17:00:00; source_type: structured_event_kb; known_before_anchor: yes.
- event_name: Violife Cheese street event; event_time: 2023-06-22 10:00:00; source_type: qwenplus_model_summary; known_before_anchor: yes.
### Model-Assisted Summary (Non-citable)
- Qwen-Plus summary: A scheduled street activation near Times Square is expected to increase localized pedestrian movement around nearby subway entrances.
### Geo-temporal consistency
- event_location: Times Square plaza; matched_station: Times Sq-42 St; distance_to_station: 180m; geo_consistency_score: 0.879; temporal_alignment_score: 0.950; conflict_flags: none.

## 4. Candidate Event-Station-Channel Units
### Included
- station/channel: Times Sq-42 St; relation: direct venue proximity; gate score: 0.92; reason: same plaza catchment.
### Excluded
- station/channel: distant station; exclusion reason: outside event-station mask.

## 5. Historical Residual Memory
### Matched cases
- event: Spring Broadway Fair; split: train_val_memory; event_type_match: yes; day_type_match: yes; station_rank_match: yes; residual_direction: increase; median correction: +0.8%; IQR: 0.4; N_eff: 32.
- event: Broadway Astoria Fall Festival; split: train_val_memory; event_type_match: yes; day_type_match: partial; station_rank_match: yes; residual_direction: increase; median correction: +0.7%; IQR: 0.3; N_eff: 24.
### Residual support summary
- direction agreement: increase; median correction: +0.8%; confidence cap: 0.85.

## 6. Calibration Controller
### Calibration Decision
- Calibration decision: allow bounded correction.
- Confidence: 0.80
- Correction bound: 0.050
- Max correction: 0.0482
- Adjusted station/channel units: Times Sq-42 St and Bryant Park area.
- Excluded station/channel units: 20
- Abstention reason if any: none.

## 7. Adjusted Forecast
- Adjusted model: frozen residual adapter
- Adapter checkpoint: event_adapter_formal_frozen_real_lpmoment
- Overall 192h Raw WAPE: 17.631
- Overall 192h Adjusted WAPE: 17.536
- Event-window subset: raw WAPE=12.300; adjusted WAPE=11.900
- Adjusted-channel subset: raw WAPE=15.400; adjusted WAPE=14.600
- Correction magnitude statistics: max 0.0482, mean 0.0061.

## 8. Explanation
### Factual evidence
The structured event record places the planned Times Square activation inside the 192-hour forecast horizon and near the Times Sq-42 St station complex.
### Residual analogues
Historical street-fair cases in train/validation memory show a consistent positive residual around similarly ranked stations, so the analogue evidence supports an upward bounded correction.
### Calibration rationale
The controller allows correction because source, geo-temporal, semantic, and residual-support scores pass their thresholds and the candidate station is inside the event-channel mask.
### Uncertainty
The model-assisted summary is non-citable and the residual memory is not an exact event replica, so uncertainty remains moderate.
### Risk control
The residual adapter is limited by a 0.05 correction bound and distant channels are excluded to avoid global forecast distortion.

## Post-hoc Metrics
Ground truth metrics are reported outside forecast-time reasoning.
"""


ABSTAIN_MD = CASE_MD.replace("event_adapter_frozen_moment", "rag_explain").replace(
    "Calibration enabled: True", "Calibration enabled: False"
).replace("allow bounded correction", "abstain and keep PT-MOMENT").replace(
    "Max correction: 0.0482", "Max correction: 0.0000"
)


class PaperCaseCardTests(unittest.TestCase):
    def _root(self) -> Path:
        root = Path(tempfile.mkdtemp())
        for case_name, text in {
            "event_intervention": CASE_MD,
            "safe_abstention": ABSTAIN_MD,
        }.items():
            exp = root / "qwenplus_live_case_set_summary_only" / case_name / "explanations"
            exp.mkdir(parents=True, exist_ok=True)
            (exp / "case.md").write_text(text, encoding="utf-8")
            bad = exp / ".ipynb_checkpoints"
            bad.mkdir()
            (bad / "case-checkpoint.md").write_text("# stale checkpoint", encoding="utf-8")

        physical_event_key = "tsq live plaza programming|2023-06-20 17:00|times square plaza"
        prediction = root / "controller_abstention_case.json"
        prediction.write_text(
            json.dumps(
                {
                    "request": {
                        "date": "2023-06-19 10:00:00",
                        "horizon": 192,
                        "mode": "event_adapter_frozen_moment",
                    },
                    "numerical": {"raw_forecast": [[100.0, 110.0]]},
                    "adjusted_forecast": [[100.0, 110.0]],
                    "raw_metrics": {"wape": 17.6},
                    "metrics": {"wape": 17.6},
                    "event_window_metrics": {
                        "raw": {"wape": 21.215},
                        "adjusted": {"wape": 21.215},
                    },
                    "focused_prediction": {
                        "focused_forecast_rows": [
                            {
                                "channel_name": "N060__Times_Sq_42_St",
                                "raw_forecast": 100.0,
                                "adjusted_forecast": 100.0,
                            }
                        ],
                        "focused_forecast_metrics": {},
                    },
                    "decision": {
                        "controller_allowed": False,
                        "abstain": True,
                        "reason": "Calibration controller abstained: source validity below threshold",
                        "audit_scores": {"source_validity_score": 0.0},
                        "correction_stats": {
                            "max_abs_correction": 0.04,
                            "active_correction_cells": 1,
                        },
                    },
                    "evidence": {
                        "structured_events": [
                            {
                                "title": "TSQ LIVE Plaza Programming",
                                "event_time": "2023-06-20 17:00:00",
                                "event_type": "public_programming",
                                "location": "Times Square plaza | synthetic test fixture",
                                "station_complex": "Times Sq-42 St",
                                "channel_name": "N060__Times_Sq_42_St",
                                "content": "distance_to_station=180m; end_datetime=2023-06-20 20:00:00",
                            }
                        ],
                        "model_assisted_summaries": [{"summary": "Synthetic forecast-time summary."}],
                        "historical_event_cases": [],
                        "evidence_audit": {
                            "source_validity_score": 0.0,
                            "geo_consistency_score": 0.879,
                            "temporal_alignment_score": 0.95,
                            "residual_support_score": 0.95,
                            "evidence_validity_score": 0.7,
                            "severe_conflict_flags": [],
                            "residual_support_summary": {},
                            "thresholds": {
                                "source_validity_score": 0.7,
                                "geo_consistency_score": 0.6,
                                "temporal_alignment_score": 0.6,
                                "residual_support_score": 0.6,
                            },
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        reports = root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "selected_visualization_cases.json").write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_type": "controller_abstention",
                            "status": "selected",
                            "case_id": "synthetic_controller_abstention",
                            "method_label": "EAF-MAS-C",
                            "input_file": str(prediction),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (reports / "qwenplus_live_evidence_summary.json").write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "physical_event_key": physical_event_key,
                            "summaries": [
                                {
                                    "event_key": physical_event_key,
                                    "summary": "Synthetic forecast-time summary.",
                                    "event_summary": "Synthetic forecast-time summary.",
                                    "event_relevance_to_station": "Synthetic station relevance.",
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_case_card_blocks_use_complete_bullets_and_multihop_label(self):
        from experiments.assemble_paper_figures import _case_card_blocks, _case_payload

        payload = _case_payload(self._root(), "event_intervention")
        blocks = _case_card_blocks(payload)
        titles = [title for title, _ in blocks]
        all_text = "\n".join([title + "\n" + "\n".join(bullets) for title, bullets in blocks])

        self.assertIn("Multi-hop reasoning", titles)
        self.assertNotIn("Human-readable reasoning", titles)
        self.assertNotIn("...", all_text)
        self.assertNotIn("…", all_text)
        self.assertIn("event-channel mask", all_text)
        self.assertIn("positive residual", all_text)
        self.assertIn("0.05 correction bound", all_text)

    def test_generate_case_cards_writes_separate_main_figures(self):
        from experiments.assemble_paper_figures import generate_case_cards

        root = self._root()
        manifest = []
        out = root / "figure_paper" / "main"
        out.mkdir(parents=True)
        generate_case_cards(root, out, manifest)

        self.assertTrue((out / "fig7a_bounded_event_intervention_case_card.png").is_file())
        self.assertTrue((out / "fig7b_weak_evidence_abstention_case_card.png").is_file())
        self.assertFalse((out / "fig7_case_card_pair.png").exists())
        names = {entry["filename"] for entry in manifest}
        self.assertIn("fig7a_bounded_event_intervention_case_card", names)
        self.assertIn("fig7b_weak_evidence_abstention_case_card", names)


if __name__ == "__main__":
    unittest.main()
