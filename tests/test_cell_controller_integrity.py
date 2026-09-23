"""Fictional controller inputs only; no datasets, models, or external services."""

import unittest

import numpy as np

from agents.calibration_controller import CalibrationController


class CellControllerIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.controller = CalibrationController()
        self.raw = np.full((2, 4), 100.0)
        self.corr = np.full((2, 4), 0.02)
        self.audit = {key: 1.0 for key in self.controller.thresholds}

    def apply(self, **changes):
        args = {
            "raw_forecast": self.raw,
            "proposed_adjusted": self.raw * (1 + self.corr),
            "correction": self.corr,
            "channel_names": ["Fictional_A", "Fictional_B"],
            "adjusted_channel_candidates": ["Fictional_A", "Fictional_B"],
            "event_channel_mask": np.ones(self.raw.shape, dtype=bool),
            "evidence_audit": self.audit,
        }
        args.update(changes)
        return self.controller.apply(**args)

    def test_positional_api_and_old_keys_are_retained(self):
        result = self.controller.apply(
            self.raw, self.raw * (1 + self.corr), self.corr,
            ["Fictional_A", "Fictional_B"], ["Fictional_A", "Fictional_B"],
            np.ones((2, 4), dtype=bool), self.audit,
        )
        for key in ("controller_allowed", "abstain", "final_adjusted_forecast",
                    "adjusted_channels", "abstention_reason", "correction_stats",
                    "thresholds", "audit_scores", "included_units", "excluded_units"):
            self.assertIn(key, result)
        self.assertEqual(result["decision"], "apply")

    def test_proposal_cannot_bypass_hour_or_candidate_mask(self):
        mask = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])
        result = self.apply(
            proposed_adjusted=np.full((2, 4), 999.0),
            event_channel_mask=mask,
            adjusted_channel_candidates=["Fictional_A"],
        )
        expected = self.raw.copy()
        expected[0, 0] = 102.0
        np.testing.assert_array_equal(result["final_adjusted_forecast"], expected)
        self.assertEqual(result["decision"], "apply")
        self.assertEqual(result["adjusted_channels"], ["Fictional_A"])
        self.assertTrue(result["diagnostics"]["proposed_adjusted_ignored"])

    def test_core_secondary_weak_conflict_bounds(self):
        classes = np.tile(["core", "secondary", "weak", "conflict"], (2, 1))
        bounds = np.tile([0.05, 0.01, 0.05, 0.05], (2, 1))
        corr = np.array([[0.2] * 4, [-0.2] * 4])
        result = self.apply(correction=corr, cell_classes=classes, bound_matrix=bounds)
        expected_bounds = np.tile([0.05, 0.01, 0, 0], (2, 1))
        np.testing.assert_array_equal(result["bound_matrix"], expected_bounds)
        np.testing.assert_allclose(result["final_adjusted_forecast"],
                                   [[105, 101, 100, 100], [95, 99, 100, 100]])
        self.assertEqual(result["decision"], "apply")
        np.testing.assert_allclose(
            result["final_adjusted_forecast"],
            self.raw * (1 + np.asarray(result["mask"]) * np.clip(corr, -expected_bounds, expected_bounds)),
        )

    def test_secondary_requires_explicit_bounds_or_rho(self):
        classes = np.full((2, 4), "secondary")
        with self.assertRaisesRegex(ValueError, "secondary"):
            self.apply(cell_classes=classes)
        result = self.apply(cell_classes=classes, rho_partial=0.01)
        np.testing.assert_allclose(result["bound_matrix"], 0.01)
        np.testing.assert_allclose(result["final_adjusted_forecast"], 101.0)
        self.assertEqual(result["decision"], "apply")
        full_bound = self.apply(cell_classes=classes, rho_partial=self.controller.correction_bound)
        np.testing.assert_allclose(full_bound["bound_matrix"], self.controller.correction_bound)
        zero_bound = self.apply(cell_classes=classes, rho_partial=0.0)
        self.assertEqual(zero_bound["decision"], "abstain")
        for rho in (-0.1, 0.051, 0.2, 1.0, np.nan, [0.01]):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                self.apply(cell_classes=classes, rho_partial=rho)

    def test_large_finite_proposal_is_clipped_not_rejected(self):
        result = self.apply(correction=np.full((2, 4), 5.0))
        self.assertEqual(result["decision"], "apply")
        self.assertTrue(result["controller_allowed"])
        self.assertEqual(result["diagnostics"]["clipped_cells"], 8)
        self.assertEqual(result["correction_stats"]["max_abs_correction"], 0.05)
        self.assertEqual(result["correction_stats"]["mean_abs_correction"], 0.05)
        self.assertEqual(result["diagnostics"]["proposal_max_abs_correction"], 5.0)
        np.testing.assert_allclose(result["final_adjusted_forecast"], 105.0)

    def test_noevent_and_explain_preserve_raw(self):
        for kwargs, decision in (({"has_admissible_events": False}, "no_event_keep_raw"),
                                 ({"operating_mode": "EAF-MAS-X"}, "explain_only"),
                                 ({"operating_mode": "EAF-MAS-X", "has_admissible_events": False}, "no_event_keep_raw")):
            with self.subTest(decision=decision):
                result = self.apply(**kwargs)
                self.assertEqual(result["decision"], decision)
                self.assertFalse(result["controller_allowed"])
                self.assertFalse(np.any(result["corrected_mask"]))
                self.assertFalse(np.any(result["correction"]))
                self.assertEqual(result["correction_stats"]["max_abs_correction"], 0.0)
                self.assertEqual(result["diagnostics"]["proposal_max_abs_correction"], 0.02)
                np.testing.assert_array_equal(result["final_adjusted_forecast"], self.raw)

    def test_missing_low_or_severely_conflicted_audit_fails_closed(self):
        audits = [None, {}, dict(self.audit, source_validity_score=None),
                  dict(self.audit, geo_consistency_score=0.1),
                  dict(self.audit, severe_conflict_flags=["fictional_conflict"])]
        for audit in audits:
            with self.subTest(audit=audit):
                result = self.apply(evidence_audit=audit)
                self.assertEqual(result["decision"], "abstain")
                self.assertTrue(result["abstention_reason"])
                np.testing.assert_array_equal(result["final_adjusted_forecast"], self.raw)
        zero_thresholds = CalibrationController(0, 0, 0, 0)
        result = zero_thresholds.apply(self.raw, self.raw, self.corr,
                                       ["Fictional_A", "Fictional_B"], ["Fictional_A"],
                                       np.ones((2, 4)), {})
        self.assertFalse(result["controller_allowed"])

    def test_cell_scores_broadcast_exactly_and_do_not_override_aggregate(self):
        scores = {"temporal_alignment_score": [[1, 0, 1, 0]],
                  "geo_consistency_score": [[1], [0]]}
        result = self.apply(evidence_audit=dict(self.audit, cell_scores=scores))
        expected = self.raw.copy()
        expected[0, [0, 2]] = 102.0
        np.testing.assert_array_equal(result["final_adjusted_forecast"], expected)
        self.assertEqual(result["decision"], "partial_apply")
        direct = self.apply(evidence_audit=dict(self.audit, geo_consistency_score=[[1], [0]]))
        self.assertEqual(direct["adjusted_channels"], ["Fictional_A"])
        blocked = self.apply(evidence_audit=dict(self.audit, source_validity_score=0.1,
                                                cell_scores={"source_validity_score": np.ones((2, 4))}))
        self.assertEqual(blocked["decision"], "abstain")
        for value in (np.ones((2,)), np.ones((2, 4, 1)), np.full((2, 4), np.nan)):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                self.apply(evidence_audit=dict(self.audit, cell_scores={"source_validity_score": value}))

    def test_zero_raw_and_zero_correction_are_not_updates(self):
        for kwargs in ({"raw_forecast": np.zeros((2, 4))},
                       {"correction": np.zeros((2, 4))},
                       {"adjusted_channel_candidates": []}):
            with self.subTest(kwargs=kwargs):
                result = self.apply(**kwargs)
                self.assertEqual(result["decision"], "abstain")
                self.assertEqual(result["adjusted_channels"], [])
                self.assertEqual(result["correction_stats"]["active_correction_cells"], 0)
                self.assertEqual(result["correction_stats"]["max_abs_correction"], 0.0)
                self.assertFalse(np.any(result["correction"]))
                self.assertFalse(np.any(result["corrected_mask"]))

    def test_partial_requires_changed_and_unchanged_eligible_cells(self):
        mask = np.zeros((2, 4), dtype=bool)
        mask[0, :2] = True
        for unchanged_cause in ("zero_correction", "zero_raw", "failed_audit"):
            with self.subTest(unchanged_cause=unchanged_cause):
                raw = self.raw.copy()
                corr = self.corr.copy()
                scores = np.ones((2, 4))
                if unchanged_cause == "zero_correction":
                    corr[0, 1] = 0.0
                elif unchanged_cause == "zero_raw":
                    raw[0, 1] = 0.0
                else:
                    scores[0, 1] = 0.0
                result = self.apply(
                    raw_forecast=raw, correction=corr, event_channel_mask=mask,
                    evidence_audit=dict(self.audit, source_validity_score=scores),
                )
                self.assertEqual(result["decision"], "partial_apply")
                self.assertEqual(result["diagnostics"]["eligible_cells"], 2)
                self.assertEqual(result["diagnostics"]["corrected_cells"], 1)
                expected = raw.copy()
                expected[0, 0] = 102.0
                np.testing.assert_array_equal(result["final_adjusted_forecast"], expected)

    def test_outside_support_magnitude_does_not_affect_applied_statistics(self):
        mask = np.zeros((2, 4), dtype=bool)
        mask[0, 0] = True
        corr = np.full((2, 4), 10.0)
        corr[0, 0] = 0.01
        result = self.apply(correction=corr, event_channel_mask=mask)
        self.assertEqual(result["decision"], "apply")
        self.assertEqual(result["correction_stats"]["max_abs_correction"], 0.01)
        self.assertEqual(result["correction_stats"]["mean_abs_correction"], 0.01 / 8)
        self.assertEqual(result["diagnostics"]["proposal_max_abs_correction"], 10.0)
        self.assertEqual(result["diagnostics"]["requested_effect_cells"], 1)

    def test_full_candidate_list_with_inactive_zero_scores_and_conflict_bounds(self):
        mask = np.array([[True, True, False, False], [False] * 4])
        classes = np.array([["core", "conflict", "weak", "weak"], ["weak"] * 4])
        bounds = np.zeros((2, 4), dtype=np.float32)
        bounds[0, 0] = 0.05
        corr = np.zeros((2, 4))
        corr[0, 0] = 0.02
        scores = np.zeros((2, 4))
        scores[0, 0] = 1.0
        result = self.apply(
            correction=corr, event_channel_mask=mask, cell_classes=classes,
            bound_matrix=bounds, operating_mode="EAF-MAS-C", has_admissible_events=True,
            evidence_audit={key: scores.copy() for key in self.controller.thresholds},
        )
        expected = self.raw.copy()
        expected[0, 0] = 102.0
        np.testing.assert_array_equal(result["final_adjusted_forecast"], expected)
        np.testing.assert_array_equal(result["correction"], corr)
        self.assertEqual(result["decision"], "apply")
        self.assertEqual(result["adjusted_channels"], ["Fictional_A"])
        self.assertEqual(result["diagnostics"]["audit_failures"], [])
        self.assertEqual(result["bound_matrix"][0][1], 0.0)

    def test_shapes_nonfinite_and_negative_raw_raise_value_error(self):
        for key in ("raw_forecast", "proposed_adjusted", "correction", "event_channel_mask", "bound_matrix"):
            for value in ([1, 1], np.ones((2, 1)), [[1], [1, 2]],
                          np.full((2, 4), np.nan), np.full((2, 4), np.inf)):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.apply(**{key: value})
        for kwargs in ({"raw_forecast": -self.raw}, {"event_channel_mask": self.raw},
                       {"bound_matrix": np.full((2, 4), -0.1)},
                       {"bound_matrix": np.full((2, 4), 0.1)},
                       {"evidence_audit": dict(self.audit, source_validity_score=np.nan)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.apply(**kwargs)

    def test_names_classes_and_modes_validate(self):
        invalid = [
            {"channel_names": ["Fictional_A"]},
            {"channel_names": ["Fictional_A", "Fictional_A"]},
            {"channel_names": ["Fictional_A", ""]},
            {"channel_names": "AB"},
            {"adjusted_channel_candidates": ["Unknown"]},
            {"adjusted_channel_candidates": ["Fictional_A", "Fictional_A"]},
            {"adjusted_channel_candidates": "Fictional_A"},
            {"cell_classes": ["core", "weak"]},
            {"cell_classes": np.full((2, 4), "unknown")},
            {"operating_mode": "unknown"},
            {"has_admissible_events": "False"},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.apply(**kwargs)


if __name__ == "__main__":
    unittest.main()
