import unittest


class FullPaperResultsTests(unittest.TestCase):
    def test_bootstrap_ci_reports_delta_direction(self):
        from experiments.run_full_paper_results import bootstrap_wape_delta_ci

        rows = [
            {"mode": "numerical_only", "anchor": 1, "wape": 20.0},
            {"mode": "event_adapter_frozen_moment", "anchor": 1, "wape": 18.0},
            {"mode": "numerical_only", "anchor": 2, "wape": 30.0},
            {"mode": "event_adapter_frozen_moment", "anchor": 2, "wape": 27.0},
            {"mode": "numerical_only", "anchor": 3, "wape": 25.0},
            {"mode": "event_adapter_frozen_moment", "anchor": 3, "wape": 24.0},
        ]

        ci = bootstrap_wape_delta_ci(rows, baseline_mode="numerical_only", compare_mode="event_adapter_frozen_moment", n_boot=50, seed=7)

        self.assertGreater(ci["mean_delta"], 0.0)
        self.assertLessEqual(ci["ci_low"], ci["ci_high"])
        self.assertEqual(ci["n_windows"], 3)


if __name__ == "__main__":
    unittest.main()
