import unittest

import numpy as np


class VisualizationMetricsTests(unittest.TestCase):
    def test_forecast_metrics_and_gain_flags(self):
        from experiments.visualization.eaf_viz.metrics import (
            beneficial_flag,
            bound_utilization,
            correction_magnitude,
            harmful_flag,
            mae,
            relative_wape_gain,
            rmse,
            smape,
            wape,
            wape_gain,
        )

        actual = np.array([100.0, 200.0, 300.0])
        raw = np.array([110.0, 180.0, 330.0])
        adjusted = np.array([104.0, 195.0, 306.0])

        self.assertAlmostEqual(mae(actual, raw), 20.0)
        self.assertAlmostEqual(rmse(actual, raw), np.sqrt((100 + 400 + 900) / 3.0))
        self.assertAlmostEqual(wape(actual, raw), 10.0)
        self.assertGreater(smape(actual, raw), 0.0)

        gain = wape_gain(actual, raw, adjusted)
        self.assertGreater(gain, 0.0)
        self.assertGreater(relative_wape_gain(actual, raw, adjusted), 0.0)
        self.assertTrue(beneficial_flag(gain, neutral_threshold=0.01))
        self.assertFalse(harmful_flag(gain, neutral_threshold=0.01))

        corr = correction_magnitude(raw, adjusted)
        np.testing.assert_allclose(corr, adjusted - raw)
        util = bound_utilization(raw, adjusted, correction_bound=0.05)
        self.assertEqual(util.shape, raw.shape)
        self.assertTrue(np.all(util >= 0.0))


if __name__ == "__main__":
    unittest.main()
