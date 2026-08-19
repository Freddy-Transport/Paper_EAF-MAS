import unittest

import numpy as np

from agents.schemas import EventInfo


class EventAdapterTests(unittest.TestCase):
    def test_feature_cube_has_stable_shape_and_missing_enrichment_fallback(self):
        from agents.event_adapter import FEATURE_NAMES, build_event_feature_cube

        raw = np.full((2, 4), 100.0, dtype=np.float32)
        timestamps = ["2023-02-01 18:00:00", "2023-02-01 19:00:00", "2023-02-01 20:00:00", "2023-02-01 21:00:00"]
        channel_names = ["R610__Atlantic_Av_Barclays_Ctr", "N060__Times_Sq_42_St"]
        events = [
            EventInfo(
                title="Brooklyn Nets vs Knicks",
                content="station_complex_id=R610; channel_name=R610__Atlantic_Av_Barclays_Ctr",
                event_time="2023-02-01 19:30:00",
                location="Barclays Center",
                event_type="Sports",
                impact_tier="A",
                distance_m=120.0,
            )
        ]
        channel_meta = {
            "R610__Atlantic_Av_Barclays_Ctr": {"rank": 20, "station_complex_id": "R610"},
            "N060__Times_Sq_42_St": {"rank": 1, "station_complex_id": "N060"},
        }

        cube = build_event_feature_cube(raw, timestamps, channel_names, events, channel_meta)

        self.assertEqual(cube.shape, (2, 4, len(FEATURE_NAMES)))
        self.assertGreater(float(cube[0].sum()), 0.0)
        self.assertEqual(float(cube[1, :, FEATURE_NAMES.index("station_match")].max()), 0.0)

    def test_no_event_adapter_correction_near_zero(self):
        from agents.event_adapter import build_adapter_from_config, apply_correction

        adapter = build_adapter_from_config({"input_dim": 8, "hidden_dim": 4, "max_correction": 0.10})
        features = np.zeros((2, 3, 8), dtype=np.float32)
        raw = np.full((2, 3), 100.0, dtype=np.float32)

        corrected, correction = apply_correction(adapter, raw, features, device="cpu")

        self.assertTrue(np.allclose(correction, 0.0, atol=1e-4))
        self.assertTrue(np.allclose(corrected, raw, atol=1e-3))

    def test_adapter_correction_is_clamped(self):
        from agents.event_adapter import EventResidualAdapter, apply_correction

        adapter = EventResidualAdapter(input_dim=2, hidden_dim=4, max_correction=0.10)
        for param in adapter.parameters():
            param.data.fill_(5.0)
        raw = np.full((1, 2), 100.0, dtype=np.float32)
        features = np.ones((1, 2, 2), dtype=np.float32)

        _, correction = apply_correction(adapter, raw, features, device="cpu")

        self.assertLessEqual(float(np.max(correction)), 0.1001)
        self.assertGreaterEqual(float(np.min(correction)), -0.1001)

    def test_loaded_adapter_uses_validation_blend_factor(self):
        import tempfile
        from agents.event_adapter import EventResidualAdapter, apply_correction, load_adapter, save_adapter

        with tempfile.TemporaryDirectory() as td:
            adapter = EventResidualAdapter(input_dim=2, hidden_dim=4, max_correction=0.10)
            for param in adapter.parameters():
                param.data.fill_(5.0)
            save_adapter(
                adapter,
                td,
                {"input_dim": 2, "hidden_dim": 4, "max_correction": 0.10},
                {"validation_blend_calibration": {"blend_factor": 0.0}},
            )

            loaded, config = load_adapter(td, device="cpu")
            raw = np.full((1, 2), 100.0, dtype=np.float32)
            features = np.ones((1, 2, 2), dtype=np.float32)
            _, correction = apply_correction(loaded, raw, features, device="cpu")

            self.assertEqual(config["blend_factor"], 0.0)
            self.assertTrue(np.allclose(correction, 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
