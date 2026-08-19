import sys
import types
import unittest

if 'torch' not in sys.modules:
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float32 = 'float32'
    torch.bfloat16 = 'bfloat16'
    torch.no_grad = lambda: (lambda fn: fn)
    torch_utils = types.ModuleType('torch.utils')
    torch_utils_data = types.ModuleType('torch.utils.data')
    torch_utils_data.DataLoader = object
    sys.modules['torch'] = torch
    sys.modules['torch.utils'] = torch_utils
    sys.modules['torch.utils.data'] = torch_utils_data



class Top128MultiAgentTests(unittest.TestCase):
    def test_event_relevance_agent_keeps_high_impact_station_event(self):
        from agents.event_relevance_agent import EventRelevanceAgent
        from agents.schemas import EventInfo

        agent = EventRelevanceAgent(
            channel_map={
                "R610__Atlantic_Av_Barclays_Ctr": {
                    "station_complex_id": "R610",
                    "station_complex": "Atlantic Av-Barclays Ctr",
                }
            },
            relevance_threshold=0.5,
        )
        events = [
            EventInfo(
                source="unit",
                title="Brooklyn Nets vs Knicks",
                content="station_complex_id=R610; channel_name=R610__Atlantic_Av_Barclays_Ctr",
                event_time="2023-02-01 19:30:00",
                location="Barclays Center",
                event_type="Sports",
            ),
            EventInfo(
                source="unit",
                title="Routine construction",
                content="station_complex_id=R610; channel_name=R610__Atlantic_Av_Barclays_Ctr",
                event_time="2023-02-01 00:00:00",
                location="Street",
                event_type="Construction",
            ),
        ]

        filtered, relevance = agent.filter_events(events)

        self.assertEqual([event.title for event in filtered], ["Brooklyn Nets vs Knicks"])
        self.assertEqual(relevance[0].semantic_category, "sports")
        self.assertEqual(relevance[0].affected_channels, ["R610__Atlantic_Av_Barclays_Ctr"])
        self.assertGreater(relevance[0].relevance_score, relevance[1].relevance_score)

    def test_prediction_fusion_does_not_apply_empty_event_to_all_station_channels(self):
        from agents.prediction_fusion import PredictionFusion
        from agents.schemas import EventImpact, NumericalPrediction

        prediction = NumericalPrediction(
            forecast=[[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]],
            channel_names=["A__Alpha", "B__Beta", "C__Gamma"],
            forecast_timestamps=["2023-02-01 19:00:00", "2023-02-01 20:00:00"],
        )
        impact = EventImpact(
            event_summary="Unmapped event",
            affected_channels=[],
            impact_direction="increase",
            impact_magnitude=0.2,
            impact_duration_hours=2,
            event_time="2023-02-01 19:00:00",
        )

        result = PredictionFusion().fuse(prediction, [impact])

        self.assertEqual(result.adjusted_forecast, prediction.forecast)
        self.assertEqual(result.events_considered, [])


if __name__ == "__main__":
    unittest.main()
