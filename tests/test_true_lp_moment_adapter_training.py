import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class FakePrediction:
    def __init__(self, forecast, ground_truth, forecast_timestamps):
        self.forecast = forecast
        self.ground_truth = ground_truth
        self.forecast_timestamps = forecast_timestamps


class FakeNumericalAgent:
    def __init__(self, n_channels=2, horizon=3):
        self.n_channels = n_channels
        self.horizon = horizon
        self.calls = []

    def predict_for_date(self, data_path, target_date, forecast_horizon=None):
        h = int(forecast_horizon or self.horizon)
        self.calls.append((str(data_path), str(target_date), h))
        forecast = np.arange(self.n_channels * h, dtype=np.float32).reshape(self.n_channels, h) + 10.0
        actual = forecast + 1.0
        timestamps = [str(pd.Timestamp(target_date) + pd.Timedelta(hours=i)) for i in range(h)]
        return FakePrediction(forecast.tolist(), actual.tolist(), timestamps)


class ShortHorizonAgent(FakeNumericalAgent):
    def predict_for_date(self, data_path, target_date, forecast_horizon=None):
        pred = super().predict_for_date(data_path, target_date, forecast_horizon=forecast_horizon)
        pred.forecast = [row[:-1] for row in pred.forecast]
        pred.ground_truth = [row[:-1] for row in pred.ground_truth]
        pred.forecast_timestamps = pred.forecast_timestamps[:-1]
        return pred


def write_tiny_traffic(path: Path, n_rows: int = 40) -> None:
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='h')
    pd.DataFrame({
        'date': dates,
        'a': np.arange(n_rows, dtype=np.float32),
        'b': np.arange(n_rows, dtype=np.float32) + 100,
    }).to_csv(path, index=False)


class TrueLpMomentAdapterTrainingTests(unittest.TestCase):
    def test_exporter_writes_complete_long_format_predictions(self):
        from event_post_training.lp_prediction_exporter import export_lp_moment_predictions

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            traffic = root / 'traffic.csv'
            out_csv = root / 'predictions.csv'
            write_tiny_traffic(traffic)
            anchors = [10, 13]
            agent = FakeNumericalAgent(n_channels=2, horizon=3)

            df, manifest = export_lp_moment_predictions(
                agent=agent,
                traffic_csv=traffic,
                channel_names=['a', 'b'],
                anchors=[{'split': 'train', 'anchor': anchors[0]}, {'split': 'val', 'anchor': anchors[1]}],
                horizon=3,
                out_csv=out_csv,
            )

            self.assertEqual(len(df), 2 * 2 * 3)
            self.assertEqual(set(df.columns), {
                'split', 'anchor', 'date', 'channel_name', 'channel_idx', 'horizon_idx', 'timestamp', 'moment_pred', 'actual'
            })
            self.assertEqual(set(df['split']), {'train', 'val'})
            self.assertTrue(out_csv.is_file())
            self.assertEqual(manifest['prediction_source'], 'real_lp_moment')
            self.assertEqual(manifest['row_count'], 12)
            first = df.sort_values(['anchor', 'channel_idx', 'horizon_idx']).iloc[0].to_dict()
            self.assertEqual(first['anchor'], 10)
            self.assertEqual(first['channel_name'], 'a')
            self.assertEqual(first['horizon_idx'], 0)
            self.assertEqual(first['moment_pred'], 10.0)
            self.assertEqual(first['actual'], 11.0)

    def test_exporter_refuses_short_horizon_predictions(self):
        from event_post_training.lp_prediction_exporter import export_lp_moment_predictions

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            traffic = root / 'traffic.csv'
            write_tiny_traffic(traffic)
            with self.assertRaisesRegex(ValueError, 'prediction horizon too short'):
                export_lp_moment_predictions(
                    agent=ShortHorizonAgent(n_channels=2, horizon=3),
                    traffic_csv=traffic,
                    channel_names=['a', 'b'],
                    anchors=[{'split': 'train', 'anchor': 10}],
                    horizon=3,
                    out_csv=root / 'predictions.csv',
                )

    def test_strict_prediction_csv_rejects_missing_rows_and_excludes_test(self):
        from event_post_training.sample_builder import build_training_arrays

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            traffic = root / 'traffic.csv'
            events = root / 'events.json'
            channel_map = root / 'channel_map.json'
            preds = root / 'preds.csv'
            write_tiny_traffic(traffic, n_rows=40)
            events.write_text('[]', encoding='utf-8')
            channel_map.write_text('{"channels":[{"channel_name":"a"},{"channel_name":"b"}]}', encoding='utf-8')
            pd.DataFrame([
                {'anchor': 8, 'channel_name': 'a', 'horizon_idx': 0, 'moment_pred': 1.0},
            ]).to_csv(preds, index=False)

            with self.assertRaisesRegex(ValueError, 'Missing complete moment_predictions_csv'):
                build_training_arrays(
                    traffic, events, channel_map,
                    seq_len=4, horizon=2, train_rows=16, val_rows=8, residual_bound=0.05,
                    stride=4, negative_ratio=1.0, max_windows=3,
                    moment_predictions_csv=preds,
                    splits=('train', 'val'),
                    strict_prediction_csv=True,
                    allow_deterministic_fallback=False,
                )

    def test_train_cli_rejects_peft_and_requires_real_predictions(self):
        import experiments.train_event_aware_posttrainer as trainer_cli

        peft_args = argparse.Namespace(mode='peft_moment', moment_predictions_csv='preds.csv', generate_moment_predictions=False, allow_deterministic_fallback=False)
        with self.assertRaisesRegex(SystemExit, 'PEFT-head training is disabled'):
            trainer_cli.validate_prediction_policy(peft_args)

        missing_args = argparse.Namespace(mode='frozen_moment', moment_predictions_csv=None, generate_moment_predictions=False, allow_deterministic_fallback=False)
        with self.assertRaisesRegex(SystemExit, 'requires --moment_predictions_csv or --generate_moment_predictions'):
            trainer_cli.validate_prediction_policy(missing_args)


if __name__ == '__main__':
    unittest.main()
