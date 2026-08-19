import tempfile
import unittest
from pathlib import Path


class VisualizationManifestTests(unittest.TestCase):
    def test_manifest_records_fig1_skipped_by_request(self):
        from experiments.visualization.eaf_viz.plotting_utils import FigureManifest

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "figure_manifest.csv"
            manifest = FigureManifest()
            manifest.add(
                figure_id="fig1_framework_eaf_mas_v2",
                title="EAF-MAS v2 framework",
                input_files=[],
                output_pdf="",
                output_png="",
                required_fields=[],
                status="skipped_by_request",
                notes="Fig. 1 is intentionally skipped in this iteration.",
            )
            manifest.write(path)

            text = path.read_text(encoding="utf-8")
            self.assertIn("fig1_framework_eaf_mas_v2", text)
            self.assertIn("skipped_by_request", text)

    def test_run_all_visualizations_cleans_only_stale_reports(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "run_all_visualizations.py"
        spec = importlib.util.spec_from_file_location("run_all_visualizations", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "experiments" / "visualization" / "outputs" / "reports"
            reports.mkdir(parents=True)
            stale = reports / "top_venue_visual_experiment_plan.md"
            keep = reports / "fig10_ablation_protocol.md"
            stale.write_text("LP-MOMENT stale report", encoding="utf-8")
            keep.write_text("PT-MOMENT current report", encoding="utf-8")

            module.clean_stale_reports(root)

            self.assertFalse(stale.exists())
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
