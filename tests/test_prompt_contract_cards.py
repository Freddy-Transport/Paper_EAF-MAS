import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PromptContractCardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output_root = Path(cls.tmp.name) / "prompt_cards"
        script = cls.repo_root / "experiments" / "visualize_prompt_contract_cards.py"
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("DASHSCOPE_API_KEY", None)
        env.pop("HF_TOKEN", None)
        subprocess.check_call(
            [
                sys.executable,
                str(script),
                "--repo_root",
                str(cls.repo_root),
                "--output_root",
                str(cls.output_root),
            ],
            env=env,
        )
        cls.manifest_path = cls.output_root / "reports" / "prompt_contract_manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.cards = cls.manifest["cards"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _card(self, prompt_id):
        for row in self.cards:
            if row["prompt_id"] == prompt_id:
                return row
        self.fail(f"missing card {prompt_id}")

    def _row_text(self, row):
        return json.dumps(row, ensure_ascii=False)

    def test_manifest_exists_and_contains_seven_cards(self):
        self.assertTrue(self.manifest_path.is_file())
        self.assertGreaterEqual(len(self.cards), 7)

    def test_redaction_status_passed_for_all_cards(self):
        self.assertTrue(all(row["redaction_status"] == "passed" for row in self.cards))
        audit = self.output_root / "reports" / "prompt_contract_redaction_audit.md"
        self.assertIn("status: `passed`", audit.read_text(encoding="utf-8"))

    def test_all_card_figures_exist(self):
        for row in self.cards:
            pdf = self.output_root / "figures" / f"{row['prompt_id']}.pdf"
            png = self.output_root / "figures" / f"{row['prompt_id']}.png"
            self.assertTrue(pdf.is_file(), pdf)
            self.assertTrue(png.is_file(), png)
            self.assertGreater(pdf.stat().st_size, 0)
            self.assertGreater(png.stat().st_size, 0)
        combined = self.output_root / "figures" / "prompt_contract_cards_all.pdf"
        self.assertTrue(combined.is_file())
        self.assertGreater(combined.stat().st_size, 0)

    def test_qwenplus_card_is_summary_only_and_no_urls(self):
        row = self._card("prompt_card_01_qwenplus_event_summary")
        text = self._row_text(row)
        self.assertIn("Do not return URLs", text)
        self.assertIn("summary only", text.replace("-", " "))
        self.assertIn("not citation-quality evidence", text)

    def test_evidence_auditor_scores_only_supplied_evidence(self):
        row = self._card("prompt_card_02_local_qwen_evidence_auditor")
        text = self._row_text(row)
        self.assertIn("score only supplied evidence", text)
        self.assertIn("do not browse", text)

    def test_explanation_agent_blocks_posthoc_and_enforces_abstention(self):
        row = self._card("prompt_card_03_forecast_explanation_agent")
        text = self._row_text(row)
        self.assertIn("Do not claim ground truth", text)
        self.assertIn("post-hoc metrics", text)
        self.assertIn("abstention consistency", text)

    def test_autoskill_contract_cannot_change_forecast_arrays(self):
        row = self._card("prompt_card_05_autoskill_skill_memory_contract")
        text = self._row_text(row)
        self.assertIn("use_for_prediction=false", text)
        self.assertIn("forecast_array_changed=false", text)
        self.assertFalse(row["forecast_array_can_change"])

    def test_skillbench_card_is_proxy_not_independent_judge(self):
        row = self._card("prompt_card_06_skillbench_judge_template_proxy")
        text = self._row_text(row)
        self.assertIn("proxy", text)
        self.assertIn("not formal independent LLM judge", text)

    def test_manifest_disallows_posthoc_metrics_in_prompts(self):
        for row in self.cards:
            self.assertFalse(row["uses_posthoc_metrics"], row["prompt_id"])

    def test_outputs_do_not_contain_secret_markers(self):
        forbidden = [
            "OPENAI_API_KEY",
            "DASHSCOPE_API_KEY",
            "HF_TOKEN",
            "sk-",
        ]
        for path in self.output_root.rglob("*"):
            if path.is_dir() or path.suffix.lower() in {".png", ".pdf"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                self.assertNotIn(token, text, path)
            self.assertNotRegex(text, r"Bearer\s+(?!\[REDACTED\])")

    def test_latex_snippet_and_manifest_table_exist(self):
        self.assertTrue((self.output_root / "latex" / "appendix_prompt_contract_cards.tex").is_file())
        self.assertTrue((self.output_root / "tables" / "prompt_contract_manifest.tex").is_file())


if __name__ == "__main__":
    unittest.main()
