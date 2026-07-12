"""End-to-end tests for the official input/output contract."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track1_agent.config import Settings
from src.track1_agent.pipeline import TaskProcessor, run


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TASKS = ROOT / "data" / "input" / "public_style_80_questions.json"


class PipelineTests(unittest.TestCase):
    def test_all_public_math_tasks_remain_local(self):
        tasks = [
            task
            for task in json.loads(PUBLIC_TASKS.read_text())
            if task["category"] == "math_reasoning"
        ]

        def unexpected_api(**kwargs):
            self.fail("A public math task unexpectedly reached the API")

        processor = TaskProcessor("allowed/kimi", api_chat=unexpected_api)
        self.assertTrue(
            all(processor.process(task["prompt"]).source == "local:math" for task in tasks)
        )

    def test_output_matches_submission_schema(self):
        def fake_chat(**kwargs):
            return {"text": "API answer", "total_tokens": 9, "finish_reason": "stop"}

        tasks = [
            {"task_id": "local", "prompt": "A tank is 3/5 full and contains 120 liters."},
            {"task_id": "api", "prompt": "Explain how DNS works."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text(json.dumps(tasks))
            settings = Settings(input_path, output_path, "allowed/kimi")
            results = run(settings, TaskProcessor("allowed/kimi", api_chat=fake_chat))

            self.assertEqual(results, json.loads(output_path.read_text()))
            self.assertEqual(
                [
                    {
                        "task_id": "local",
                        "answer": (
                            "Total capacity = 120 ÷ (3/5) = 200 liters. "
                            "It needs 200 - 120 = 80 more liters."
                        ),
                    },
                    {"task_id": "api", "answer": "API answer"},
                ],
                results,
            )

    def test_invalid_input_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tasks.json"
            output_path = root / "results.json"
            input_path.write_text(json.dumps([{"task_id": "missing-prompt"}]))
            settings = Settings(input_path, output_path, "allowed/kimi")

            with self.assertRaisesRegex(ValueError, "must contain task_id and prompt"):
                run(settings, TaskProcessor("allowed/kimi", api_chat=lambda **kwargs: {}))


if __name__ == "__main__":
    unittest.main()
