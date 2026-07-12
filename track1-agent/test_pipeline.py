import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from output_optimizer import detect_task_type
from src.track1_agent.config import Settings, select_kimi_model
from src.track1_agent.pipeline import TaskProcessor, run


EXPECTED_TYPES = {
    "factual_knowledge": "knowledge_qa",
    "math_reasoning": "math_solving",
    "sentiment_classification": "sentiment_analysis",
    "text_summarization": "summarization",
    "named_entity_recognition": "entity_extraction",
    "code_debugging": "bug_fixing",
    "logical_reasoning": "logical_puzzles",
    "code_generation": "code_authoring",
}


class PipelineTests(unittest.TestCase):
    def test_public_categories_are_detected_without_metadata(self):
        path = Path(__file__).parent / "data" / "public_style_80_questions.json"
        tasks = json.loads(path.read_text())
        mismatches = [
            (task["id"], task["category"], detect_task_type(task["prompt"]))
            for task in tasks
            if detect_task_type(task["prompt"]) != EXPECTED_TYPES[task["category"]]
        ]
        self.assertEqual([], mismatches)

    def test_kimi_is_selected_from_allowlist(self):
        allowed = "allowed/minimax,allowed/kimi"
        with patch.dict(os.environ, {"ALLOWED_MODELS": allowed}, clear=True):
            self.assertEqual("allowed/kimi", select_kimi_model())

    def test_missing_kimi_fails_instead_of_using_another_model(self):
        with patch.dict(os.environ, {"ALLOWED_MODELS": "allowed/a,allowed/b"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "does not contain a Kimi"):
                select_kimi_model()

    def test_api_response_is_returned_without_validation_or_retry(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {
                "text": "partial response",
                "total_tokens": 17,
                "finish_reason": "length",
            }

        result = TaskProcessor("allowed/kimi", api_chat=fake_chat).process(
            "Explain how DNS works in two sentences."
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("partial response", result.answer)
        self.assertEqual("length", result.finish_reason)
        self.assertEqual(17, result.tokens)

    def test_supported_math_is_handled_locally(self):
        called = False

        def fake_chat(**kwargs):
            nonlocal called
            called = True
            return {}

        prompt = "A tank is 3/5 full and contains 120 liters. How many more liters are needed?"
        result = TaskProcessor("allowed/kimi", api_chat=fake_chat).process(prompt)
        self.assertTrue(result.source.startswith("local:"))
        self.assertIn("80 more liters", result.answer)
        self.assertFalse(called)

    def test_all_public_math_tasks_remain_local(self):
        path = Path(__file__).parent / "data" / "public_style_80_questions.json"
        tasks = [
            task
            for task in json.loads(path.read_text())
            if task["category"] == "math_reasoning"
        ]

        def unexpected_api(**kwargs):
            self.fail("A public math task unexpectedly reached the API")

        processor = TaskProcessor("allowed/kimi", api_chat=unexpected_api)
        self.assertTrue(all(processor.process(task["prompt"]).source == "local:math" for task in tasks))

    def test_end_to_end_output_matches_submission_schema(self):
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
                    {"task_id": "local", "answer": "Total capacity = 120 ÷ (3/5) = 200 liters. It needs 200 - 120 = 80 more liters."},
                    {"task_id": "api", "answer": "API answer"},
                ],
                results,
            )


if __name__ == "__main__":
    unittest.main()
