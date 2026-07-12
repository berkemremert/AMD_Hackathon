"""Routing tests: category detection, local dispatch, and Kimi fallback."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from output_optimizer import detect_task_type
from src.track1_agent.config import select_kimi_model
from src.track1_agent.pipeline import TaskProcessor


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TASKS = ROOT / "data" / "input" / "public_style_80_questions.json"
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


class RouterTests(unittest.TestCase):
    def test_all_public_categories_are_detected_without_metadata(self):
        tasks = json.loads(PUBLIC_TASKS.read_text())
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

    def test_supported_math_routes_locally_without_api(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {}

        processor = TaskProcessor("allowed/kimi", api_chat=fake_chat)
        result = processor.process(
            "A tank is 3/5 full and contains 120 liters. How many more liters are needed?"
        )

        self.assertEqual("local:math", result.source)
        self.assertIn("80 more liters", result.answer)
        self.assertEqual([], calls)

    def test_unsupported_local_task_routes_to_kimi_once(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {"text": "DNS answer", "total_tokens": 11, "finish_reason": "stop"}

        result = TaskProcessor("allowed/kimi", api_chat=fake_chat).process(
            "Explain how DNS works in two sentences."
        )

        self.assertEqual("api", result.source)
        self.assertEqual("allowed/kimi", result.model)
        self.assertEqual("DNS answer", result.answer)
        self.assertEqual(1, len(calls))
        self.assertEqual("allowed/kimi", calls[0]["model"])

    def test_truncated_api_response_is_not_validated_or_retried(self):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs)
            return {
                "text": "partial response",
                "total_tokens": 17,
                "finish_reason": "length",
            }

        result = TaskProcessor("allowed/kimi", api_chat=fake_chat).process(
            "Explain how vaccination works."
        )

        self.assertEqual(1, len(calls))
        self.assertEqual("partial response", result.answer)
        self.assertEqual("length", result.finish_reason)


if __name__ == "__main__":
    unittest.main()
