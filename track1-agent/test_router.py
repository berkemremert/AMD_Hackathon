import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from output_optimizer import detect_task_type
from router.infer_router import predict
from router.model_selection import resolve_model_roles


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
    def test_public_style_categories_do_not_need_metadata(self):
        path = Path(__file__).parent / "data" / "public_style_80_questions.json"
        tasks = json.loads(path.read_text())
        mismatches = [
            (task["id"], task["category"], detect_task_type(task["prompt"]))
            for task in tasks
            if detect_task_type(task["prompt"]) != EXPECTED_TYPES[task["category"]]
        ]
        self.assertEqual([], mismatches)

    def test_simple_task_stays_on_efficient_model(self):
        self.assertEqual("easy", predict("Explain how DNS works in two sentences."))

    def test_dense_concurrent_code_task_escalates(self):
        prompt = (
            "Analyze this production-ready thread-safe cache for every race condition, "
            "deadlock, memory leak, exception-safety issue, and edge case. "
            "Provide a corrected implementation and justify every change.\n"
            + "```python\nclass Cache:\n    pass\n```\n"
            + "1. Handle concurrent readers.\n2. Handle concurrent writers.\n"
            + "3. Ensure bounded memory.\n4. Preserve exception safety.\n"
            + "5. Avoid deadlocks.\n6. Use monotonic time.\n"
        )
        self.assertEqual("hard", predict(prompt))

    def test_long_non_code_task_does_not_escalate_without_evidence(self):
        prompt = "Explain entropy carefully. " + "Include relevant context. " * 150
        self.assertEqual("easy", predict(prompt))

    def test_model_roles_come_from_allowlist(self):
        allowed = "accounts/fireworks/models/kimi-test,accounts/fireworks/models/minimax-test"
        with patch.dict(os.environ, {"ALLOWED_MODELS": allowed}, clear=True):
            efficient, escalation = resolve_model_roles()
        self.assertEqual("accounts/fireworks/models/kimi-test", efficient)
        self.assertEqual("accounts/fireworks/models/kimi-test", escalation)

    def test_unallowed_overrides_are_ignored(self):
        env = {
            "ALLOWED_MODELS": "allowed/minimax,allowed/kimi",
            "MODEL_CHEAP": "not-allowed/cheap",
            "MODEL_EXPENSIVE": "not-allowed/expensive",
        }
        with patch.dict(os.environ, env, clear=True):
            efficient, escalation = resolve_model_roles()
        self.assertEqual("allowed/kimi", efficient)
        self.assertEqual("allowed/kimi", escalation)

    def test_allowlist_without_kimi_fails_instead_of_using_another_model(self):
        allowed = "allowed/model-a,allowed/model-b"
        with patch.dict(os.environ, {"ALLOWED_MODELS": allowed}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "does not contain a Kimi"):
                resolve_model_roles()


if __name__ == "__main__":
    unittest.main()
