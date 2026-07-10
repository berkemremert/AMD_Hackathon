import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from classifier import Category, classify


class TestClassify(unittest.TestCase):
    CASES = [
        ("What is the capital of Australia?", Category.FACTUAL),
        ("A store has 240 items. It sells 15% on Monday. How many remain?", Category.MATH),
        ("Classify the sentiment of this review: the food was cold.", Category.SENTIMENT),
        ("Summarize the following text in one sentence: ...", Category.SUMMARIZATION),
        ("Extract all named entities (person, organization, location, date) from: ...", Category.NER),
        ("Fix the bug in this function so it returns the average.", Category.CODE_DEBUG),
        ("Write a Python function that reverses a string.", Category.CODE_GEN),
        ("Three friends each own a different pet. Who owns the bird?", Category.LOGIC),
    ]

    def test_categories(self):
        for prompt, expected in self.CASES:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify(prompt), expected)

    def test_never_raises_on_empty(self):
        self.assertIsInstance(classify(""), Category)
        self.assertIsInstance(classify("   "), Category)


if __name__ == "__main__":
    unittest.main()
