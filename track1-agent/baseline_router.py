"""Prompt-based routing baseline: ask an LLM whether a query is easy or hard
before answering it. This is what most "LLM routing" guides describe. Unlike
the fine-tuned DistilBERT router, every single request pays for an extra
Fireworks call just to make the routing decision - that overhead is the
whole point of comparing the two approaches in evaluate.py.
"""
import os

from fireworks_client import chat

MODEL_CHEAP = os.environ["MODEL_CHEAP"]

CLASSIFY_PROMPT = """Classify the following query as either "easy" or "hard" for an AI \
model to answer well. "Hard" means it requires multi-step reasoning, precise algorithmic \
correctness, or careful handling of subtle edge cases. "Easy" means a competent model will \
very likely get it right on the first try.

Query: {prompt}

Respond with exactly one word: easy or hard."""


def classify(prompt: str) -> dict:
    """Returns {"label": "easy"|"hard", "tokens": int}."""
    result = chat(MODEL_CHEAP, CLASSIFY_PROMPT.format(prompt=prompt), max_tokens=150, temperature=0.0)
    label = "hard" if "hard" in result["text"].lower() else "easy"
    return {"label": label, "tokens": result["total_tokens"]}
