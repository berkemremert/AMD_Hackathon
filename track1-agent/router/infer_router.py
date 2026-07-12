"""Loads the fine-tuned DistilBERT router and predicts easy/hard for a prompt.

This never calls Fireworks - it's a local forward pass, so it costs zero
tokens under the hackathon's scoring rules. That's the entire point of
fine-tuning a router instead of asking an LLM to classify difficulty.
"""
from pathlib import Path

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "router-distilbert"

_model = None
_tokenizer = None
_device = None


def _load():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    _device = torch.device("cpu")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(_device)
    _model.eval()


def predict(prompt: str) -> str:
    """Returns "easy" or "hard"."""
    _load()
    enc = _tokenizer(prompt, truncation=True, padding=True, max_length=256, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = _model(**enc).logits
    label_id = logits.argmax(dim=-1).item()
    return "hard" if label_id == 1 else "easy"


if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1] if len(sys.argv) > 1 else "What is 2 + 2?"))
