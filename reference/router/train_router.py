"""Fine-tunes DistilBERT as a binary easy/hard classifier on labeled_dataset.json.

This is the piece that actually runs locally (or on AMD Developer Cloud via
ROCm) and never touches Fireworks: the router's only job is to pick which
Fireworks-hosted model answers a query, so it costs zero tokens under the
hackathon's scoring rules.

The dataset here is heavily skewed (80 easy / 3 hard out of 83 examples,
even after adversarial stress-testing gpt-oss-120b - see the session notes
in the tutorial for why). Class weighting is applied so gradient signal
from the 3 hard examples isn't drowned out, but with this few positives,
expect the model to mostly learn "predict easy" - which matches reality
for this particular model pair. Minority-class precision/recall are
reported explicitly rather than only overall accuracy, which would be
misleadingly high on a dataset this skewed.
"""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_dataset.json"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "router-distilbert"
LABEL2ID = {"easy": 0, "hard": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

random.seed(7)
torch.manual_seed(7)


class QueryDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["prompt"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(LABEL2ID[ex["label"]], dtype=torch.long),
        }


def stratified_split(records, test_hard=1, test_frac=0.2):
    """Manual split: guarantees at least `test_hard` hard examples land in
    test, since sklearn's stratified split can't handle a class with only
    a handful of members."""
    hard = [r for r in records if r["label"] == "hard"]
    easy = [r for r in records if r["label"] == "easy"]
    random.shuffle(hard)
    random.shuffle(easy)

    n_test_hard = min(test_hard, len(hard))
    test = hard[:n_test_hard]
    train = hard[n_test_hard:]

    n_test_easy = max(1, int(len(easy) * test_frac))
    test += easy[:n_test_easy]
    train += easy[n_test_easy:]

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            for p, l in zip(preds.tolist(), labels.tolist()):
                if p == 1 and l == 1:
                    tp += 1
                elif p == 1 and l == 0:
                    fp += 1
                elif p == 0 and l == 0:
                    tn += 1
                else:
                    fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy, "hard_precision": precision, "hard_recall": recall,
        "hard_f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main(epochs=6, batch_size=8, lr=2e-5):
    records = json.loads(DATA_PATH.read_text())
    train_records, test_records = stratified_split(records)
    n_hard_train = sum(1 for r in train_records if r["label"] == "hard")
    n_hard_test = sum(1 for r in test_records if r["label"] == "hard")
    print(f"train: {len(train_records)} ({n_hard_train} hard) | test: {len(test_records)} ({n_hard_test} hard)")

    device = get_device()
    print("device:", device)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    train_ds = QueryDataset(train_records, tokenizer)
    test_ds = QueryDataset(test_records, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    n_easy_train = len(train_records) - n_hard_train
    weight_hard = n_easy_train / max(n_hard_train, 1)
    class_weights = torch.tensor([1.0, weight_hard], dtype=torch.float32).to(device)
    print(f"class weights: easy=1.0, hard={weight_hard:.2f}")

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metrics = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch + 1}/{epochs} loss={total_loss / len(train_loader):.4f} "
            f"test_acc={metrics['accuracy']:.3f} hard_precision={metrics['hard_precision']:.3f} "
            f"hard_recall={metrics['hard_recall']:.3f} hard_f1={metrics['hard_f1']:.3f}"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"Saved model to {CHECKPOINT_DIR}")

    final_metrics = evaluate(model, test_loader, device)
    print("Final test metrics:", json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
