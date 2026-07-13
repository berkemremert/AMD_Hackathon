"""Download build-time model assets into a self-contained runtime directory."""
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer


def download_models():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    destination = Path(
        os.environ.get("LOCAL_EMBEDDING_MODEL", "/models/all-MiniLM-L6-v2")
    )
    print(f"Downloading model {model_name}...")
    model = SentenceTransformer(model_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination), safe_serialization=True)
    print(f"Model saved to {destination}.")


if __name__ == "__main__":
    download_models()
