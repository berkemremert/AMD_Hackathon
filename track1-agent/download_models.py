import os
from sentence_transformers import SentenceTransformer
from local_solvers import get_gliner, get_sentiment_pipeline

def download_models():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Downloading model {model_name}...")
    SentenceTransformer(model_name)
    print("Download complete.")
    
    print("Downloading GLiNER model to Docker cache...")
    get_gliner()

    print("Downloading Sentiment model to Docker cache...")
    get_sentiment_pipeline()

    print("Downloads complete and cached successfully!")

if __name__ == "__main__":
    download_models()
