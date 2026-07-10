import os
from sentence_transformers import SentenceTransformer
from local_solvers import get_sentiment_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_models():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Downloading model {model_name}...")
    SentenceTransformer(model_name)
    print("Download complete.")
    
    print("Downloading Sentiment model to Docker cache...")
    get_sentiment_pipeline()

    print("Downloading Qwen model to Docker cache...")
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")

    print("Downloads complete and cached successfully!")

if __name__ == "__main__":
    download_models()
