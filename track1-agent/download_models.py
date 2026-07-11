import os
from sentence_transformers import SentenceTransformer
from gliner import GLiNER

def download_models():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Downloading model {model_name}...")
    SentenceTransformer(model_name)
    print("Download complete.")
    

    print("Downloading GLiNER model to Docker cache...")
    GLiNER.from_pretrained("urchade/gliner_small-v2.1")

    print("Downloads complete and cached successfully!")

if __name__ == "__main__":
    download_models()
