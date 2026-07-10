from local_solvers import get_gliner, get_sentiment_pipeline

print("Downloading GLiNER model to Docker cache...")
get_gliner()

print("Downloading Sentiment model to Docker cache...")
get_sentiment_pipeline()

print("Downloads complete and cached successfully!")
