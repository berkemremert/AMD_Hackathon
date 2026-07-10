from local_solvers import get_sentiment_pipeline

print("Downloading Sentiment model to Docker cache...")
get_sentiment_pipeline()

print("Downloads complete and cached successfully!")
