from transformers import pipeline

sentiment_pipe = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def analyze_sentiment(text: str):
    result = sentiment_pipe(text)[0]
    return {"label": result["label"], "score": result["score"]}