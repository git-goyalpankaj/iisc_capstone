from fastapi import APIRouter
from models.schemas import TextInput, SentimentResponse
from services.sentiment_service import analyze_sentiment

router = APIRouter()

@router.post("/", response_model=SentimentResponse)
def sentiment_route(input: TextInput):
    return analyze_sentiment(input.text)