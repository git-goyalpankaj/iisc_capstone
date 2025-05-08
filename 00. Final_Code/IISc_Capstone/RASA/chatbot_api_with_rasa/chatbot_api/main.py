from fastapi import FastAPI, Request
from pydantic import BaseModel
from routers import llm, sentiment, intent, action, rasa, rag
import httpx

app = FastAPI(title="Chatbot API")

app.include_router(llm.router, prefix="/llm", tags=["LLM"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment"])
app.include_router(intent.router, prefix="/intent", tags=["Intent"])
app.include_router(action.router, prefix="/action", tags=["Action"])
app.include_router(rasa.router, prefix="/rasa", tags=["Rasa"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])

@app.get("/health")
def health():
    return {"status": "OK"}


