from fastapi import FastAPI, Request
from pydantic import BaseModel
from routers import llm, sentiment, intent, action
import httpx

from routers import rasa
RASA_WEBHOOK_URL = "http://localhost:5005/webhooks/rest/webhook"

class UserMessage(BaseModel):
    sender: str
    message: str

async def chat_with_rasa(user_msg: UserMessage):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(RASA_WEBHOOK_URL, json=user_msg.dict())
            response.raise_for_status()
            return {"responses": response.json()}
        except httpx.HTTPError as e:
            return {"error": str(e)}
