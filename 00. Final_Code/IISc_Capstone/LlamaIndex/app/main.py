# app/main.py
from fastapi import FastAPI, Depends, Cookie, Response
from pydantic import BaseModel
import uuid
from .orchestrator.chatbot_engine import ChatbotEngine

app = FastAPI()

@app.get("/", tags=["health"])
def read_root():
    return {"status": "ok", "message": "Chatbot API is running"}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    session_id: str

def get_session_id(session_id: str = Cookie(None), response: Response = None):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)
    return session_id

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, response: Response, session_id: str = Depends(get_session_id)):
    engine = ChatbotEngine(session_id)
    reply = engine.step(req.message)
    return ChatResponse(reply=reply, session_id=session_id)
