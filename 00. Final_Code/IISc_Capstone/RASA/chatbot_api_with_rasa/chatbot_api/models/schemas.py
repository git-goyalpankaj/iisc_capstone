from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class LLMResponse(BaseModel):
    response: str

class SentimentResponse(BaseModel):
    label: str
    score: float

class IntentResponse(BaseModel):
    intent: str
    confidence: float