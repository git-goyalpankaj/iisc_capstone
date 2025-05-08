from fastapi import APIRouter
from models.schemas import TextInput, IntentResponse
from services.intent_service import intent_model

router = APIRouter()

@router.post("/", response_model=IntentResponse)
def intent_route(input: TextInput):
    return intent_model.predict(input.text)