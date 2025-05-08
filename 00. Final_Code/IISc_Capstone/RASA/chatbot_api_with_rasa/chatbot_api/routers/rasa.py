from fastapi import APIRouter
from models.schemas import TextInput
import requests

router = APIRouter()
RASA_SERVER_URL = "http://localhost:5005"

@router.post("/chat")
def rasa_chat(input: TextInput):
    try:
        # Step 1: Get bot response from RASA
        response = requests.post(
            f"{RASA_SERVER_URL}/webhooks/rest/webhook",
            json={"sender": "default", "message": input.text},
            timeout=30
        )
        rasa_response = [msg.get("text", "") for msg in response.json()]
        
        # Step 2: Get detected intent
        intent_resp = requests.post(
            f"{RASA_SERVER_URL}/model/parse",
            json={"text": input.text},
            timeout=5
        )
        intent_data = intent_resp.json().get("intent", {})

        return {
            "response": rasa_response,
            "intent": intent_data.get("name"),
            "confidence": intent_data.get("confidence")
        }
    except Exception as e:
        return {"error": str(e)}