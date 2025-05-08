from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
import os
import traceback
import datetime


app = Flask(__name__)


# === Config ===
TWILIO_ACCOUNT_SID = 'AC6aed04a82a0b8faced75641da9209f95'
TWILIO_AUTH_TOKEN = '71131beaa280f7f1849c0224ffdb66f8'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'

RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"
DEBUG_MODE = True

FASTAPI_URL = "http://localhost:8000"

def invoke_rasa(message_text: str, sender_id: str = "default") -> tuple[str, str | None]:
    
    rasa_response = "Sorry, I couldn't understand, can you please rephrase?"
    
    try:
        webhook_response = requests.post(
            RASA_SERVER_URL,
            json={"sender": sender_id, "message": message_text},
            timeout=60  
        )
        
        if webhook_response.ok:
            responses = [msg.get("text", "") for msg in webhook_response.json()]
            rasa_response = "\n".join(responses)  # Single newline between responses
            log_rasa_debug(f"Raw RASA responses: {responses[0]}")

    except requests.Timeout:
        rasa_response = "Request timed out. Please try again."

    except Exception as e:
        rasa_response = f"System error: {str(e)}"
        # if DEBUG_MODE:
        #     log_rasa_debug(f"RASA Error: {traceback.format_exc()}")

    return rasa_response

def log_rasa_debug(message):
    """Helper function to log debug messages"""
    if DEBUG_MODE:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"INFO RASA DEBUG: {message}")  # Also prints to console

# === Utility Functions ===
def get_sentiment(text):
    try:
        response = requests.post(f"{FASTAPI_URL}/sentiment/", json={"text": text}, timeout=60)
        data = response.json()
        label = data.get("label", "unknown")
        score = data.get("score", 0.0)
        return response.json().get("label", "unknown")
    
    except Exception as e:
        print(f"[Sentiment API Error]: {e}")
        return "error"

def get_intent(text):
    try:
        response = requests.post(f"{FASTAPI_URL}/intent/", json={"text": text}, timeout=60)
        data = response.json()
        return data.get("intent", "unknown"), data.get("confidence", 0.0)
    except Exception as e:
        print(f"[Intent API Error]: {e}")
        return "error", 0.0

def get_rasa_response(text):
    try:
        response = requests.post(f"{FASTAPI_URL}/rasa/chat", json={"text": text}, timeout=60)
        data = response.json()
        print("Rasa Response:",data)
        return data.get("response", "unknown")
    except Exception as e:
        print(f"[RASA API Error]: {e}")
        return "error", 0.0

def query_llm(prompt):

    try:
        response = requests.post(f"{FASTAPI_URL}/llm/", json={"text": prompt}, timeout=60)
        data = response.json()
        print(data)
        return data.get("response", "unknown")
    
    except Exception as e:
        print(f"[LLM API Error]: {e}")
        return "error"

# === WhatsApp Webhook ===
@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_data = request.form
    user_msg = incoming_data.get('Body', '')
    from_number = incoming_data.get('From', '')

    print(f"[Webhook] Message from {from_number}: {user_msg}")
    reply_text = "Hmm, something went wrong."

    try:
    
        reply_text =  invoke_rasa(user_msg)

        if len(reply_text) > 1600:
            print("Character limit exceeded")
        

    except Exception as e:
        print(f"[Error] Processing failed: {e}")
        traceback.print_exc()
        reply_text = f"Error: {str(e)}"

    # Send reply back via Twilio
    try:
        twilio_response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json",
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            data={
                "From": TWILIO_WHATSAPP_NUMBER,
                "To": from_number,
                "Body": reply_text
            },
            timeout=10
        )
        print(f"[Twilio] Status: {twilio_response.status_code}")
    except Exception as e:
        print(f"[Error] Failed to send Twilio message: {e}")
        traceback.print_exc()

    return Response(str(MessagingResponse()), mimetype="application/xml")

# === Run Server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)



