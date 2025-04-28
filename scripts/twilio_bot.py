from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import os
import traceback

# === Config ===
TWILIO_ACCOUNT_SID = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
TWILIO_AUTH_TOKEN = '7XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+1XXXXXXXXXXXXXXX'


OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

PDF_DIR = "/Users/amitdutta/Documents/Indigo"
FAISS_INDEX_DIR = "faiss_index"

app = Flask(__name__)

# === Load or Build Vector DB ===
def load_or_create_vector_db():
    try:
        if os.path.exists(FAISS_INDEX_DIR):
            print("[INFO] Loading existing FAISS index...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            #return FAISS.load_local(FAISS_INDEX_DIR, embeddings)
            return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        else:
            print("[INFO] Creating FAISS index from PDFs...")
            all_docs = []
            for file in os.listdir(PDF_DIR):
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(PDF_DIR, file))
                    docs = loader.load()
                    all_docs.extend(docs)

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(FAISS_INDEX_DIR)
            return db
    except Exception as e:
        print("[ERROR] Failed to load/create vector DB:", e)
        traceback.print_exc()
        raise

vector_db = load_or_create_vector_db()

# === WhatsApp Webhook ===
@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_data = request.form
    user_msg = incoming_data.get('Body', '')
    from_number = incoming_data.get('From', '')

    print(f"[Webhook] Message from {from_number}: {user_msg}")

    reply_text = "Hmm, something went wrong."

    try:
        # Retrieve relevant context from vector DB
        matches = vector_db.similarity_search(user_msg, k=3)
        context = "\n\n".join([m.page_content for m in matches]) if matches else "No relevant context found."

        prompt = f"""Answer the question based on the context below:\n\n{context}\n\nQuestion: {user_msg}"""

        # Call Ollama
        ollama_response = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )

        print(f"[Ollama] Status: {ollama_response.status_code}")
        print(f"[Ollama] Response: {ollama_response.text}")

        if ollama_response.ok:
            reply_text = ollama_response.json().get("response", "No reply from model.")
        else:
            reply_text = f"Ollama error: {ollama_response.status_code}"

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

