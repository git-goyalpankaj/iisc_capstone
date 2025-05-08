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

# === Config ===
TWILIO_ACCOUNT_SID = 'AC6aed04a82a0b8faced75641da9209f95'
TWILIO_AUTH_TOKEN = '71131beaa280f7f1849c0224ffdb66f8'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'

FASTAPI_URL = "http://localhost:8000"

PDF_DIR = "C:/IISc_Capstone/RASA/chatbot_api_with_rasa/PolicyDocuments/"

FAISS_INDEX_PATH = os.path.join(PDF_DIR, "faiss_index")

app = Flask(__name__)

# === Load or Build Vector DB ===


def load_or_create_vector_db():

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        print("FAISS Already Present")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

    documents = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(PDF_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))

    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

#Old Code
"""     try:
        if os.path.exists(FAISS_INDEX_DIR):
            print(f"[INFO] Loading existing FAISS index...", {FAISS_INDEX_DIR})
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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

                        if not chunks:
                raise ValueError("No documents loaded — cannot create FAISS index.")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print(f"[DEBUG] Number of chunks: {len(chunks)}")
            if not chunks:
                print("[ERROR] No document chunks were loaded. Check your document path or loader.")

            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(FAISS_INDEX_DIR)
            return db
    except Exception as e:
        print("[ERROR] Failed to load/create vector DB:", e)
        traceback.print_exc()
        raise
 """

vector_db = load_or_create_vector_db()

# === Utility Functions ===
def get_sentiment(text):
    try:
        response = requests.post(f"{FASTAPI_URL}/sentiment/", json={"text": text}, timeout=5)
        data = response.json()
        label = data.get("label", "unknown")
        score = data.get("score", 0.0)
        return response.json().get("label", "unknown")
    
    except Exception as e:
        print(f"[Sentiment API Error]: {e}")
        return "error"

def get_intent(text):
    try:
        response = requests.post(f"{FASTAPI_URL}/intent/", json={"text": text}, timeout=5)
        data = response.json()
        return data.get("intent", "unknown"), data.get("confidence", 0.0)
    except Exception as e:
        print(f"[Intent API Error]: {e}")
        return "error", 0.0

def get_rasa_response(text):
    try:
        response = requests.post(f"{FASTAPI_URL}/rasa/chat", json={"text": text}, timeout=15)
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
        # Analyze sentiment and intent
        sentiment = get_sentiment(user_msg)
        intent, confidence = get_intent(user_msg)
        reply_text = f"*Sentiment*: {sentiment} \n*Intent*: {intent} (Conf: {confidence}) \n"


        # Retrieve context
        
        matches = vector_db.max_marginal_relevance_search(user_msg, k=3)
        #print(matches)
        #retriever = vector_db.as_retriever(score_threshold=0.7)
        #matches = retriever.get_relevant_documents(user_msg)

        context = "\n\n".join([m.page_content for m in matches]) if matches else "No relevant context found."
        sources = [doc.metadata.get("source", "unknown") for doc in matches]

        prompt = f"""Answer the question based on the context below:\n\n{context}\n\nQuestion: {user_msg}"""

        # Query LLM
        reply_text = reply_text + "\n*AI Agent Reponse*:" + query_llm(prompt)

        # print("Message being sent to RASA", user_msg)
        # rasa_response = get_rasa_response(user_msg)
        # rasa_response_str =""
        # if rasa_response[0]:
        #     print(rasa_response[0])
        #     rasa_response_str = rasa_response[0]

        #reply_text = reply_text + "\n*RASA Reponse*:" + rasa_response_str 
        
        # Optional: Modify reply based on sentiment/intent
        if sentiment == "NEGATIVE":
            reply_text = "Sorry to hear that. " + reply_text
        elif intent == "greeting":
            reply_text = "Hey there! 😊 " + reply_text

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



