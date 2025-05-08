# rag_service.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import requests
import os
import csv
import traceback
from typing import List

PDF_DIR = "C:/IISc_Capstone/RASA/chatbot_api_with_rasa/PolicyDocuments/"
FAISS_INDEX_PATH = os.path.join(PDF_DIR, "faiss_index")

SCHEDULE_FILE_PATH="C:/IISc_Capstone/RASA/chatbot_api_with_rasa/PolicyDocuments/skywings_flight_schedule.txt"
SCHEDULE_INDEX_PATH = os.path.join(PDF_DIR, "schedule_faiss_index")

FASTAPI_LLM_URL = "http://localhost:8000/llm"  # update as needed


# Load embeddings 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_or_create_vector_db():
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

def load_create_schedule_index():
    if os.path.exists(SCHEDULE_INDEX_PATH):
        print("SCHEDULE Already Present")
        return FAISS.load_local(SCHEDULE_INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)

    schedules = []
    with open(SCHEDULE_FILE_PATH, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = (
                f"Flight {row['Flight Number']} departs from {row['Origin']} to {row['Destination']} "
                f"at {row['Departure(LT)']} and arrives at {row['Arrival(LT)']}. "
                f"It is a {row['Flight Type']} flight operating on {row['Days of Operation']}."
            )
            schedules.append(Document(page_content=content))

    sch_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_schedule = FAISS.from_documents(schedules,sch_embedding_model)
    vectorstore_schedule.save_local(SCHEDULE_INDEX_PATH)


def query_llm(prompt: str) -> str:
    try:
        response = requests.post(f"{FASTAPI_LLM_URL}/", json={"text": prompt}, timeout=60)
        data = response.json()
        return data.get("response", "unknown")
    except Exception as e:
        print(f"[LLM API Error]: {e}")
        return "error"

def run_rag_pipeline(question: str) -> dict:

    policy_store = load_or_create_vector_db()
    policy_retriever = policy_store.as_retriever(score_threshold=0.7)
    policy_matches = policy_retriever.get_relevant_documents(question)
    if not policy_matches:
        return {"response": "No relevant policies found.", "sources": []}

    schedule_store = load_create_schedule_index()
    schedule_retriever = schedule_store.as_retriever(score_threshold=0.7)
    schedule_matches = schedule_retriever.get_relevant_documents(question)
    if not schedule_matches:
        return {"response": "No relevant schedules found.", "sources": []}

    context = "\n\n".join([f"{doc.page_content}" for doc in policy_matches + schedule_matches])
    sources = [doc.metadata.get("source", "unknown") for doc in policy_matches]

    prompt = f"""
    Use the following context to answer the question. If the answer isn't in the context, say "I don't know".

    Context:
    {context}

    Question: {question}
    """

    llm_response = query_llm(prompt)
    return {"response": llm_response, "sources": sources}
