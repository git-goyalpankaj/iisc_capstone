# iisc_capstone Project Repository

This repository contains the implementation of a multi-dialog chatbot integrated with FAISS-based RAG (Retrieval-Augmented Generation), custom intent BERT model for intent detection and local LLM (Mistral) for llm response.

## IMPORTANT ##
This streamlit app can be run for LLM based Question and Answer currently. 01-May-2025.
App also calls RASA Server for multi-turn demo for 1 scenario. However, that flow is not fully tested / working as on 01-May-25. 

## 📂 Folder Structure

- `data/`
  - Contains source data files, embeddings, or datasets used in the project.

- `faiss_index/`
  - Stores FAISS index files generated for similarity search.

- `scripts/`
  - Contains Python utility scripts and supporting modules used by the chatbot.

- 'custom_intent'
  - Custom intent classifier

- `chatbot_app_v8_multidialog_Llm_SkyRAG_RASA_flow.py`
  - Main Python application file for the chatbot system.


## 📌 How to Run

1. Install required packages given below (if available).


PIP package dependencies for Streamlit file "chatbot_app_v8_multidialog_Llm_SkyRAG_RASA_flow.py"
pip install streamlit
pip install torch
pip install transformers
pip install langchain
pip install langchain-community
pip install huggingface-hub
pip install nest_asyncio
pip install numpy
pip install joblib
pip install requests
pip install faiss-cpu  # For FAISS vectorstore

One Command
- pip install streamlit torch transformers langchain langchain-community huggingface-hub nest_asyncio numpy joblib requests faiss-cpu

2. Make sure the FAISS index files are generated and present in `faiss_index/`.
3. Download Custom Intent BERT model from drive that Aditi has published in GitHub repo. 
4. Update all local PATHS in streamlit run chatbot_app_v8_multidialog_Llm_SkyRAG_RASA_flow.py file

   FAISS_INDEX_PATH = "D:/03-Ollama/Dummy Setup chat LLM bot/faiss_index"
   model_wrapper = ModelWrapper(r"D:\03-Ollama\Dummy Setup chat LLM bot\custom_intent")
5. Make sure you update code for which llm to call 
   llm = Ollama(model="mistral")

6. Run the main chatbot script with command "streamlit run chatbot_app_v8_multidialog_Llm_SkyRAG_RASA_flow.py"
