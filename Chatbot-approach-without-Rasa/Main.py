import os
from pathlib import Path

# ----------------------------
# ENVIRONMENT & CACHE SETUP
# ----------------------------

# 1. Set ALL possible cache locations (new HuggingFace versions need this)
cache_dir = Path("D:/huggingface_cache")
cache_dir.mkdir(exist_ok=True, parents=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)

# Configure paths
os.environ["HF_HOME"] = "D:/huggingface_cache"  # Set before loading pipeline
os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'  # Stores Ollama models here


# 2. Patch the cache before any imports (critical!)
import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_CACHE = str(cache_dir)

# Now proceed with necessary imports

# Standard libraries
import asyncio
import nest_asyncio
import pickle
import torch
import string
import base64
import random
import time

# Hugging Face and Transformers
from transformers import logging
from transformers import BertForSequenceClassification, BertTokenizer

# LangChain and other libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory

# LangChain Community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

# Pipeline
from transformers import pipeline

# Streamlit
import streamlit as st

# ----------------------------
# STREAMLIT UI CONFIGURATION
# ----------------------------

# Set page title and layout
st.set_page_config(page_title="SkyWings Airline Assistant ✈️", layout="wide")

# Custom CSS to change font, size, color, and center alignment
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;  /* Full viewport height */
            text-align: center;
        }
        .title {
            font-size: 48px;
            color: #000080;  /* Navy Blue */
            font-family: 'Quicksand', sans-serif;
            margin-bottom: 10px;  /* Reduced space between lines */
        }
        .header {
            font-family: 'Quicksand', sans-serif;
            font-size: 30px;
            color: #DC143C;  /* Rich Crimson (Magenta-ish) */
            margin-bottom: 8px;  /* Reduced space between lines */
        }
        .body {
            font-family: 'Quicksand', sans-serif;
            font-size: 18px;
            color: #000000;  /* Black */
            margin-top: 5px;  /* Reduced space between lines */
        }
    </style>
""", unsafe_allow_html=True)

# Centered container for the content
st.markdown('<div class="container">', unsafe_allow_html=True)

# Title with custom font and color
st.markdown('<p class="title">SkyWings Airline Assistant ✈️</p>', unsafe_allow_html=True)

# Welcome header with custom color
st.markdown('<p class="header">Welcome to SkyWings</p>', unsafe_allow_html=True)

# Body text with black color
st.markdown('<p class="body">How can we assist you today?</p>', unsafe_allow_html=True)

# Close the container
st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------
# BACKEND
# ----------------------------

# Fix Streamlit event loop
nest_asyncio.apply()
logging.set_verbosity_error()  # Reduce warnings

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = Ollama(model="mistral", temperature = 0.8)  # Updated Ollama class

# Save the FAISS vectorstore to disk
def save_vectorstore(vectordb):
    vectordb.save_local("faiss_vectorstore")
    st.sidebar.success("✅ FAISS Vectorstore saved.")

# Load FAISS vectorstore from disk
def load_vectorstore_from_disk():
    if os.path.exists("faiss_vectorstore"):
        embedding_model = embeddings  # Use the same embedding model
        return FAISS.load_local(
            "faiss_vectorstore",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        st.sidebar.warning("No existing FAISS vectorstore found.")
        return None

# Create and save a new vectorstore from documents in a directory
def create_and_save_vectorstore(directory_path):
    docs = []
    loader = TextLoader('')
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for filepath in Path(directory_path).rglob('*.txt'):
        loader.file_path = str(filepath)
        document = loader.load()
        split_docs = splitter.split_documents(document)
        for doc in split_docs:
            doc.metadata["source"] = filepath.name
        docs.extend(split_docs)

    vectordb = FAISS.from_documents(docs, embeddings)
    save_vectorstore(vectordb)
    return vectordb

# Load model with neutral detection (3-class)
sentiment_pipe = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1
)

# BERT Intent Classification function
def classify_intent(query, model_path, label_encoder_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Force the use of CPU
    device = torch.device("cpu")  # Forces use of CPU
    model.to(device)

    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)

    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        intent = le.inverse_transform([predicted_class])[0]

    return intent

# Function to run sentiment and intent classification in parallel
async def get_sentiment_and_intent(user_input):
    sentiment_future = asyncio.to_thread(sentiment_pipe, user_input)
    intent_future = asyncio.to_thread(classify_intent, user_input, intent_model_path, intent_label_encoder_path)
    
    sentiment_result, intent_label = await asyncio.gather(sentiment_future, intent_future)
    return sentiment_result, intent_label

# Get the list of possible phrases from a pre-defined list
def load_phrases_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Clean the user's phrase by removing punctuations and converting it to lowercase
def clean_phrase(user_input):
    # Remove punctuation from input and convert to lowercase
    return user_input.translate(str.maketrans("", "", string.punctuation)).strip().lower()

# Greeting detection function
def is_small_talk_or_greeting(user_input):
    small_talk_phrases = load_phrases_from_file('user_greetings_farewells/greetings_small_talk_phrases.txt')
    # Clean the user input
    user_input_clean = clean_phrase(user_input)
    
    # Clean the small talk phrases and check if any match
    small_talk_phrases_cleaned = [clean_phrase(phrase) for phrase in small_talk_phrases]
    
    # Check if any small talk or greeting phrase matches the cleaned user input
    return user_input_clean in small_talk_phrases_cleaned

# Farewell detection function
def is_farewell(user_input):
    farewell_phrases = load_phrases_from_file('user_greetings_farewells/farewell_phrases.txt')
    
    # Clean the user input
    user_input_clean = clean_phrase(user_input)
    
    # Clean the farewell phrases and check if any match
    farewell_phrases_cleaned = [clean_phrase(phrase) for phrase in farewell_phrases]
    
    # Check if any farewell phrase matches the cleaned user input
    return user_input_clean in farewell_phrases_cleaned

# Filler word detection function
def is_filler(user_input):
    filler_phrases = load_phrases_from_file('user_greetings_farewells/filler_phrases.txt')
    
    # Clean the user input
    user_input_clean = clean_phrase(user_input)
    
    # Clean the farewell phrases and check if any match
    filler_phrases_cleaned = [clean_phrase(phrase) for phrase in filler_phrases]
    
    # Check if any farewell phrase matches the cleaned user input
    return user_input_clean in filler_phrases_cleaned


greetings_responses = load_phrases_from_file("assistant_responses/greetings_responses.txt")
farewell_responses = load_phrases_from_file("assistant_responses/farewell_responses.txt")
filler_responses = load_phrases_from_file("assistant_responses/filler_responses.txt")

# Load zero-shot classifier
sub_intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sub-intent labels
sub_intents = [
    "Booking Request",
    "Cancellation Request",
    "Modification Request",
    "Policy Question"
]

def classify_sub_intent(user_query):
    result = sub_intent_classifier(user_query, sub_intents, multi_label=False)
    return result['labels'][0], result['scores'][0]

# Assistant's polite response generator
def generate_assistant_response(intent, user_query):
    special_responses = {
        "Other": (
            "I'm not sure about that, but I can help with other queries! "
            "For assistance with this matter, you can reach us at Tel: +91 (0)124 4973838, "
            "or email customer-support@skywings.com. You can also reach out to us on Facebook (https://www.facebook.com/goSkyWings.in) "
            "or Twitter (@SkyWings6E) for feedback, concerns, or comments."
        ),
        "Irrelevant": "Sorry, that's out of my scope. Could you please refine your question?",
    }

    if intent in special_responses:
        return special_responses[intent]
    else:
        return "vector_db"  # Signal to use vector retrieval for actual answer

# ----------------------------
# Embedding & Retrieval
# ----------------------------

# Load or create vector store
vectordb = load_vectorstore_from_disk()

if not vectordb:
    directory_path_policies = "/Users/aditiravishankar/Desktop/Capstone/28 APRIL iisc_capstone-main/SkyWings Policy Document"
    # Fallback: Create a new index from sample documents if no vectorstore is found
    vectordb = create_and_save_vectorstore(directory_path_policies)
    
retriever = vectordb.as_retriever()
memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# ----------------------------
# FRONTEND
# ----------------------------

# Set background image function
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
# Function for simulating token-by-token response with typing effect
def stream_response_tokens(response):
    """Simulate real-time token generation."""
    for word in response.split():
        yield word + " "  # Return one word at a time
        time.sleep(0.1)  # Simulate a delay between tokens to create the typing effect

def display_typing_effect(placeholder, response_text):
    """Display assistant message with typing effect using a Streamlit placeholder."""
    full_response = ""
    for token in stream_response_tokens(response_text):
        full_response += token
        placeholder.markdown(
            f"""
            <style>
            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0; }}
                100% {{ opacity: 1; }}
            }}
            .blinking-cursor {{
                animation: blink 1s step-start infinite;
                display: inline;
            }}
            </style>
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 60%; text-align: right;'>
                    <strong>✈️ SkyWings Assistant</strong><br>
                    <span>{full_response}<span class='blinking-cursor'>|</span></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Final display without blinking cursor
        response_placeholder.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end;'>
                <div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 60%; text-align: right;'>
                    <strong>✈️ SkyWings Assistant</strong><br>
                    <span>{full_response}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
def display_message(msg, show_analysis=False):
    """Helper function to display messages with proper formatting"""
    avatars = {"Customer": "🙋", "Assistant": "✈️"}
    colors = {"Customer": "#DCF8C6", "Assistant": "#F1F0F0"}
    
    if msg['role'] == "Customer":
        st.markdown(
            f"""
            <div style='display: flex; gap: 8px; margin-bottom: 10px;'>
                <div style='background-color: {colors['Customer']}; padding: 10px; border-radius: 10px; max-width: 45%; text-align: left;'>
                    <strong>{avatars['Customer']} Customer</strong><br>
                    <span>{msg['content']}</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        
        if show_analysis and 'sentiment' in msg:
            st.markdown(
                f"""
                <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-size: 14px; max-width: 35%; min-width: 150px;'>
                    <strong>🧠 Sentiment:</strong> {msg.get('sentiment', '')}<br>
                    <strong>🎯 Intent:</strong> {msg.get('intent', '')} {f"({msg.get('score', '')}%)" if msg.get('score') else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    elif msg['role'] == "Assistant":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                <div style='background-color: {colors['Assistant']}; padding: 10px; border-radius: 10px; max-width: 60%; text-align: right;'>
                    <strong>{avatars['Assistant']}✈️ SkyWings Assistant</strong><br>
                    <span>{msg['content']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Sidebar: Controls
# ----------------------------
st.sidebar.button("🪠 Clear Chat", on_click=lambda: st.session_state.update(
    messages=[],
    display_stage=0,
    current_message=None
))

# ----------------------------
# Session Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = None

# ----------------------------
# Main Chat Interface
# ----------------------------

# Background image for the UI
add_bg_from_local("Images/Background1.jpg")  # Update with the correct path to your image

# Display previous messages
for msg in st.session_state.messages:
    display_message(msg, show_analysis=(msg['role'] == 'Customer'))

# Define paths for intent classification model and label encoder file
intent_model_path = "/Users/aditiravishankar/Desktop/Capstone/Best Bert Model/Best_Bert_Model"
intent_label_encoder_path = "/Users/aditiravishankar/Desktop/Capstone/Best Bert Model/Best_Bert_Model/label_encoder.pkl"

# Handle new user input
user_input = st.chat_input("Say something...")
if user_input and st.session_state.display_stage == 0:
    # Process sentiment and intent first before adding user message
    sentiment_result, intent_label = asyncio.run(get_sentiment_and_intent(user_input))
    sentiment_label = sentiment_result[0]['label']
    sentiment_emoji = {"POSITIVE": "😄", "NEGATIVE": "😞", "NEUTRAL": "😐"}.get(sentiment_label.upper(), "💬")


    # Greetings logic block
    if is_small_talk_or_greeting(user_input):
        assistant_response = random.choice(greetings_responses)
        intent_label = "Greeting / Small Talk"

    elif is_farewell(user_input):
        assistant_response = random.choice(farewell_responses)
        intent_label = "Farewell"
    
    elif is_filler(user_input):
        assistant_response = random.choice(filler_responses)
        intent_label = "Filler Words"

    else:
        intent_label = classify_intent(user_input, intent_model_path, intent_label_encoder_path)
        print("Got a non-greeting intent")
        if intent_label == "Booking, Modifications And Cancellations":
            print("Got a Booking, Modifications And Cancellations intent")
            sub_intent, score = classify_sub_intent(user_input)
            print("Got subintent = ", sub_intent)
            print("Got score = ", score)
            if sub_intent == "Booking Request" and score >= 0.5:
                print("inside booking request action")
                assistant_response = "[Trigger booking request]"

            elif sub_intent == "Cancellation Request" and score >= 0.5:
                print("inside cancellation request action")
                assistant_response = "[Trigger cancellation request]"
            else:
                print("inside modification / policy subintent")
                assistant_response = generate_assistant_response(intent_label, user_input)
        else:
            print("normal RAG flow")
            assistant_response = generate_assistant_response(intent_label, user_input)

    # Store assistant response in current_message with calculated sentiment and intent
    st.session_state.current_message = {
        "role": "Customer",
        "content": user_input,
        "sentiment": f"{sentiment_label} {sentiment_emoji}",
        "intent": intent_label,
        "assistant_logic": assistant_response,
        "response": None
    }

    st.session_state.display_stage = 1
    st.rerun()

elif st.session_state.display_stage == 1:
    # Add the user message only after sentiment and intent are calculated
    display_message(st.session_state.current_message, show_analysis=True)

    # Show typing animation (don't overwrite the previous chat bubble)
    response_placeholder = st.empty()
    response_placeholder.markdown(
        """
        <style>
        @keyframes blinkDots {
            0% { content: ""; }
            33% { content: "."; }
            66% { content: ".."; }
            100% { content: "..."; }
        }
        .typing-dots::after {
            content: "...";
            animation: blinkDots 1s steps(3, end) infinite;
        }
        </style>
        <div style='display: flex; justify-content: flex-end;'>
            <div style='background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin-bottom: 10px; max-width: 60%; text-align: right;'>
                <strong>✈️ SkyWings Assistant</strong><br>
                <span class='typing-dots'>Typing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate assistant response only once
    if st.session_state.current_message["response"] is None:
        user_query = st.session_state.current_message["content"]
        assistant_logic = st.session_state.current_message["assistant_logic"]

        if assistant_logic == "vector_db":
            result = qa_chain.run(user_query)
        else:
            result = assistant_logic
        st.session_state.current_message["response"] = result

    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    # Show enriched user message again (with sentiment/intent)
    display_message(st.session_state.current_message, show_analysis=True)

    # Add typing animation placeholder for assistant's response
    response_placeholder = st.empty()
    assistant_response = st.session_state.current_message["response"]

    # Start typing effect
    if assistant_response:
        display_typing_effect(response_placeholder, assistant_response)

    # Store both messages after typing effect
    enriched_user_message = {
        "role": "Customer",
        "content": st.session_state.current_message["content"],
        "sentiment": st.session_state.current_message["sentiment"],
        "intent": st.session_state.current_message["intent"],
        "response": None
    }
    st.session_state.messages.append(enriched_user_message)
    st.session_state.messages.append({
        "role": "Assistant",
        "content": assistant_response.strip()
    })

    st.session_state.display_stage = 0
    st.session_state.current_message = None

