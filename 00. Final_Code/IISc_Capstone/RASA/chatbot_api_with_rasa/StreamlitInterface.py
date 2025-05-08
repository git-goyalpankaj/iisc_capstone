import os
import requests
import time
import re
from torch import classes
import joblib
import streamlit as st
import nest_asyncio
import numpy as np
import datetime
import base64

from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import messages_from_dict
from langchain.schema import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="Welcome to Skywings", layout="wide")
st.title("🤖 Welcome to Skywings! ")

RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"

RASA_KNOWN_INTENTS = []
DEBUG_MODE = True

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


def sanitize_intent_name(raw_intent: str) -> str:
    return raw_intent.replace(",", "").replace(" ", "_")

# ---------------------------
# UI Functions
# ---------------------------

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

def display_message(msg, show_analysis=False):
    avatars = {"Customer": "🙋", "ChatAgent": "🤖"}
    colors = {"Customer": "#DCF8C6", "ChatAgent": "#F1F0F0"}

    # === NEW CODE START ===
    # Special styling for RASA form questions
    if msg.get("is_form_question", False):
        st.markdown(f"""
        <div style='background-color:#E3F2FD; 
                    padding:12px; 
                    border-radius:8px;
                    border-left:4px solid #2196F3;
                    margin:10px 0;'>
            <strong>📝 {avatars['ChatAgent']} Form Question</strong><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)
        return  # Skip normal rendering for form questions
    # === NEW CODE END ===

    # Special styling for RASA form prompts
    if msg.get("is_rasa_form", False):
        colors["ChatAgent"] = "#E3F2FD"  # Light blue for form questions
        msg['content'] = f"📝 {msg['content']}"  # Add form icon

    if msg['role'] == "Customer":
        st.markdown(
            f"""<div style='display: flex; gap: 8px; margin-bottom: 10px;'>
            <div style='background-color: {colors['Customer']}; padding: 10px; border-radius: 10px; max-width: 45%; text-align: left;'>
            <strong>{avatars['Customer']} Customer</strong><br>
            <span>{msg['content']}</span></div>""",
            unsafe_allow_html=True
        )
        if show_analysis:
            st.markdown(
                f"""<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-size: 14px; max-width: 35%; min-width: 150px;'>
                <strong>🧠 Sentiment:</strong> {msg.get('sentiment', '')}<br>
                <strong>🎯 Intent:</strong> {msg.get('intent', '')} ({msg.get('score', '')}%)
                </div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    elif msg['role'] == "ChatAgent":
        st.markdown(
            f"""<div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
            <div style='background-color: {colors['ChatAgent']}; padding: 10px; border-radius: 10px; max-width: 60%; text-align: right;'>
            <strong>{avatars['ChatAgent']} Assistant</strong><br>
            <span>{msg['content']}</span></div></div>""",
            unsafe_allow_html=True
        )


def format_messages_as_string(messages: list[BaseMessage]) -> str:
    """Convert list of LangChain messages to a readable string for prompt history."""
    lines = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def log_rasa_debug(message):
    """Helper function to log debug messages"""
    if DEBUG_MODE:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.rasa_debug_log.append(f"[{timestamp}] {message}")
        print(f"RASA DEBUG: {message}")  # Also prints to console


# ---------------------------
# Session Initialization
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
if "current_message" not in st.session_state:
    st.session_state.current_message = {}
#if "chat_memory" not in st.session_state:
#    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)
#if "route" not in st.session_state:
#    st.session_state.route = "llm"  # Default to LLM until intent says otherwise
#if "in_rasa_flow" not in st.session_state:
#    st.session_state.in_rasa_flow = False
if "rasa_debug_log" not in st.session_state:
    st.session_state.rasa_debug_log = []

# Background image for the UI
add_bg_from_local("./Background1.jpg")  # Update with the correct path to your image


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.button(
    "🪠 Clear Chat", 
    key="clear_chat_button",
    on_click=lambda: st.session_state.update(
        messages=[],
        display_stage=0,
        current_message=None,
        route="llm",
        in_rasa_flow=False
    )
)

# ---------------------------
# Chat Handling
# ---------------------------
for msg in st.session_state.messages:
    display_message(msg)

user_input = st.chat_input("Say something...")

if user_input and st.session_state.display_stage == 0:
    st.session_state.current_message = {
        "role": "Customer",
        "content": user_input,
        "response": None
    }
    st.session_state.display_stage = 1
    st.rerun()

elif st.session_state.display_stage == 1:
    temp_msg = {**st.session_state.current_message}
    display_message(temp_msg)
    time.sleep(0.5)
    st.session_state.display_stage = 2
    st.rerun()

elif st.session_state.display_stage == 2:
    display_message(st.session_state.current_message)

    if st.session_state.current_message["response"] is None:
        with st.spinner("Thinking..."):
            user_message = st.session_state.current_message["content"]
            print("user input :", user_message)
            response = invoke_rasa(user_message)
            print("DS 2 -> Rasa Response", response)
            st.session_state.current_message = {
                "role": "Customer",
                "content": user_message,
                "response": response
            }
    time.sleep(0.5)
    st.session_state.display_stage = 3
    st.rerun()
            
            
            # if st.session_state.route == "rasa":
            #     try:
            #         # Only get what we actually use
            #         rasa_response, rasa_intent = send_message_to_rasa(user_message)

            #         # In stage 2 after RASA response
            #         st.sidebar.write(f"Last RASA Intent: {rasa_intent}")
            #         st.sidebar.write(f"Last RASA Response: {rasa_response}")

            #         # Simplified state update
            #         st.session_state.current_message.update({
            #             "response": rasa_response,
            #         #    "response": rasa_intent,
            #         #    "intent": rasa_intent
            #         })
                    
            #         # Memory handling
            #         #st.session_state.chat_memory.add_user_message(user_message)
            #         #st.session_state.chat_memory.add_ai_message(rasa_response)

            #         # Auto-reset logic
            #         if rasa_intent not in RASA_KNOWN_INTENTS:
            #             st.session_state.in_rasa_flow = False
            #             st.session_state.route = "llm"

            #     except Exception as e:
            #         st.session_state.current_message["response"] = (
            #             "Sorry, I'm having trouble with the booking system. "
            #             "Please try again later."
            #         )
            #         #log_rasa_debug(f"RASA Error: {str(e)}")
            #         st.error(f"RASA Error: {str(e)}")

            # else:        
            #     # Directly use LLM + FAISS
            #     context_text = ""
            #     if retriever:
            #         relevant_docs = retriever.get_relevant_documents(user_message)
            #         context_text = "\n\n".join(doc.page_content for doc in relevant_docs[:3])

            #     num_messages = st.session_state.memory_turns * 2  # user+bot per turn
            #     history_text = format_messages_as_string(
            #         st.session_state.chat_memory.chat_memory.messages[-num_messages:]
            #     )

            #     prompt_prefix = intent_prompt_templates.get(
            #         st.session_state.current_message["intent"],
            #         "You are a helpful airline assistant."
            #     )

            #     # Build full prompt
            #     full_prompt = f"""
            #     {prompt_prefix}

            #     Context:
            #     {context_text}

            #     Conversation so far:
            #     {history_text}

            #     User: {user_message}
            #     Assistant:
            #     Please reply in two parts:
            #     (1) Brief explanation (2-3 lines)  
            #     (2) Then give the final answer clearly in the last sentence.
            #     """

            #     result = llm.invoke(
            #         full_prompt,
            #         temperature=st.session_state.temperature,
            #         top_p=st.session_state.top_p,
            #         options={
            #             "num_predict": st.session_state.max_tokens
            #         }
            #     )
            #     # Save turn to memory
            #     st.session_state.chat_memory.chat_memory.add_user_message(user_message)
            #     st.session_state.chat_memory.chat_memory.add_ai_message(result)                
            #     st.session_state.current_message["response"] = result



elif st.session_state.display_stage == 3:
    # Get the assistant's response
    assistant_response = st.session_state.current_message["response"].strip()
    
    # Enhanced formatting for different response patterns
    if any(marker in assistant_response for marker in ("1.", "2.", "- ")) or "\n-" in assistant_response:
            # Format numbered/bulleted lists with better styling
        formatted_response = []
        lines = assistant_response.split('\n')
        
        for line in lines:
            if line.startswith(("1.", "2.", "3.", "4.")):
                formatted_response.append(f"<div style='margin-bottom: 8px; text-align: left;'><strong>{line}</strong></div>")
            elif line.startswith("- "):
                formatted_response.append(f"<div style='margin-left: 20px; margin-bottom: 4px; text-align: left;'>• {line[2:]}</div>")
            else:
                formatted_response.append(f"<div style='margin-bottom: 8px; text-align: left;'>{line}</div>")
        
        assistant_content = "".join(formatted_response)
    else:
        assistant_content = assistant_response

    # # Create and display the assistant message with left-aligned text
    assistant_msg = {
        "role": "ChatAgent",
        "content": f"""
        <div style='
            background-color: #F1F0F0;
            padding: 12px;
            border-radius: 12px;
            border-left: 4px solid #4CAF50;
            margin: 12px 0;
            text-align: left;
        '>
            <div style='font-size: 14px; color: #555; margin-bottom: 8px;'>
                <strong>🤖 Assistant</strong>
            </div>
            <div style='line-height: 1.5;'>
                {assistant_content}
            </div>
        </div>
        """,
        "is_html": True
    }
    
    # Display the assistant message
    st.markdown(assistant_msg["content"], unsafe_allow_html=True)
    
    # Store both messages in history
    st.session_state.messages.append(st.session_state.current_message)
    st.session_state.messages.append({
        "role": "ChatAgent",
        "content": assistant_response  # Store original text
    })
    
    # Reset for next message
    st.session_state.display_stage = 0
    st.session_state.current_message = None
    st.rerun()