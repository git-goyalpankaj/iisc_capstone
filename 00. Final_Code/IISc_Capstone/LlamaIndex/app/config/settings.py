# app/config/settings.py
"""
Configuration settings for the chatbot system, including Hugging Face authentication.
"""
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Hugging Face API token (set this in your environment or in a .env file)
# Example .env entry: HUGGINGFACEHUB_API_TOKEN=hf_xxx
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(HF_API_TOKEN)
# Other settings can go here (Redis URL, model names, etc.)