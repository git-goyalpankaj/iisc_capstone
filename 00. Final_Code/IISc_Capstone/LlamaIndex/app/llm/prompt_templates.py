import os
from langchain.prompts import PromptTemplate

# Directory where .txt templates are stored
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "templates")

# Helper to load template text
def load_template(template_filename: str) -> str:
    path = os.path.join(TEMPLATE_DIR, template_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file {template_filename} not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# PromptTemplate instances built from .txt files
BASIC_PROMPT = PromptTemplate(
    input_variables=["user_message", "context", "intent", "sentiment"],
    template=load_template("basic_prompt_template.txt")
)

ACTION_PROMPT = PromptTemplate(
    input_variables=["user_message", "context", "intent", "sentiment", "api_result"],
    template=load_template("action_prompt_template.txt")
)

FALLBACK_PROMPT = PromptTemplate(
    input_variables=["user_message"],
    template=load_template("fallback_prompt_template.txt")
)
