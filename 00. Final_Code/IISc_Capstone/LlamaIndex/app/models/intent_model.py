# app/models/intent_model.py
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class IntentModel:
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""Identify the customer's intent in the text below. Output one of: book/cancel/query/other.
Text: "{text}"
Intent:"""
        )

    def predict(self, text: str) -> str:
        formatted = self.prompt.format(text=text)
        inputs = self.tokenizer(formatted, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=32)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
