# app/models/sentiment_model.py
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SentimentModel:
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""Classify the sentiment of the following text as Positive, Negative or Neutral.
Text: "{text}"
Sentiment:"""
        )

    def predict(self, text: str) -> str:
        formatted = self.prompt.format(text=text)
        inputs = self.tokenizer(formatted, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=32)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
