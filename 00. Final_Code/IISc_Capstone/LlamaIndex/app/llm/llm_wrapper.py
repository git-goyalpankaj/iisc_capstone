# app/llm/llm_wrapper.py

from huggingface_hub import InferenceClient
from app.config.settings import HF_API_TOKEN

class LLMWrapper:
    def __init__(self, model_name: str):
        print(model_name)
        self.client = InferenceClient(model_name)
        self.model_name = model_name

    def generate(self, messages: list) -> str:
        """
        Generate a response from the LLM using a list of messages.
        Each message should be a dict with 'role' and 'content' keys.
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError(f"Invalid messages format: {messages!r}")

        # Call chat_completion method to get the model's response
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=512,        # How many tokens to generate
            temperature=0.7,       # Control randomness
            top_p=0.95,            # Nucleus sampling
            repetition_penalty=1.1 # Slightly discourage repetitive outputs
        )

        # response is usually a dictionary with 'choices'
        return response.choices[0].message["content"]
