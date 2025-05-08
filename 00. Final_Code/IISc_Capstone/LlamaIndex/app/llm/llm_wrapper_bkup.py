# app/llm/llm_wrapper.py
from langchain_huggingface import HuggingFaceEndpoint

class LLMWrapper:
    def __init__(self, repo_id: str):
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=512,
            top_k=30,
            temperature=0.1,
            repetition_penalty=1.03,
        )

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)
