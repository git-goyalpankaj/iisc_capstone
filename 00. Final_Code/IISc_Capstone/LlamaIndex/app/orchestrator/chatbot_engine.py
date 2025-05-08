# app/orchestrator/chatbot_engine.py
from ..memory.memory_store import MemoryStore
#from ..models.intent_model import IntentModel # when implemented
#from ..models.sentiment_model import SentimentModel # when implemented
from ..llm.llm_wrapper_bkup import LLMWrapper
from ..orchestrator.prompt_manager import PromptManager
# from ..rag_engine.retriever import Retriever    # when implemented
# from ..tools_api.api_caller import APICaller    # when implemented

class ChatbotEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = MemoryStore()
        # begins the loading of the IntentModel 
        #self.intent_model = IntentModel()
        # begins the loading of the SentimentModel 
        #self.sentiment_model = SentimentModel()
        self.llm = LLMWrapper("HuggingFaceH4/zephyr-7b-beta")
        self.prompt_mgr = PromptManager(template_dir="templates")

    def step(self, user_message: str) -> str:

        # 1) Append user turn
        self.memory.append_turn(self.session_id, {"role": "user", "text": user_message})

        # 2) Query enhancement
        #intent = self.intent_model.predict(user_message)
        #sentiment = self.sentiment_model.predict(user_message)
        intent = None
        sentiment = None

        # 3) Build prompt context
        history = self.memory.get_history(self.session_id)
        history_text = "\n".join(f"{t['role']}: {t['text']}" for t in history)

        prompt = self.prompt_mgr.format(
            "basic_prompt_template",
            user_message=user_message,
            context=history_text or "",
            intent=intent or "",
            sentiment=sentiment or ""
        )

        # 4) (Optional) call RAG / APIs here…

        # 5) LLM generate
        messages = [
            {"role": "user", "content": user_message}
        ]
        reply = self.llm.generate(messages)
        print(reply)

        # 6) Append assistant turn
        self.memory.append_turn(self.session_id, {"role": "assistant", "text": reply})
        return reply
