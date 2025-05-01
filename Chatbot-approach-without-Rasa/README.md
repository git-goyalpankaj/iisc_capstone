
# 🛫 SkyWings AI Assistant

SkyWings is a conversational AI chatbot for airline customer service, powered by **LangChain**, **Streamlit**, **Ollama**, **Mistral**, **FAISS**, and **Hugging Face Transformers**. It supports intent and sentiment detection, document-grounded answers, small talk, and action-triggering sub-intents.

---

## 🌟 Features

- 🧠 **Conversational Memory** using `ConversationSummaryBufferMemory` from LangChain.
- 🎯 **Intent Classification** using a fine-tuned `bert-base-uncased` model.
- 😊 **Sentiment Analysis** via `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- 📚 **RAG with FAISS Vector Store** built from SkyWings policy documents.
- 💬 **Sub-Intent Classification** using zero-shot (`facebook/bart-large-mnli`):
  - Booking Request
  - Cancellation Request
  - Modification Request
  - Policy Question
- 🤖 **Small Talk / Greeting / Farewell / Filler Phrase Detection** using file-matching.
- 🎨 **Stylized Chat UI**:
  - Right-aligned user messages
  - Left-aligned assistant responses
  - Typing animation
  - Light theme with background image
- ⚡ **Streaming Assistant Responses** (character-by-character typing).
- 🧩 **Ollama Integration** with `mistral` model.

## 🧪 Models Used

| Task | Model |
|------|-------|
| Intent Classification | Fine-tuned `bert-base-uncased` |
| Sentiment Analysis | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Sub-Intent (Zero-shot) | `facebook/bart-large-mnli` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | `mistral` via Ollama |

---

## 📚 Core Libraries

- `streamlit`
- `langchain`
- `ollama`
- `faiss-cpu`
- `transformers`
- `torch`
- `sentence-transformers`
- `scikit-learn`
- `pandas`, `numpy`, `re`, `os`, etc.

---

## 🔍 Functionality Breakdown

| Feature | Method |
|--------|--------|
| Greeting/Farewell Detection | Rule-based text match from file |
| Intent Detection | Fine-tuned BERT classification |
| Sentiment Detection | RoBERTa sentiment pipeline |
| RAG Answering | FAISS + LangChain Retriever |
| Sub-Intent Handling | Zero-shot classification |
| Assistant Generation | Ollama LLM (Mistral) |
| Multi-turn Memory | LangChain ConversationSummaryBufferMemory |

---
