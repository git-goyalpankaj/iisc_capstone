from transformers import BertForSequenceClassification, BertTokenizer
import torch
import joblib

# ------------------------------
# 1. Load model and tokenizer
# ------------------------------
model_dir = r"D:\03-Ollama\rasa-chatbot-iisc\rasa_backend\models\custom_nlu"  # your model folder
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

# ------------------------------
# 2. Load label encoder
# ------------------------------
label_encoder_path = r"D:\03-Ollama\rasa-chatbot-iisc\rasa_backend\models\custom_nlu\label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)

# ------------------------------
# 3. Predict on sample text
# ------------------------------
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    label_index = predicted_class.item()
    intent_name = label_encoder.inverse_transform([label_index])[0]
    return intent_name, confidence.item()

# ------------------------------
# 4. Test some examples
# ------------------------------
examples = [
    "Can I modify my seat selection?",
    "How do I check in online?",
    "I want a refund for my flight",
    "How do I claim lost baggage?"
]

for text in examples:
    try:
        intent, conf = predict_intent(text)
        print(f"Input: {text}")
        print(f"Predicted Intent: {intent}, Confidence: {conf:.4f}")
        print("-" * 50)
    except Exception as e:
        print(f"Error for input '{text}': {str(e)}")
