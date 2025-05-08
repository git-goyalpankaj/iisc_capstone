import os
import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

class ModelWrapper:
    def __init__(self, model_dir):
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
        label = self.label_encoder.inverse_transform([predicted_class.item()])[0]
        return {"intent": label, "confidence": confidence.item()}

intent_model = ModelWrapper("C:\IISc_Capstone\RASA\chatbot_api_with_rasa\custom_intent")