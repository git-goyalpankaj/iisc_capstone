import os
import torch
import joblib
import numpy as np  # Explicit import
from typing import Any, Dict, List, Text
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT, INTENT
from transformers import BertForSequenceClassification, BertTokenizer

class ModelWrapper:
    def __init__(self, model, tokenizer, label_encoder):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
        label_index = predicted_class.item()
        intent = self.label_encoder.inverse_transform([label_index])[0]
        return intent, confidence.item()

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
    is_trainable=False
)
class MyCustomIntentClassifier(GraphComponent):
    def __init__(self, config: Dict[Text, Any]) -> None:
        self.model_wrapper = self._load_model()
        self.threshold = 0.5  # Set fallback threshold

    def _load_model(self):
        model_dir = r"D:\03-Ollama\rasa-chatbot-iisc\rasa_backend\models\custom_nlu"
        
        try:
            # Verify numpy is available
            import numpy as np
            np.zeros(1)  # Simple test
            
            tokenizer = BertTokenizer.from_pretrained(model_dir)
            model = BertForSequenceClassification.from_pretrained(model_dir)
            model.eval()

            label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            label_encoder = joblib.load(label_encoder_path)
            
            return ModelWrapper(model, tokenizer, label_encoder)
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "MyCustomIntentClassifier":
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        updated_messages = []
        for message in messages:
            text = message.get(TEXT)
            if text:
                try:
                    intent, confidence = self.model_wrapper.predict(text)
                    print(f"[DEBUG] Text: '{text}' → Predicted Intent: '{intent}' with Confidence: {confidence:.4f}")
                    
                    if confidence < self.threshold:
                        print(f"[DEBUG] Confidence {confidence:.4f} below threshold {self.threshold}. Falling back to 'nlu_fallback'.")
                        intent = "nlu_fallback"
                        confidence = 1.0

                    # 💡 Ensure confidence is pure Python float
                    message.set(INTENT, {"name": intent, "confidence": float(confidence)})
 
                except Exception as e:
                    print(f"[ERROR] Prediction failed for text '{text}': {str(e)}")
                    message.set(INTENT, {"name": "nlu_fallback", "confidence": 1.0})

            updated_messages.append(message)  # 💡 Append modified message

        return updated_messages    

    def predict(self, messages: List[Message]) -> List[Message]:
        updated_messages = []
        for message in messages:
            text = message.get(TEXT)
            if text:
                try:
                    intent, confidence = self.model_wrapper.predict(text)
                    print(f"[DEBUG] Text: '{text}' → Predicted Intent: '{intent}' with Confidence: {confidence:.4f}")

                    if confidence < self.threshold:
                        print(f"[DEBUG] Confidence {confidence:.4f} below threshold {self.threshold}. Falling back to 'nlu_fallback'.")
                        intent = "nlu_fallback"
                        confidence = 1.0

                    message.set(INTENT, {"name": intent, "confidence": float(confidence)})

                except Exception as e:
                    print(f"[ERROR] Prediction failed for text '{text}': {str(e)}")
                    message.set(INTENT, {"name": "nlu_fallback", "confidence": 1.0})

            updated_messages.append(message)

        return updated_messages
    
    @staticmethod
    def get_output_signature() -> Dict[Text, Any]:
        return {INTENT: Any}  # 🔥 This tells Rasa you are setting INTENT
