import os
import torch
import joblib
from typing import Any, Dict, List, Text
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT, INTENT

from transformers import BertForSequenceClassification, BertTokenizer

def load_my_model():
    model_dir = r"D:\03-Ollama\rasa-chatbot-iisc\rasa_backend\models\custom_nlu"

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    label_encoder = joblib.load(label_encoder_path)

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

    return ModelWrapper(model, tokenizer, label_encoder)

# Registering with DefaultV1Recipe
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class MyCustomIntentClassifier(GraphComponent):
    def __init__(self, config: Dict[Text, Any]) -> None:
        self.model_wrapper = load_my_model()

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
        for message in messages:
            text = message.get(TEXT)
            if text:
                intent, confidence = self.model_wrapper.predict(text)
                message.set(INTENT, {"name": intent, "confidence": confidence})
        return messages