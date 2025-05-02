# Airline Query Classifier API

This FastAPI application provides a robust classification service for airline customer queries, designed to work seamlessly with RASA NLU. It classifies incoming queries into various types (action, policy, safety concerns, etc.) and provides the necessary information for downstream processing.

## Features

- **Fine-grained query classification**: Distinguishes between 8+ different query types
- **Context-aware safety detection**: Intelligently differentiates between legitimate baggage policy questions and safety concerns
- **Farewell/greeting detection**: Handles conversation flow markers appropriately
- **Competitor airline detection**: Identifies when customers are asking about other airlines
- **Semantic similarity matching**: Uses sentence embeddings for robust classification
- **Optimized for RASA integration**: Outputs formatted for easy consumption by RASA NLU

## Installation

### Prerequisites

- Python 3.8+
- FastAPI
- Sentence-Transformers
- Transformers
- PyTorch
- Pydantic
- Uvicorn

### Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install fastapi uvicorn pydantic sentence-transformers transformers torch numpy

# Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

### Classify a Query

**Endpoint**: `POST /classify`

**Request Body**:
```json
{
  "query": "Can I pack a small knife in my checked baggage?",
  "session_id": "user123"  // Optional
}
```

**Response**:
```json
{
  "query": "Can I pack a small knife in my checked baggage?",
  "type": "policy",
  "confidence": 0.95,
  "specific_area": "baggage_restrictions",
  "response": null,
  "session_id": "user123"
}
```

## Integration with RASA

The classifier is designed to be called from your RASA system. The output type and additional fields can be used to:

1. Direct the conversation flow in RASA
2. Trigger specific RASA intents based on the classification
3. Fill RASA slots with additional information
4. Use the confidence score to determine fallback behaviors

### Example RASA Integration

In your RASA HTTP API client:

```python
import requests

def classify_query(query, session_id=None):
    response = requests.post(
        "http://localhost:8000/classify",
        json={"query": query, "session_id": session_id}
    )
    return response.json()

def process_in_rasa(classification_result):
    # Map classification types to RASA intents
    type_to_intent = {
        "action": "action_intent",
        "policy": "policy_intent",
        "safety_concern": "safety_concern_intent",
        "greeting": "greeting_intent",
        "farewell": "farewell_intent"
    }
    
    # Use classification in RASA
    intent = type_to_intent.get(classification_result["type"], "default_intent")
    confidence = classification_result["confidence"]
    
    # Now you can use this intent in your RASA flow
    # ...
```

## Classification Types

- **action**: Customer wants to perform an action (book, cancel, modify)
- **policy**: Customer is asking about airline policies or information
- **safety_concern**: Query contains potentially dangerous content
- **greeting**: Simple conversation opener
- **farewell**: Conversation closer or thanks
- **emotional**: Customer expressing frustration or complaints
- **competitor**: Query about another airline
- **irrelevant**: Non-airline related query
- **invalid_format**: Improperly formatted request (e.g., flight number)

## Additional Information

For queries classified as "policy" with a specific focus on baggage restrictions, the API includes a `specific_area` field with the value "baggage_restrictions". This helps RASA provide more targeted responses.

For types like "greeting", "farewell", "safety_concern", etc., the API includes a pre-generated response that can be used directly.

## Performance Considerations

The first query might take longer to process as models are loaded into memory. Subsequent queries will be much faster. For production deployment, consider:

1. Using a production ASGI server like Gunicorn
2. Implementing caching for model loading
3. Optional GPU acceleration for faster inference
