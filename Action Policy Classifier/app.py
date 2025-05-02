import re
import random
import numpy as np
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Pydantic models for request and response
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ClassifierResponse(BaseModel):
    query: str
    type: str
    confidence: float
    specific_area: Optional[str] = None
    response: Optional[str] = None
    session_id: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="Airline Query Classifier API",
    description="Classifies customer queries for airline customer service",
    version="1.0.0"
)

class EnhancedQueryClassifier:
    def __init__(self):
        # Load a small, efficient sentence embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define patterns for safety concerns
        self.safety_patterns = [            
            # Weapons and threats
            r'\b(bomb|explosive|weapon|gun|hijack|attack|threat|terror)\b',
            r'\b(blow\s+up|take\s+over|take\s+down)\b',
            # Dangerous items
            r'\b(flammable|toxic|poisonous|hazardous)\b',
            # Security bypass
            r'\b(bypass|smuggle|sneak|evade|hide)\b.{0,20}\b(security|checkpoint|screening)\b',
            r'\b(hack|break|compromise)\b.{0,20}\b(system|security|account)\b',
            # Clear dangerous intent
            r'\bhow\s+to\s+(smuggle|hide|sneak|bypass)\b',
            r'\bsmuggle\b.{0,30}\b(item|prohibited|illegal)\b',
            r'\bmake\s+a\s+bomb\b',
            # Illegal activities
            r'\b(smuggle|smuggling|trafficking|illegal\s+substance)\b'
        ]
        
        # Define pattern for baggage policy questions
        self.baggage_policy_patterns = [
            r'\b(carry|bring|pack|allowed|permitted|prohibited|restricted)\s+(item|knife|scissor|liquid|battery|medication|sharp|weapon)\b',
            r'\b(allowed|permitted|can\s+i\s+bring|can\s+i\s+carry)\s+in\s+(cabin|carry[\s-]on|checked|luggage|baggage)\b',
            r'\bwhat\s+(items|things)\s+(can|cannot|are)\s+(allowed|permitted|prohibited|banned)\b',
            r'\b(restrictions|rules|policy)\s+on\s+(carrying|bringing)\b'
        ]
        
        # Define patterns for greeting detection
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings|good\s(morning|afternoon|evening))\b',
            r'\bhow\s(are\syou|is\sit\sgoing|are\sthings)\b',
            r'\bnice\sto\smeet\syou\b',
            r'^(hi|hello|hey)$'
        ]
        
        # Add farewell detection patterns
        self.farewell_patterns = [
            r'\b(goodbye|good[\s-]bye|bye|farewell|adios|see\s+you|take\s+care)\b',
            r'\b(end|quit|exit|stop|finish)\s+(conversation|chat|talking|session)\b',
            r'\b(i\s*am|i\'m)\s+(done|finished|leaving|quitting)\b',
            r'^(bye|thanks|thank\s+you)$',
            r'\bthanks?\b.*',  # Simple thanks/thank you
            r'that\'s all'
        ]

        # Define patterns for SkyWings specific mentions
        self.skywings_patterns = [
            r'\b(skywings|sky\s*wings)\b',
            r'\bsw\s*\d{4}\b',  # SkyWings flight code pattern
        ]
        
        # Define patterns for airline/travel relevance
        self.airline_patterns = [
            # Core airline concepts
            r'\b(flight|plane|aircraft|airline|aviation|airport)\b',
            r'\b(booking|reservation|ticket|e-ticket|itinerary)\b',
            r'\b(passenger|traveller|customer|flyer)\b',
            # Travel services
            r'\b(baggage|luggage|suitcase|carry[\s-]on|check[\s-]in)\b',
            r'\b(seat|boarding|gate|terminal|runway|departure|arrival)\b',
            r'\b(delay|cancellation|reschedule|diversion)\b',
            # Travel documents
            r'\b(passport|visa|id|identification|boarding\s+pass)\b',
            # Services
            r'\b(meal|catering|entertainment|wifi|lounge|class)\b',
            # Customer service
            r'\b(refund|compensation|complaint|feedback|review)\b',
            # Common travel activities
            r'\b(book|reserve|cancel|change|modify|upgrade|check[\s-]in)\b'
        ]
        
        # Define patterns for action queries
        self.action_patterns = [
            r'\b(book|reserve|change|modify|cancel|reschedule|upgrade|downgrade)\b',
            r'\bcheck[- ]in\b',
            r'\b(need|want)\s+to\s+(book|cancel|change|modify)\b',
            r'\b(make|get|request)\s+a\s+(reservation|booking|cancellation)\b',
            r'\bhelp\s+me\s+(book|cancel|change)\b'
        ]
        
        # Define patterns for policy queries
        self.policy_patterns = [
            r'\b(policy|policies|rules|allowed|requirements|restrictions|limit|regulation|procedure)\b',
            r'\bwhat\s+(is|are|about)\b',
            r'\bhow\s+(much|many|do|does|can)\b',
            r'\bcan\s+i\b',
            r'\b(tell|explain|clarify)\s+.*\s+(about|regarding)\b',
            r'(\?|tell me about|explain|how|what|when|where|why|which|who)'
        ]
        
        # Define patterns for rants and emotional statements
        self.emotional_patterns = [
            r'\b(bad|terrible|awful|horrible|worst)\s+(day|experience|service|flight)\b',
            r'\b(angry|upset|frustrated|disappointed|annoyed)\b',
            r'\bunhappy\s+with\b',
            r'\bpoorly\s+treated\b',
            r'\b(issue|problem|trouble|difficulty)\b',
            r'\b(awful|terrible|poor|bad)\b',
            r'\b(sucks|useless|pathetic|disappointing)\b',
            r'service\s+is\s+(bad|poor|awful|terrible)',
            r'(not|never)\s+(happy|satisfied|pleased)',
            r'(worst|poorest)\s+(service|experience|airline)'
        ]
        
        # Define response templates
        self.safety_responses = [
            "I'm not able to provide information on that topic. SkyWings is committed to the safety and security of all passengers. If you have questions about permitted items, please refer to our safety guidelines on the website.",
            "I cannot assist with that request. For information about airline safety procedures and permitted items, please visit our website or speak with an airport security representative.",
            "That topic falls outside the scope of what I can discuss. If you have legitimate travel safety concerns, please contact our customer service directly.",
            "For safety and security reasons, I'm unable to address that query. Please refer to official aviation security guidelines for information about permitted and prohibited items."
        ]
        
        self.greeting_responses = [
            "Hello! How can I assist you with your SkyWings travel plans today?",
            "Hi there! I'm here to help with your SkyWings airline queries. What can I do for you?",
            "Welcome to SkyWings support! How may I help you today?",
            "Good day! How can I make your travel experience with SkyWings better today?",
            "Hello! I'm your SkyWings virtual assistant. What information or assistance do you need?"
        ]
        
        self.farewell_responses = [
            "Thank you for chatting with SkyWings customer service. Have a great day!",
            "I appreciate you reaching out to SkyWings. Goodbye and have a pleasant day!",
            "Thank you for using SkyWings virtual assistant. Is there anything else I can help with before you go?",
            "Thank you for flying with SkyWings. Goodbye!"
        ]
        
        self.emotional_responses = [
            "I understand your frustration. I'm here to help resolve your concerns with SkyWings. How can I assist you?",
            "I'm sorry to hear that. Let me help make things better. What specifically can I help you with regarding SkyWings?",
            "I apologize for any inconvenience you've experienced with SkyWings. I'd like to help make things right. What do you need assistance with?",
            "I'm here to turn your experience with SkyWings around. Please let me know how I can help with your travel needs.",
            "Thank you for sharing your concerns. I'm ready to assist you with your SkyWings query. What specific information or help do you need?"
        ]
        
        self.irrelevant_responses = [
            "I'm your SkyWings airline assistant and can only help with airline-related questions. Please ask me about flights, bookings, baggage, or travel policies.",
            "Sorry, but I'm specialized in SkyWings airline services. Could you please ask me something related to your travel with SkyWings?",
            "I'm designed to help with your air travel needs with SkyWings. For other topics, please consult a different service.",
            "I can only assist with SkyWings airline queries. Is there something about your flight, booking, or travel that I can help with?"
        ]
        
        # Create semantic examples for action vs policy
        self.action_examples = [
            "I want to book a SkyWings flight",
            "Help me cancel my SkyWings reservation",
            "Need to change my SkyWings flight date",
            "I'd like to modify my SkyWings booking",
            "Can you help me check in for my SkyWings flight?",
            "I need to reschedule my SkyWings flight",
            "Book a SkyWings ticket to Mumbai",
            "Change my seat assignment on SkyWings flight",
            "Cancel SkyWings flight SW 0123"
        ]
        
        self.policy_examples = [
            "What is SkyWings baggage policy?",
            "How much does extra baggage cost with SkyWings?",
            "What documents do I need for international travel with SkyWings?",
            "Can I bring my pet on a SkyWings flight?",
            "What are the rules for check-in with SkyWings?",
            "Tell me about SkyWings refund policy",
            "Can I get a refund if I cancel my SkyWings ticket?",
            "What is the SkyWings baggage allowance?",
            "Do children get a discount on SkyWings?"
        ]
        
        self.airline_examples = [
            "How do I book a flight?",
            "What's the baggage allowance for international flights?",
            "Can I change my flight reservation date?",
            "When should I arrive at the airport for check-in?",
            "Do you have direct flights to Mumbai from Delhi?",
            "What travel documents do I need for international flights?",
            "How can I check in online for my flight?",
            "What meals are served on long-haul flights?",
            "How much is the flight cancellation fee?"
        ]
        
        self.non_airline_examples = [
            "What's the weather like today?",
            "Can you recommend a good restaurant?",
            "How do I fix my computer?",
            "What's the capital of France?",
            "Tell me a joke",
            "What time is it?",
            "How tall is the Eiffel Tower?",
            "Who won the cricket match yesterday?",
            "Can you help me with my homework?"
        ]
        
        # Pre-compute embeddings for examples
        self.action_embeddings = self.model.encode(self.action_examples)
        self.policy_embeddings = self.model.encode(self.policy_examples)
        self.airline_embeddings = self.model.encode(self.airline_examples)
        self.non_airline_embeddings = self.model.encode(self.non_airline_examples)
        
        # Load the NER model
        try:
            self.ner = pipeline("ner", model="dslim/bert-base-NER", device=-1)
        except:
            self.ner = None
            print("Warning: NER model could not be loaded. Competitor airline detection will be limited.")
        
        # Initialize the expanded airline list
        self.common_airlines = [
            # Indian Airlines
            "air india", "indigo", "spicejet", "vistara", "air india express", "alliance air", 
            "go first", "go air", "akasa air", "star air", "flybig", "trujet", 
            "air asia india", "air costa", "air deccan", "jet airways", "kingfisher", 
            
            # Other major airlines
            "delta", "united", "american", "southwest", "jetblue", "lufthansa", 
            "british airways", "air france", "klm", "emirates", "qatar", "etihad", 
            "singapore airlines", "cathay pacific", "qantas"
        ]

    def _check_pattern_match(self, patterns, text):
        """Check if text matches any of the patterns"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _calculate_semantic_similarity(self, query_embedding, example_embeddings):
        """Calculate semantic similarity between query and examples"""
        similarities = np.dot(query_embedding, example_embeddings.T)
        return np.max(similarities)

    def is_competitor_airline(self, query):
        """Detect if query mentions a non-SkyWings airline using hybrid approach"""
        query_lower = query.lower()
        
        # First check: Direct pattern matching with known airlines
        for airline in self.common_airlines:
            if airline in query_lower:
                return airline
        
        # Second check: Look for "[Something] Airlines/Airways" pattern
        airline_suffixes = ["airline", "airlines", "airways", "air"]
        words = query_lower.split()
        for i in range(len(words) - 1):
            if words[i+1] in airline_suffixes and words[i] not in ["skywings", "sky"]:
                return words[i] + " " + words[i+1]
        
        # Third check: Use NER to find organizations (only if NER is available)
        if self.ner is not None:
            try:
                entities = self.ner(query)
                for entity in entities:
                    if entity['entity'] == 'B-ORG' or entity['entity'] == 'I-ORG':
                        org_name = entity['word'].lower()
                        # Check if this looks like an airline
                        if any(suffix in org_name for suffix in airline_suffixes):
                            return org_name
                        # Check if near airline keywords
                        org_position = query_lower.find(org_name)
                        if org_position != -1:
                            context = query_lower[max(0, org_position-10):min(len(query_lower), org_position+len(org_name)+20)]
                            if any(keyword in context for keyword in ["flight", "fly", "airline", "airways", "ticket"]):
                                return org_name
            except Exception as e:
                print(f"NER error: {str(e)}")
        
        # Fourth check: Check if query is asking about "other airlines" generally
        if re.search(r"other\s+(airline|carrier)", query_lower) or re.search(r"different\s+airline", query_lower):
            return "other airlines"
            
        return None

    def classify_query(self, query):
        """
        Classify a query with the following hierarchy:
        1. Safety check
        2. Greeting/farewell check
        3. Competitor airline check
        4. Airline relevance check
        5. Action vs policy classification
        """
        query = query.strip()
        query_lower = query.lower()

        # Safety check with context analysis
        has_dangerous_intent = any(intent in query_lower for intent in ["how to", "smuggle", "sneak", "hide", "bypass"])
        has_security_context = any(term in query_lower for term in ["security", "screening", "checkpoint", "detection"])
        has_checked_baggage_context = any(term in query_lower for term in ["checked baggage", "checked luggage", "pack in", "put in my luggage"])

        # Check if query contains safety concerns
        high_risk_keywords = ["bomb", "explosive", "smuggle", "bypass security", "sneak through"]
        if any(keyword in query_lower for keyword in high_risk_keywords) and not has_checked_baggage_context:
            return {
                "type": "safety_concern",
                "response": random.choice(self.safety_responses),
                "confidence": 0.95
            }

        # For items like knives, context matters
        if ("knife" in query_lower or "blade" in query_lower or "scissors" in query_lower):
            # If asking about sneaking past security or bypassing - safety concern
            if has_dangerous_intent and has_security_context:
                return {
                    "type": "safety_concern",
                    "response": random.choice(self.safety_responses),
                    "confidence": 0.95
                } 
            # If asking about packing in checked baggage - policy question
            elif has_checked_baggage_context:
                return {
                    "type": "policy",
                    "confidence": 0.95,
                    "specific_area": "baggage_restrictions"
                }

        if self._check_pattern_match(self.safety_patterns, query_lower) and not has_checked_baggage_context:
            return {
                "type": "safety_concern",
                "response": random.choice(self.safety_responses),
                "confidence": 0.95
            }

        # Check for baggage policy questions
        if self._check_pattern_match(self.baggage_policy_patterns, query_lower):
            return {
                "type": "policy",
                "confidence": 0.95,
                "specific_area": "baggage_restrictions"
            }

        # Check if query is a greeting
        if self._check_pattern_match(self.greeting_patterns, query_lower):
            return {
                "type": "greeting",
                "response": random.choice(self.greeting_responses),
                "confidence": 0.95
            }
            
        # Check if query is a farewell
        if self._check_pattern_match(self.farewell_patterns, query_lower):
            return {
                "type": "farewell",
                "response": random.choice(self.farewell_responses),
                "confidence": 0.95
            }

        # Check if query expresses emotion/rant
        if self._check_pattern_match(self.emotional_patterns, query_lower):
            return {
                "type": "emotional",
                "response": random.choice(self.emotional_responses),
                "confidence": 0.90
            }

        # Handle cases where query is just a 4-digit number (likely invalid)
        if re.fullmatch(r"\d{4}", query.strip()):
            return {
                "type": "invalid_format",
                "response": "It looks like you've entered only a flight number. Please enter a valid SkyWings flight number in the format 'SW 1234' so I can assist you.",
                "confidence": 0.95
            }

        # Check if query mentions competitor airlines
        competitor_airline = self.is_competitor_airline(query)
        mentions_skywings = self._check_pattern_match(self.skywings_patterns, query_lower)

        # Check for invalid SkyWings flight number format
        if mentions_skywings:
            if re.search(r'\bsy\s*\d+\b', query_lower) and not re.search(r'\bsw\s*\d{4}\b', query_lower):
                return {
                    "type": "invalid_flight_code",
                    "response": "It looks like you've mentioned a SkyWings flight as 'SY123'. Please note that valid SkyWings flight numbers start with 'SW' followed by 4 digits, like 'SW 1234'.",
                    "confidence": 0.95
                }

        if competitor_airline and not mentions_skywings:
            return {
                "type": "competitor",
                "response": f"I'm a SkyWings virtual assistant and can only provide information about SkyWings services and policies. For information about {competitor_airline.title()}, please visit their website or contact their customer service.",
                "confidence": 0.90
            }

        # Check airline relevance through pattern matching
        is_airline_related = (self._check_pattern_match(self.airline_patterns, query_lower)
                            or self._check_pattern_match(self.skywings_patterns, query_lower))

        # If not clearly airline-related by patterns, check with semantic similarity
        if not is_airline_related:
            query_embedding = self.model.encode([query])
            airline_similarity = self._calculate_semantic_similarity(query_embedding, self.airline_embeddings)
            non_airline_similarity = self._calculate_semantic_similarity(query_embedding, self.non_airline_embeddings)

            # If more similar to non-airline examples, consider it irrelevant
            if non_airline_similarity > airline_similarity * 0.8:
                return {
                    "type": "irrelevant",
                    "response": random.choice(self.irrelevant_responses),
                    "confidence": max(0.7, non_airline_similarity.item())
                }
                
            # Additional check - if very low airline similarity, consider irrelevant
            if airline_similarity < 0.4:
                return {
                    "type": "irrelevant",
                    "response": random.choice(self.irrelevant_responses),
                    "confidence": 0.75
                }

        # Check action patterns
        action_matches = sum(1 for pattern in self.action_patterns if re.search(pattern, query_lower))

        # Check policy patterns
        policy_matches = sum(1 for pattern in self.policy_patterns if re.search(pattern, query_lower))

        # If we have clear pattern matches
        if action_matches > policy_matches + 1:
            return {
                "type": "action",
                "confidence": min(0.7 + 0.1 * action_matches, 0.95)
            }

        if policy_matches > action_matches + 1:
            return {
                "type": "policy",
                "confidence": min(0.7 + 0.1 * policy_matches, 0.95)
            }

        # If pattern matching is inconclusive, use semantic similarity
        query_embedding = self.model.encode([query])

        # Calculate similarity with action examples
        action_similarity = self._calculate_semantic_similarity(query_embedding, self.action_embeddings)

        # Calculate similarity with policy examples
        policy_similarity = self._calculate_semantic_similarity(query_embedding, self.policy_embeddings)

        # Compare similarities with a small threshold to avoid ties
        if action_similarity > policy_similarity + 0.05:
            return {
                "type": "action",
                "confidence": action_similarity.item()
            }
        elif policy_similarity > action_similarity + 0.05:
            return {
                "type": "policy",
                "confidence": policy_similarity.item()
            }
        else:
            # If we're still unsure, check for question marks as a hint for policy
            if "?" in query:
                return {
                    "type": "policy",
                    "confidence": 0.6
                }

            # Default to action as it's safer to send to a human agent
            return {
                "type": "action",
                "confidence": 0.55
            }

# Create a singleton instance of the classifier
classifier = EnhancedQueryClassifier()

@app.get("/")
async def root():
    return {"message": "Welcome to the Airline Query Classifier API"}

@app.post("/classify", response_model=ClassifierResponse)
async def classify_query(request: QueryRequest):
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Use the classifier to get the result
    result = classifier.classify_query(request.query)
    
    # Format the response
    response = ClassifierResponse(
        query=request.query,
        type=result["type"],
        confidence=result["confidence"],
        session_id=request.session_id
    )
    
    # Add optional fields if present
    if "response" in result:
        response.response = result["response"]
    
    if "specific_area" in result:
        response.specific_area = result["specific_area"]
    
    return response

# Example usage with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
