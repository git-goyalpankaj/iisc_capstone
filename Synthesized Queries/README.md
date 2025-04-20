# Synthesized Queries for Intent and Sentiment Classification

### PLEASE VERIFY THE QUERIES IN THIS SHEET - [🔗 SYNTHESIZED QUERIES FOR MANUAL VERIFICATION](https://docs.google.com/spreadsheets/d/1nQx-kuhMfhFfr40RDW358Fqr9ouZAqXFXHBkP7CdA8E/edit?usp=sharing)

Queries have been programmatically synthesized using Phi-2 for the purpose of generating a labeled dataset suitable for training and evaluating models on intent and sentiment classification tasks.

## **Key Components**

### **Sentiments Covered**:
- Positive
- Neutral
- Negative

### **Intents Included**:
- Baggage
- Check-in and Boarding
- Booking, Modifications and Cancellations
- Travel Documents
- Fares and Payments
- Refunds
- Flight Operations
- Passenger Services
- Loyalty and Rewards
- Customer Support
- Other (Intents other than the above)
- Irrelevant (Intents unrelated to the domain)

## **Approach**

### **Prompt Templates**:
Manually crafted prompt templates were designed for each intent and sentiment combination. These templates simulate realistic customer queries for a domestic Indian airline.

### **Model Used**:
Phi-2 was used to generate high-quality, diverse user queries by filling in sentiment-specific instructions within each prompt template.

### **Labels**:
Each query generated has two associated ground truth labels:
1. Intent (one of the 12 listed categories)
2. Sentiment (positive, neutral, negative)

## **Purpose**

The goal is to create a realistic, balanced dataset for training or benchmarking intent and sentiment classification models.

The dataset can be used to:
- Fine-tune a classification model.
- Evaluate the performance of existing models.
- Serve as a standard for comparison in model experimentation pipelines for our project.
