# Intent Classification

Focuses on **intent classification** for customer service queries in the airline industry.  
The goal is to accurately predict the intent behind customer queries to assist in automated support and service optimization.
The dataset consists of **1620 manually verified records** synthesized for **intent classification**. It includes various **customer service intents** related to airline queries, such as booking, check-in, baggage, and refunds. The dataset is **balanced** to represent real-world customer queries.

## Dataset Overview

- **1620 manually verified records** synthesized for intent classification.
- Covers diverse **customer service intents** like booking, baggage, check-in, refunds, and more.
- **Balanced dataset**: 135 records per intent category.
- **Synthetic queries** generated using the **Phi-2 language model** for broad coverage of real-world scenarios.

### Intent Categories

1. Travel Documents
2. Customer Support
3. Flight Operations
4. Refunds
5. Baggage
6. Passenger Services
7. Check-in and Boarding
8. Other
9. Booking, Modifications, and Cancellations
10. Loyalty and Rewards
11. Irrelevant
12. Fares and Payments

### Data Splits

- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

---

## Model Development

Several models were explored for the intent classification task:

- **XGBoost Classifier**
- **CatBoost Classifier**
- **BERT (bert-base-uncased)** fine-tuning
- **DistilBERT (distilbert-base-uncased)** fine-tuning

All models used **BERT embeddings** as input features for consistency.

### Final Outcome

- **BERT** achieved the best performance across evaluation metrics (accuracy, precision, recall, F1-score).
- The **final model** selected and saved is the **fine-tuned BERT** model.
- The **saved BERT model file** is linked for direct usage.

---

## Repository Contents

- **Classification Reports**  
  Images of the classification reports for each model comparison are attached for easy reference.

- **Training Scripts**  
  Full training scripts for XGBoost, CatBoost, BERT, and DistilBERT are included.

- **Final Intent Classifier**  
  The final script for **loading the saved BERT model** and running inference on new queries is provided.

---

## How to Use

1. **Train Models** (Optional)  
   Run the respective training scripts for each model if you wish to retrain.

2. **Use Final Model**  
   Load the final fine-tuned BERT model and use it for classifying new customer queries.
