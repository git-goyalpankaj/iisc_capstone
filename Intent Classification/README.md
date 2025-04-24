# Intent Classification

This dataset consists of **1620 manually verified records** synthesized for **intent classification**. It includes various **customer service intents** related to airline queries, such as booking, check-in, baggage, and refunds. The dataset is **balanced** to represent real-world customer queries.

## Dataset Overview

The dataset is used to train, evaluate, and compare models for **intent classification**. Each record is labeled with an **intent**, which represents the type of customer query. The data is divided into **12 unique intent categories**, with **135 records per intent**.

## Dataset Details

- **Manually Verified**: Ensures data quality and correctness.
- **Balanced Classes**: Each intent category has an equal number of records to avoid class imbalance.
- **Synthetic Queries**: Queries are generated using a language model (Phi-2) to cover diverse real-world scenarios.

### Intent Categories

1. **Travel Documents**
2. **Customer Support**
3. **Flight Operations**
4. **Refunds**
5. **Baggage**
6. **Passenger Services**
7. **Check-in and Boarding**
8. **Other**
9. **Booking, Modifications, and Cancellations**
10. **Loyalty and Rewards**
11. **Irrelevant**
12. **Fares and Payments**

### Data Splits

The data is divided into three sets for model development:

- **Training Set**: 70% of the data used to train the model.
- **Validation Set**: 15% of the data used for hyperparameter tuning.
- **Test Set**: 15% of the data used to evaluate the final model.
