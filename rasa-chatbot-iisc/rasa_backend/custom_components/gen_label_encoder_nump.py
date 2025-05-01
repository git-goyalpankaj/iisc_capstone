from sklearn import preprocessing
import joblib

# Suppose these are your class labels
intent_labels = [
    "Baggage", "Booking, Modifications And Cancellations", "Check-In And Boarding",
    "Customer Support", "Fares And Payments", "Flight Operations",
    "Irrelevant", "Loyalty And Rewards", "Other",
    "Passenger Services", "Refunds", "Travel Documents"
]

# Create and fit label encoder
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(intent_labels)

# Save safely (protocol=4 ensures compatibility)
joblib.dump(label_encoder, "label_encoder.pkl", protocol=4)
