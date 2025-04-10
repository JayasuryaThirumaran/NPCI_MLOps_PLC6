from data_preprocessing import X_train, y_train, X_test, y_test, label_encoders
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os


# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print(" Model trained successfully!")

# Predict and evaluate
y_pred = model.predict(X_test)

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print("\n Model Evaluation:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")

evaluate_model(y_test, y_pred)

# Save trained model
os.makedirs("trained_model", exist_ok=True)
joblib.dump(model, "trained_model/rf_model_term_deposit.pkl")
print("\n Model saved at: trained_model/rf_model_term_deposit.pkl")

