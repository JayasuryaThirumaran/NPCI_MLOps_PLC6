from data_preprocessing import X_train, y_train, X_test, y_test, label_encoders
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os


# Initialize and train the Random Forest model
n_estimators = 100
max_depth = 9
max_features = 7
random_state = 42

rf_model = RandomForestClassifier(
    n_estimators = n_estimators, 
    max_depth = max_depth,
    max_features = max_features,
    random_state=42)

rf_model.fit(X_train, y_train)
print(" Model trained successfully!")


# Predict and evaluate
y_pred = rf_model.predict(X_test)

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

# Training and testing accuracy
training_acc = accuracy_score(rf_model.predict(X_train), y_train)
testing_acc = accuracy_score(rf_model.predict(X_test), y_test)

# Save trained model
os.makedirs("trained_model", exist_ok=True)
joblib.dump(rf_model, "trained_model/rf_model_term_deposit.pkl")
print("\n Model saved at: trained_model/rf_model_term_deposit.pkl")


################# MLflow related code below #########################

import mlflow

# Log parameters
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
mlflow.log_param("max_features", max_features)
mlflow.log_param("random_state", random_state)

# Log metrics
mlflow.log_metric("training_accuracy", training_acc)
mlflow.log_metric("testing_accuracy", testing_acc)

# Log model  
# Ignore the WARNING saying `Model logged without a signature and input example.`
mlflow.sklearn.log_model(rf_model, "term-deposit-pred-model")
