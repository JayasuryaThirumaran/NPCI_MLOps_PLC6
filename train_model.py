from data_preprocessing import X_train, X_test, y_train, y_test, label_encoders
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


os.makedirs("trained_model", exist_ok=True)

def evaluate_model(y_true, y_pred):
    print(f"Accuracy: {round(accuracy_score(y_true, y_pred), 3)}")
    print(f"Precision: {round(precision_score(y_true, y_pred), 3)}")
    print(f"Recall: {round(recall_score(y_true, y_pred), 3)}")
    print(f"F1 Score: {round(f1_score(y_true, y_pred), 3)}")

# MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Term Deposit Classification")

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Log model
    joblib.dump(model, "trained_model/rf_model_term_deposit.pkl")
    joblib.dump(label_encoders, "trained_model/label_encoders.pkl")
    mlflow.sklearn.log_model(model, "random_forest_model")
