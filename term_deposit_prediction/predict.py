import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("trained_model/rf_model_term_deposit.pkl")
label_encoders = joblib.load("trained_model/label_encoders.pkl")

binary_mapping = {'yes': 1, 'no': 0}

def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df['default'] = df['default'].map(binary_mapping)
    df['housing'] = df['housing'].map(binary_mapping)
    df['loan'] = df['loan'].map(binary_mapping)

    for col in ['job', 'marital', 'education', 'day_of_week', 'month']:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    return df

def predict(data: dict):
    df = preprocess_input(data)
    prediction = model.predict(df)[0]
    return prediction
