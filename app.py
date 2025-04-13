from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and encoders
rf_model = joblib.load("trained_model/rf_model_term_deposit.pkl")
label_encoders = joblib.load("trained_model/label_encoders.pkl")

# Define request schema
class TermDepositInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: int
    balance: float
    housing: int
    loan: int
    day_of_week: str
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int

def preprocess_input(data: TermDepositInput):
    input_dict = data.dict()

    # Encode categorical features
    for col in ['job', 'marital', 'education', 'day_of_week', 'month']:
        le = label_encoders[col]
        value = input_dict[col]
        if value in le.classes_:
            input_dict[col] = le.transform([value])[0]
        else:
            input_dict[col] = 0  # fallback

    return pd.DataFrame([input_dict])

@app.post("/predict")
def predict_deposit(data: TermDepositInput):
    processed = preprocess_input(data)
    prediction = rf_model.predict(processed)[0]
    result = "Subscribed (y=1)" if prediction == 1 else " Not Subscribed (y=0)"
    return {"prediction": result}


# Webserver -> Uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080)
