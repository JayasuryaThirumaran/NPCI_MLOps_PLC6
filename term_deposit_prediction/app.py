from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI()

class InputData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    day_of_week: str
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int

@app.post("/predict")
def get_prediction(data: InputData):
    data_dict = data.dict()
    result = predict(data_dict)
    return {"prediction": "yes" if result == 1 else "no"}
