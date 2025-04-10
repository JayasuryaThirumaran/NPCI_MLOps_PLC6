import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

filepath = Path(__file__)
sys.path.append(str(filepath.parents[0]))

from predict import model as rf_model
from data_preprocessing import X_test, y_test
from predict import preprocess_input


sample_input = {
    "age": 35,
    "job": "technician",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 2000,
    "housing": "yes",
    "loan": "no",
    "day_of_week": "mon",
    "month": "may",
    "duration": 180,
    "campaign": 1,
    "pdays": -1,
    "previous": 0
}

def test_model_accuracy():
    pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    assert acc > 0.8, f"Model accuracy is below threshold: {acc:.3f}"

def test_f1_score():
    pred = rf_model.predict(X_test)
    f1 = f1_score(y_test, pred)
    assert f1 > 0.8, f"Model F1-score is below threshold: {f1:.3f}"

def test_make_prediction_function():
    processed = preprocess_input(sample_input)
    label = rf_model.predict(processed)[0]
    assert label in [0, 1], "Error: Prediction output not binary (0 or 1)"
