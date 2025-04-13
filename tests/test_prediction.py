import sys
from pathlib import Path
filepath = Path(__file__)
sys.path.append(str(filepath.parents[1]))

from predict import rf_model, make_prediction, sample_input_df

from data_preprocessing import X_test, y_test

from sklearn.metrics import accuracy_score


def test_model_accuracy():
    pred = rf_model.predict(X_test)
    acc = accuracy_score(pred, y_test)
    assert acc > 0.8, "Model accuracy is below 0.8"


def test_make_prediction_function():
    label = make_prediction(sample_input_df)
    assert label in ["Subscribed (y=1)", "Not Subscribed (y=0)"], "ErrorMessage: mismatch in prediction label sting"
