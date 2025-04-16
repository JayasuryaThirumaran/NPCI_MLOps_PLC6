import sys
from pathlib import Path
filepath = Path(__file__)
sys.path.append(str(filepath.parents[1]))

from predict import rf_model, make_prediction, sample_input_df

from data_preprocessing import X_test, y_test

from sklearn.metrics import accuracy_score


# def test_model_accuracy():
#     # ADD YOUR TEST CASE to check for model test accuracy > 0.8
#     assert False  # <-- ADD YOUR ASSERT STATEMENT and remove `False``

# def test_make_prediction_function():
#     # ADD YOUR TEST CASE to check the output from make_prediction(), should be either "Subscribe (y=1)", or "Not Subscribe (y=0)"
#     assert False  # <-- ADD YOUR ASSERT STATEMENT and remove `False``


def test_model_accuracy():
    pred = rf_model.predict(X_test)
    acc = accuracy_score(pred, y_test)
    assert acc > 0.8, "Model accuracy is below 0.8"


def test_make_prediction_function():
    label = make_prediction(sample_input_df)
    assert label in ["Subscribe (y=1)", "Not Subscribe (y=0)"], "ErrorMessage: mismatch in prediction label sting"
