# MLOps Playground Challenge-6
# Term Deposit Subscription Prediction

## Problem Statement:
In this project, you'll work with a real-world banking dataset to build a complete machine learning pipeline for predicting whether a customer will subscribe to a term deposit. The project is structured into modular steps that reflect a real industry workflow, from raw data handling to serving predictions through a REST API, followed by containerizing the application for deployment.

## Dataset
The dataset provided (`term_deposit.csv`) contains information about bank clients contacted for marketing purposes. The target is column `y`, which indicates whether the client subscribed to a term deposit.

## Objective:
Build and structure a modular ML project with:

- Data preprocessing
- Model training and evaluation
- Prediction interface
- Test the model and prediction function
- FastAPI deployment
- MLflow integration


## Folder Structure

term_deposit_prediction/
│
├── dataset/
│   ├── term_deposit.csv
│   └── data_description.txt  
│
├── trained_model/
│   ├── rf_model.pkl
│   └── label_encoders.pkl
├── tests
│   └── test_prediction.py
├── data_preprocessing.py
├── train_model.py
├── predict.py
├── app.py
├── README.md
└── requirements.txt


## Tasks:

### Preprocess the Data (data_preprocessing.py)
- Read the CSV file
- Handle missing values
- Encode categorical columns (label encode and/or mapping)
- Split data into train and test sets
- Save label encoders in trained_model/

### Data Preprocessing

    - Load CSV data into pandas dataframe
    - handle nill and missing values
    - Encode categorical variables
    - Train-test split

### Model Training
    -  train diffrent classification models (e.g., Logistic Regression, Random Forest)
    -  Save the trained model

### Experiment Tracking with MLflow
    -  Log metrics, parameters, and model artifacts
    - Track different model runs for comparison

### Unit Testing with Pytest
    -  Create tests for each stage (data loading, preprocessing, training)
    -  Ensure pipeline robustness

### FastAPI
