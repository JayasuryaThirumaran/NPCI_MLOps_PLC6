# MLOps Playground Challenge-6
## Term Deposit Subscription Prediction

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
```
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
```

## Tasks:

### 1. Preprocess the Data (`data_preprocessing.py`)

    - Read the CSV file
    - Handle missing values
    - Encode categorical columns (label encode and/or mapping)
    - Split data into train and test sets
    - Save label encoders in trained_model/

### 2. Train a Machine Learning Model (`train_model.py`)

    - Load preprocessed train-test data and encoders
    - Select and train an appropriate classification model (e.g., LogisticRegression, RandomForestClassifier, etc.)
    - Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
    - Save the trained model in trained_model/ directory
    - Log model hyperparameters, performance metrics, trained model, and any other artifact using MLflow for reproducibility and comparison

### 3. Make Predictions - Inference (`predict.py`)

    - Load the trained model and label encoders
    - Create a reusable inference function that takes new customer data, preprocess it, make prediction, and returns the predicted subscription outcome: "Subscribed (y=1)" or "Not Subscribed (y=0)"

### 4. Build Test Cases (`test/test_prediction.py`)

    - Write unit tests to verify:
      - Accuracy > 80%
      - Prediction function output validation
      - Existence of trained model and encoders
    - Use tools like `pytest`

### 5. Serve the Model via REST API using FastAPI (`app.py`)

    - Build a FastAPI application with a `/predict` endpoint
    - Accept customer input data in JSON format and return prediction results in real-time
    - Start the FastAPI application and make a prediction

### 6. Dockerize the FastAPI Application
- **Create a Dockerfile:** Write a `Dockerfile` to containerize the FastAPI application.

  The Dockerfile will include:
  - Base image with Python dependencies.
  - Installation of necessary libraries (FastAPI, scikit-learn, etc.).
  - Exposing the appropriate port for FastAPI.
  - Command to run the FastAPI app inside the container.

- **Build the Docker Image:** Run the `docker build` command to create a Docker image from the `Dockerfile`.

- **Run the Container:** Start a Docker container using the built image and run the FastAPI application, ensuring that the API is accessible and functioning as expected.

