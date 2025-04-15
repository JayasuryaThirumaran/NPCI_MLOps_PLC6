import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
# df = # YOUR CODE HERE to read the csv file from path "dataset/term_deposit.csv"
df = pd.read_csv("dataset/term_deposit.csv")

# Fill missing values
df['job'] = df['job'].fillna('unknown')
df['education'] = df['education'].fillna('unknown')

# Map binary categorical variables
binary_mapping = {'yes': 1, 'no': 0}
df['default'] = df['default'].map(binary_mapping)
df['housing'] = df['housing'].map(binary_mapping)
df['loan'] = df['loan'].map(binary_mapping)
df['y'] = df['y'].map(binary_mapping)

# Label encode other categorical columns
label_encoders = {}
for col in ['job', 'marital', 'education', 'day_of_week', 'month']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Save `label_encoders` as a pickle at path "trained_model/label_encoders.pkl"
# import joblib
# YOUR CODE HERE
joblib.dump(label_encoders, "trained_model/label_encoders.pkl")

# Split features and target
X = df.drop('y', axis=1)
y = df['y']

# Train-test split
# X_train, X_test, y_train, y_test = # YOUR CODE HERE to do train_test_split with test_size=0.2, stratify=y, random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
