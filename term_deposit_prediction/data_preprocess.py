import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
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

# Label encode multi-class categorical variables
label_encoders = {}

for col in ['job', 'marital', 'education', 'day_of_week', 'month']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save encoders if needed later (e.g., inference)

# Features and target
X = df.drop('y', axis=1)
y = df['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
