# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
file_path = 'bank-full.csv'
df = pd.read_csv(file_path)

# Prepare the data
X = df.drop(columns=['y'])
y = df['y']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=123)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the Random Forest model with the selected hyperparameters
clf2 = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=42)
clf2.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(clf2, 'random_forest_model.pkl')
