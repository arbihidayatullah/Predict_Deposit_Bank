import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('bank-full-1.csv')
data = data.drop(columns=['poutcome'])

# Preprocessing
columns_to_drop = ['job_unknown', 'contact_unknown']
for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(col, axis=1)

data = data.drop(4)

Q1 = data['balance'].quantile(0.25)
Q3 = data['balance'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['balance'] >= Q1 - 1.5 * IQR) & (data['balance'] <= Q3 + 1.5 * IQR)]

Q1 = data['campaign'].quantile(0.10)
Q3 = data['campaign'].quantile(0.90)
IQR = Q3 - Q1
data = data[(data['campaign'] >= Q1 - 1.5 * IQR) & (data['campaign'] <= Q3 + 1.5 * IQR)]

Q1 = data['duration'].quantile(0.20)
Q3 = data['duration'].quantile(0.80)
IQR = Q3 - Q1
data = data[(data['duration'] >= Q1 - 1.5 * IQR) & (data['duration'] <= Q3 + 1.5 * IQR)]

Q1 = data['previous'].quantile(0.05)
Q3 = data['previous'].quantile(0.95)
IQR = Q3 - Q1
data = data[(data['previous'] >= Q1 - 1.5 * IQR) & (data['previous'] <= Q3 + 1.5 * IQR)]

Q1 = data['age'].quantile(0.25)
Q3 = data['age'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['age'] >= Q1 - 1.5 * IQR) & (data['age'] <= Q3 + 1.5 * IQR)]

cat_features = data.select_dtypes(include=['object', 'bool']).columns.values
data_encoded = pd.get_dummies(data, columns=['job', 'marital', 'contact', 'default', 'housing', 'loan'])

label_encoder = LabelEncoder()
data_encoded['y'] = label_encoder.fit_transform(data_encoded['y'])
data_encoded['month'] = label_encoder.fit_transform(data_encoded['month'])
data_encoded['education'] = label_encoder.fit_transform(data_encoded['education'])

X = data_encoded.drop(columns=['y'])
y = data_encoded['y']

smote = SMOTE(random_state=123)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train model
clf2 = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=42)
clf2.fit(X_resampled, y_resampled)

# Streamlit app
st.title("Bank Marketing Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=-10000, max_value=100000, value=0)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
duration = st.number_input("Duration", min_value=0, max_value=5000, value=100)
campaign = st.number_input("Campaign", min_value=1, max_value=100, value=1)
pdays = st.number_input("Pdays", min_value=-1, max_value=1000, value=-1)
previous = st.number_input("Previous", min_value=0, max_value=100, value=0)
month = st.selectbox("Month", options=list(range(12)), format_func=lambda x: ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'][x])
education = st.selectbox("Education", options=list(range(4)), format_func=lambda x: ['primary', 'secondary', 'tertiary', 'unknown'][x])
job = st.selectbox("Job", options=[
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
    'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
marital = st.selectbox("Marital Status", options=['divorced', 'married', 'single'])
contact = st.selectbox("Contact", options=['cellular', 'telephone'])
default = st.selectbox("Default", options=['yes', 'no'])
housing = st.selectbox("Housing Loan", options=['yes', 'no'])
loan = st.selectbox("Personal Loan", options=['yes', 'no'])

input_data = {
    'age': age,
    'balance': balance,
    'day': day,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'month': month,
    'education': education,
    f'job_{job}': 1,
    f'marital_{marital}': 1,
    f'contact_{contact}': 1,
    f'default_{default}': 1,
    f'housing_{housing}': 1,
    f'loan_{loan}': 1
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X_resampled.columns, fill_value=0)

if st.button('Predict'):
    prediction = clf2.predict(input_df)
    prediction_label = 'yes' if prediction[0] == 1 else 'no'
    st.write(f"Prediction: {prediction_label}")
