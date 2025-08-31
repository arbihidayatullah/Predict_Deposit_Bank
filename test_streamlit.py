# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define all features
all_features = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

# Define the features for user input (selected features)
selected_features = ['duration', 'pdays', 'poutcome', 'month', 'contact']

# Streamlit app
st.title('Bank Marketing Prediction')
st.write('This application predicts if a client will subscribe to a term deposit.')

# Input fields for the selected features
duration = st.number_input('Duration', min_value=0, max_value=5000, step=1)
pdays = st.number_input('Pdays', min_value=-1, max_value=1000, step=1)
poutcome = st.selectbox('Poutcome', [1, 2, 3, 4], format_func=lambda x: {1: 'unknown', 2: 'other', 3: 'failure', 4: 'success'}[x])
month = st.selectbox('Month', list(range(1, 13)), format_func=lambda x: {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}[x])
contact = st.selectbox('Contact', [1, 2, 3], format_func=lambda x: {1: 'unknown', 2: 'cellular', 3: 'telephone'}[x])

# Create a dataframe for the input features
input_data = pd.DataFrame({
    'duration': [duration],
    'pdays': [pdays],
    'poutcome': [poutcome],
    'month': [month],
    'contact': [contact]
})

# Ensure all features are present
for feature in all_features:
    if feature not in input_data.columns:
        input_data[feature] = 0

# Reorder columns to match training data
input_data = input_data[all_features]

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Determine the prediction result and format the output accordingly
    if prediction[0] == 1:
        st.markdown(
            f"""
            <div style='border:2px solid green; padding:10px; border-radius:5px; background-color: #eaffea;'>
                <h2 style='color:green;'>Prediction: Yes</h2>
                <p style='color:green;'>The model predicts that the client will subscribe to a term deposit.</p>
                <p style='color:black;'>Prediction Probability: <strong>{prediction_proba[0][1]*100:.2f}%</strong> chance of subscribing to a term deposit (Yes)</p>
                <p style='color:black;'>Prediction Probability: <strong>{prediction_proba[0][0]*100:.2f}%</strong> chance of not subscribing to a term deposit (No)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='border:2px solid red; padding:10px; border-radius:5px; background-color: #ffeaea;'>
                <h2 style='color:red;'>Prediction: No</h2>
                <p style='color:red;'>The model predicts that the client will not subscribe to a term deposit.</p>
                <p style='color:black;'>Prediction Probability: <strong>{prediction_proba[0][0]*100:.2f}%</strong> chance of not subscribing to a term deposit (No)</p>
                <p style='color:black;'>Prediction Probability: <strong>{prediction_proba[0][1]*100:.2f}%</strong> chance of subscribing to a term deposit (Yes)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Detailed probability output
    st.write(f"Detailed Prediction Probability: {prediction_proba[0]}")

# To run the Streamlit application, use the command:
# streamlit run app.py
