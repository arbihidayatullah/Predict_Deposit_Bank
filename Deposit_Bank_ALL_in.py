import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'bank-full.csv'
df = pd.read_csv(file_path, delimiter=';')

# Check the first few rows of the dataframe to understand its structure
st.write("First few rows of the dataset:")
st.write(df.head())

# Check the columns of the dataframe
st.write("Columns of the dataset:")
st.write(df.columns)

# Ensure the target column is present
if 'y' not in df.columns:
    st.error("'y' column not found in the dataset!")
else:
    # Encode categorical features
    df = df.apply(LabelEncoder().fit_transform)

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

    # Streamlit app
    st.title('Bank Marketing Prediction')
    st.write('This application predicts if a client will subscribe to a term deposit.')

    # Define all features
    all_features = [
        'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
        'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
    ]

    # Define the features for user input (selected features)
    selected_features = ['duration', 'pdays', 'poutcome', 'month', 'contact']

    # Input fields for the selected features
    duration = st.number_input('Duration', min_value=0, max_value=5000, step=1)
    pdays = st.number_input('Pdays', min_value=-1, max_value=1000, step=1)
    poutcome = st.selectbox('Poutcome', [0, 1, 2, 3], format_func=lambda x: {0: 'unknown', 1: 'other', 2: 'failure', 3: 'success'}[x])
    month = st.selectbox('Month', list(range(1, 13)), format_func=lambda x: {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}[x])
    contact = st.selectbox('Contact', [0, 1, 2], format_func=lambda x: {0: 'unknown', 1: 'cellular', 2: 'telephone'}[x])

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

    # Prediction button
    if st.button('Prediksi'):
        prediksi = clf2.predict(input_data)
        prediksi_proba = clf2.predict_proba(input_data)

        # Determine prediction and set the border and color accordingly
        if prediksi[0] == 1:
            st.markdown(
                f"""
                <div style="border:2px solid green; padding: 10px; border-radius: 10px;">
                    <h2 style="color: green;">Prediksi: Ya</h2>
                    <p>Ada peluang sebesar <b>{prediksi_proba[0][1]*100:.2f}%</b> bahwa klien akan berlangganan deposito.</p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="border:2px solid red; padding: 10px; border-radius: 10px;">
                    <h2 style="color: red;">Prediksi: Tidak</h2>
                    <p>Ada peluang sebesar <b>{prediksi_proba[0][0]*100:.2f}%</b> bahwa klien tidak akan berlangganan deposito.</p>
                </div>
                """, unsafe_allow_html=True
            )

        # Display prediction probabilities in detail
        st.write("Detail Probabilitas Prediksi:")
        prob_df = pd.DataFrame(prediksi_proba, columns=['Tidak', 'Ya'], index=['Probabilitas'])
        st.table(prob_df)

        # Conclusion paragraph
        st.write("\n\n")
        st.markdown(
            """
            **Kesimpulan:** Berdasarkan data input yang diberikan, model memprediksi apakah seorang klien akan berlangganan deposito atau tidak. Tabel probabilitas prediksi detail di atas menunjukkan probabilitas dari kedua hasil.
            """
        )
