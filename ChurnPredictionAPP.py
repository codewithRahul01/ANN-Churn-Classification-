import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# load artifacts
model = tf.keras.models.load_model('/Users/rahuljangra/Downloads/CodeWithRahul01/model.h5')
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict churn.")

# Inputs
Geography = st.selectbox("Geography", list(one_hot_encoder.categories_[0]))
Gender = st.selectbox("Gender", list(label_encoder.classes_))
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance", value=0.0, format="%.2f")
credit_score = st.number_input("Credit Score", value=500.0, format="%.1f")
estimated_salary = st.number_input("Estimated Salary", value=0.0, format="%.2f")
tenure = st.slider("Tenure", 0, 10, 1)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict Churn"):
    try:
        # Build input dataframe (include Geography so we can drop it after OHE)
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [label_encoder.transform([Gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
            "Geography": [Geography],   # <-- include before dropping
        })

        # One-hot encode geography and concat
        geo_encoded = one_hot_encoder.transform(input_data[["Geography"]]).toarray()
        geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.drop('Geography', axis=1).reset_index(drop=True), geo_df], axis=1)

        # Ensure column order matches training scaler (optional but safer if you saved feature order)
        # If you saved feature order, use it. Example: feature_order = [...]
        # input_data = input_data[feature_order]

        # Scale and predict
        scaled = scaler.transform(input_data)
        pred = model.predict(scaled)
        p_churn = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0][1])  # handle sigmoid or softmax

        if p_churn > 0.5:
            st.error(f"The customer is likely to churn (probability = {p_churn:.2f})")
        else:
            st.success(f"The customer is unlikely to churn (probability of staying = {1 - p_churn:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # optional debug info:
        st.write("Input dataframe columns:", input_data.columns.tolist())
        st.write("Input dataframe values:", input_data.to_dict(orient='records')[0])
