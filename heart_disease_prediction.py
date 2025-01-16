import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)

# Convert input to NumPy array
features = np.array([[age, sex, chol]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("You may have heart disease. Consult a doctor.")
    else:
        st.success("You are healthy!")

