# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:55:22 2025

@author: DELL
"""

import streamlit as st
import pickle
import pandas as pd

# === Load Model and Preprocessors ===
with open('titanic_model.sav', 'rb') as f:
    model = pickle.load(f)

with open('titanic_scaler.sav', 'rb') as f:
    scaler = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

st.title("🚢 Titanic Survival Prediction App")

# === Input Fields ===
passenger_id = st.number_input("Passenger ID", min_value=1, value=1001)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
name = st.text_input("Name")  # ✅ Any name allowed
sex = st.selectbox("Sex", ['male', 'female'])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
ticket = st.text_input("Ticket Number", 'A/5 21171')  # ✅ Any ticket
fare = st.number_input("Fare Paid", min_value=0.0, value=7.25)
embarked = st.selectbox("Embarked", ['S', 'C', 'Q'])

if st.button("Predict Survival"):

    try:
        # Build DataFrame
        input_data = {
            'PassengerId': passenger_id,
            'Pclass': pclass,
            'Name': name,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Ticket': ticket,
            'Fare': fare,
            'Embarked': embarked
        }

        df = pd.DataFrame([input_data])

        # ✅ Encode categorical columns (handles unknowns as -1)
        for col in ['Sex', 'Embarked', 'Ticket', 'Name']:
            if col in encoders:
                df[col] = encoders[col].transform(df[[col]])

        # ✅ Ensure correct column order
        ordered_cols = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age',
                        'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']
        df = df[ordered_cols]

        # ✅ Scale
        X_scaled = scaler.transform(df)

        # ✅ Predict
        prediction = model.predict(X_scaled)
        prob = model.predict_proba(X_scaled)[0]

        # ✅ Output
        if prediction[0] == 1:
            st.success(f"✅ Survived! (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.error(f"🚫 Did Not Survive (Confidence: {prob[0]*100:.2f}%)")

    except Exception as e:
        st.warning(f"⚠️ Error: {e}")
