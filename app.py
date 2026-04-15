import streamlit as st
import numpy as np
import pickle
# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
st.title(" Student Performance Prediction App")
st.write("Enter student details to predict performance category")
# Inputs
hours = st.slider("Hours Studied", 0, 12, 5)
previous = st.slider("Previous Scores", 0, 100, 50)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep = st.slider("Sleep Hours", 0, 12, 6)
papers = st.slider("Sample Papers Practiced", 0, 10, 3)
# Convert Yes/No
extra = 1 if extra == "Yes" else 0
# Prediction
if st.button("Predict"):
    try:
        input_data = np.array([[hours, previous, extra, sleep, papers]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        labels = ['Excellent', 'High', 'Low', 'Medium']

        st.success("Predicted Performance: " + labels[int(prediction[0])])

    except Exception as e:
        st.error(str(e))                                               
