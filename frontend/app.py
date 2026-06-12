import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="HeartCare.ML",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 HeartCare.ML")
st.write("Heart Disease Risk Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 52)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.number_input("Chest Pain Type", 0, 3, 0)
    trestbps = st.number_input("Resting Blood Pressure", 0, 250, 125)
    chol = st.number_input("Cholesterol", 0, 600, 212)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Resting ECG", 0, 2, 1)
    thalach = st.number_input("Maximum Heart Rate", 0, 250, 168)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])

with col3:
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.number_input("Slope", 0, 2, 2)
    ca = st.number_input("Major Vessels", 0, 4, 0)
    thal = st.number_input("Thal", 0, 3, 2)

if st.button("🔍 Predict"):

    input_data = {
        "age": int(age),
        "sex": int(sex),
        "cp": int(cp),
        "trestbps": int(trestbps),
        "chol": int(chol),
        "fbs": int(fbs),
        "restecg": int(restecg),
        "thalach": int(thalach),
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal)
    }

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        prediction = result["prediction"]
        probability = result["probability"]
        diagnosis = result["diagnosis"]

        st.divider()

        st.metric(
            "Heart Disease Probability",
            f"{probability:.2f}"
        )

        if prediction == 1:
            st.error(f"❗ {diagnosis}")
        else:
            st.success(f"✅ {diagnosis}")

    except requests.exceptions.ConnectionError:
        st.error("Backend server is not running.")

    except requests.exceptions.Timeout:
        st.error("Request timed out.")

    except Exception as e:
        st.error(str(e))