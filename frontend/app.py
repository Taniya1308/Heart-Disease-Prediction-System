import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="HeartCare.ML",
    page_icon="🩺",
    layout="centered"
)

# =========================
# CUSTOM UI STYLE
# =========================
st.markdown("""
    <style>

    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    h1 {
        color: #4da3ff;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }

    p {
        text-align: center;
        color: #a0a0a0;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4da3ff, #1f6feb);
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px;
        border: none;
        margin-top: 10px;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1f6feb, #4da3ff);
        transform: scale(1.02);
        transition: 0.2s;
    }

    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #2a2f3a;
        padding: 15px;
        border-radius: 12px;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1>🩺 HeartGuard System</h1>", unsafe_allow_html=True)
st.markdown("<p>Data-driven Heart Disease Risk Prediction System</p>", unsafe_allow_html=True)

st.divider()

# =========================
# INPUT SECTION
# =========================
st.subheader("📋 Patient Information")

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

st.divider()

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Risk"):

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

    with st.spinner("Analyzing patient data..."):
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
                label="🫀 Heart Disease Probability",
                value=f"{probability:.2f}"
            )

            if prediction == 1:
                st.error(f"⚠️ {diagnosis}")
            else:
                st.success(f"✅ {diagnosis}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Make sure FastAPI is running.")

        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Try again.")

        except requests.exceptions.HTTPError as e:
            st.error(f"❌ HTTP Error: {e}")

        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")