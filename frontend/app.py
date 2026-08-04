import sys
from pathlib import Path

# Ensure the project root is on the path so `backend` resolves
# whether the app is run locally or from Streamlit Cloud
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from backend.predictor import predict


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HeartGuard ML",
    page_icon="🫀",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
        color: white;
    }

    h1 {
        color: #4da3ff;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: #a0a0a0;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(
            90deg,
            #4da3ff,
            #1f6feb
        );
        color: white;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<h1>🫀 HeartGuard ML</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>"
    "Machine Learning Based Heart Disease Risk Prediction"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.subheader(
    "📋 Patient Information"
)

st.write(
    "Enter the patient's clinical information below."
)


col1, col2, col3 = st.columns(3)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=52
    )

    sex_label = st.selectbox(
        "Sex",
        [
            "Female",
            "Male"
        ]
    )

    sex = (
        0
        if sex_label == "Female"
        else 1
    )

    cp = st.number_input(
        "Chest Pain Type",
        min_value=0,
        max_value=3,
        value=0
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0,
        max_value=300,
        value=125
    )

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=0,
        max_value=700,
        value=212
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [
            "No",
            "Yes"
        ]
    )

    fbs = (
        0
        if fbs_label == "No"
        else 1
    )

    restecg = st.number_input(
        "Resting ECG",
        min_value=0,
        max_value=2,
        value=1
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=0,
        max_value=300,
        value=168
    )

    exang_label = st.selectbox(
        "Exercise Induced Angina",
        [
            "No",
            "Yes"
        ]
    )

    exang = (
        0
        if exang_label == "No"
        else 1
    )


# =========================================================
# COLUMN 3
# =========================================================

with col3:

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.1
    )

    slope = st.number_input(
        "Slope",
        min_value=0,
        max_value=2,
        value=2
    )

    ca = st.number_input(
        "Major Vessels",
        min_value=0,
        max_value=4,
        value=0
    )

    thal = st.number_input(
        "Thal",
        min_value=0,
        max_value=3,
        value=2
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()


if st.button(
    "🔍 Analyze Heart Disease Risk"
):

    input_data = {

        "age": int(age),

        "sex": int(sex),

        "cp": int(cp),

        "trestbps": float(trestbps),

        "chol": float(chol),

        "fbs": int(fbs),

        "restecg": int(restecg),

        "thalach": float(thalach),

        "exang": int(exang),

        "oldpeak": float(oldpeak),

        "slope": int(slope),

        "ca": int(ca),

        "thal": int(thal)

    }


    with st.spinner(
        "Analyzing patient data..."
    ):

        try:

            # =========================================
            # DIRECT MODEL PREDICTION
            # =========================================

            result = predict(
                input_data
            )


            prediction = result[
                "prediction"
            ]

            probability = result[
                "probability"
            ]


            # =========================================
            # RESULT
            # =========================================

            st.divider()

            st.subheader(
                "📊 Prediction Result"
            )


            st.metric(
                "Heart Disease Probability",
                f"{probability * 100:.1f}%"
            )


            st.write(
                f"**Prediction Class:** "
                f"`{prediction}`"
            )


            # =========================================
            # DIAGNOSIS
            # =========================================

            if prediction == 1:

                st.warning(
                    "⚠️ Heart Disease Detected"
                )

            else:

                st.success(
                    "✅ Lower Risk Pattern Detected"
                )


            # =========================================
            # PATIENT DATA
            # =========================================

            with st.expander(
                "View Patient Data"
            ):

                st.json(
                    input_data
                )


        except Exception as e:

            st.error(
                f"❌ Prediction Error: {e}"
            )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ Medical Disclaimer: This application is "
    "for educational and research purposes only. "
    "It is not a medical diagnostic tool and should "
    "not replace professional medical advice."
)