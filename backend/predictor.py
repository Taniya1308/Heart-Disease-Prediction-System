import logging
from pathlib import Path

import pandas as pd
from joblib import load


# =========================================================
# FEATURE ORDER
# =========================================================

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# =========================================================
# PROJECT PATHS
# =========================================================

# predictor.py is inside:
# project_root/backend/predictor.py
#
# parent        -> backend
# parent.parent -> project root

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model_dir"
    / "heart_disease_prediction_model.joblib"
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}"
    )


model = load(MODEL_PATH)

logger.info(
    f"Model loaded successfully from: {MODEL_PATH}"
)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(input_data: dict):

    # -----------------------------------------------
    # Create DataFrame
    # -----------------------------------------------

    df = pd.DataFrame([input_data])


    # -----------------------------------------------
    # Validate required columns
    # -----------------------------------------------

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )


    # -----------------------------------------------
    # Force exact feature order
    # -----------------------------------------------

    df = df[FEATURE_COLUMNS]


    # -----------------------------------------------
    # Model prediction
    # -----------------------------------------------

    prediction = int(
        model.predict(df)[0]
    )


    # -----------------------------------------------
    # Prediction probability
    # -----------------------------------------------

    probabilities = model.predict_proba(df)[0]

    class_1_index = list(
        model.classes_
    ).index(1)

    probability = float(
        probabilities[class_1_index]
    )


    # -----------------------------------------------
    # Logging
    # -----------------------------------------------

    logger.info(
        f"Prediction={prediction} | "
        f"Probability={probability:.4f}"
    )


    # -----------------------------------------------
    # Return result
    # -----------------------------------------------

    return {
        "prediction": prediction,
        "probability": probability,
    }