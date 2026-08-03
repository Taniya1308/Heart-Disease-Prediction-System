import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
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
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT")
).resolve()


MODEL_PATH = (
    PROJECT_ROOT
    / os.getenv("MODEL_DIR")
    / os.getenv("MODEL_NAME")
)


LOG_PATH = (
    PROJECT_ROOT
    / os.getenv("LOG_DIR")
    / os.getenv("LOG_NAME")
)


LOG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)


# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


model = load(
    MODEL_PATH
)


logging.info(
    f"Model loaded successfully from: {MODEL_PATH}"
)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(input_data: dict):

    # -----------------------------------------------
    # Create DataFrame
    # -----------------------------------------------

    df = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------
    # Validate columns
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

    df = df[
        FEATURE_COLUMNS
    ]


    # -----------------------------------------------
    # Model prediction
    # -----------------------------------------------

    prediction = int(
        model.predict(df)[0]
    )


    # -----------------------------------------------
    # Prediction probability
    # -----------------------------------------------

    probabilities = model.predict_proba(
        df
    )[0]


    # Find probability corresponding to class 1

    class_1_index = list(
        model.classes_
    ).index(1)


    probability = float(
        probabilities[class_1_index]
    )


    # -----------------------------------------------
    # Logging
    # -----------------------------------------------

    logging.info(
        f"Prediction={prediction} | "
        f"Probability={probability:.4f}"
    )


    # -----------------------------------------------
    # Return result
    # -----------------------------------------------

    return {

        "prediction": prediction,

        "probability": probability

    }