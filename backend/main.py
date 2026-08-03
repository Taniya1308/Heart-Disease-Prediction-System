from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.predictor import predict


app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)


# =========================================================
# INPUT SCHEMA
# =========================================================

class HeartDiseaseInput(BaseModel):

    age: int = Field(
        ge=1,
        le=120
    )

    sex: int = Field(
        ge=0,
        le=1
    )

    cp: int = Field(
        ge=0,
        le=3
    )

    trestbps: float = Field(
        ge=0,
        le=300
    )

    chol: float = Field(
        ge=0,
        le=700
    )

    fbs: int = Field(
        ge=0,
        le=1
    )

    restecg: int = Field(
        ge=0,
        le=2
    )

    thalach: float = Field(
        ge=0,
        le=300
    )

    exang: int = Field(
        ge=0,
        le=1
    )

    oldpeak: float = Field(
        ge=0,
        le=20
    )

    slope: int = Field(
        ge=0,
        le=2
    )

    ca: int = Field(
        ge=0,
        le=4
    )

    thal: int = Field(
        ge=0,
        le=3
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok"
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict-heart-disease")
def predict_heart_disease(
    input_data: HeartDiseaseInput
):

    result = predict(
        input_data.model_dump()
    )


    prediction = result[
        "prediction"
    ]

    probability = result[
        "probability"
    ]


    if prediction == 1:

        diagnosis = (
            "Higher Risk Pattern Detected"
        )

    else:

        diagnosis = (
            "Lower Risk Pattern Detected"
        )


    return {

        "prediction": prediction,

        "probability": probability,

        "diagnosis": diagnosis

    }