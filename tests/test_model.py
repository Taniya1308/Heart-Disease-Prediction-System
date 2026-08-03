import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("dataset/heart.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# =========================
# FEATURES
# =========================

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
    "thal"
]

X = df[FEATURE_COLUMNS]
y = df["target"]

# =========================
# LOAD MODEL
# =========================

model = load(
    "model_dir/heart_disease_prediction_model.joblib"
)

print("Model loaded successfully!")

# =========================
# PREDICT
# =========================

y_pred = model.predict(X)

# =========================
# CHECK PREDICTION DISTRIBUTION
# =========================

print("\n==============================")
print("ACTUAL CLASS DISTRIBUTION")
print("==============================")

print(y.value_counts())

print("\n==============================")
print("PREDICTED CLASS DISTRIBUTION")
print("==============================")

print(pd.Series(y_pred).value_counts())

# =========================
# CONFUSION MATRIX
# =========================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(confusion_matrix(y, y_pred))

# =========================
# ACCURACY
# =========================

print("\n==============================")
print("ACCURACY")
print("==============================")

print(accuracy_score(y, y_pred))

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y,
        y_pred
    )
)

# =========================
# SHOW SOME FALSE POSITIVES
# =========================

print("\n==============================")
print("FALSE POSITIVES")
print("==============================")

false_positive_indices = df[
    (y == 0) & (y_pred == 1)
].index

print(
    df.loc[
        false_positive_indices,
        FEATURE_COLUMNS + ["target"]
    ].head(10)
)

# =========================
# SHOW SOME TRUE NEGATIVES
# =========================

print("\n==============================")
print("CORRECT NO-DISEASE PREDICTIONS")
print("==============================")

true_negative_indices = df[
    (y == 0) & (y_pred == 0)
].index

print(
    df.loc[
        true_negative_indices,
        FEATURE_COLUMNS + ["target"]
    ].head(10)
)