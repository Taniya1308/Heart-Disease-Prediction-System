import pandas as pd
from joblib import load


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


df = pd.read_csv(
    "dataset/heart.csv"
)


model = load(
    "model_dir/heart_disease_prediction_model.joblib"
)


X = df[
    FEATURE_COLUMNS
]

y = df[
    "target"
]


predictions = model.predict(
    X
)


result = pd.DataFrame({

    "actual": y,

    "predicted": predictions

})


print(
    result.head(20)
)


print(
    "\nConfusion Matrix:"
)

from sklearn.metrics import confusion_matrix

print(
    confusion_matrix(
        y,
        predictions
    )
)


print(
    "\nAccuracy:"
)

from sklearn.metrics import accuracy_score

print(
    accuracy_score(
        y,
        predictions
    )
)