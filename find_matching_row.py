import pandas as pd

df = pd.read_csv("dataset/heart.csv")

input_data = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
}

FEATURE_COLUMNS = list(input_data.keys())

matches = df[
    (df[FEATURE_COLUMNS] == pd.Series(input_data)).all(axis=1)
]

print("Matching rows:")
print(matches)