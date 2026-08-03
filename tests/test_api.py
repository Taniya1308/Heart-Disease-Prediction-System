import requests

API_URL = "http://127.0.0.1:8000/predict-heart-disease"

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

response = requests.post(
    API_URL,
    json=input_data
)

print("Status Code:", response.status_code)
print("Response:", response.json())