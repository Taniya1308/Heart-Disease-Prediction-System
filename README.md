# 🩺 Heart Disease Prediction System

A machine learning-based web application that predicts the likelihood of heart disease using patient clinical information.

The project uses a **Random Forest Classifier**, **FastAPI** backend, and **Streamlit** frontend.

> ⚠️ This project is for educational and research purposes only. It is not a medical diagnostic tool and should not replace professional medical advice.

## Features

- Heart disease prediction using Machine Learning
- Random Forest Classification
- FastAPI backend
- Streamlit frontend
- Prediction probability score
- Interactive patient input form

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- FastAPI
- Streamlit
- Joblib

## Dataset

The model uses a heart disease dataset containing 1025 records and 13 clinical features.

Features:

```text
age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal
```

Target:

```text
0 → No Heart Disease
1 → Heart Disease
```

## Model Performance

- Test Accuracy: **95.61%**
- Precision: **94.44%**
- Recall: **97.14%**
- F1 Score: **95.77%**
- ROC-AUC: **98.29%**

## Project Structure

```text
Heart-Disease-Prediction-System/
│
├── backend/
│   ├── main.py
│   ├── predictor.py
│   └── training.py
│
├── frontend/
│   └── app.py
│
├── dataset/
│   └── heart.csv
│
├── model_dir/
│   └── heart_disease_prediction_model.joblib
│
├── logs/
│   └── app.log
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Run the Project

Train the model:

```bash
python -m backend.training
```

Start the FastAPI backend:

```bash
python -m uvicorn backend.main:app --reload
```

Start the Streamlit frontend:

```bash
streamlit run frontend/app.py
```

Open the application in your browser and enter the patient information to get the prediction.

## Author

**Taniya Sharma**

B.Tech Student | Machine Learning & Full-Stack Development

GitHub: `<YOUR_GITHUB_PROFILE>`

LinkedIn: `<YOUR_LINKEDIN_PROFILE>`
