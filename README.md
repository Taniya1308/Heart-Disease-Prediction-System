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
├── .streamlit/
│   └── config.toml
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── predictor.py
│   └── training.py
│
├── frontend/
│   ├── __init__.py
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
├── env_template.txt
├── requirements.txt
└── README.md
```

## Deploy on Streamlit Community Cloud

The app is ready to deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) for free.

### Steps

1. **Push the repository to GitHub**

   Make sure the following files are committed and pushed:
   - `frontend/app.py`
   - `backend/predictor.py`
   - `model_dir/heart_disease_prediction_model.joblib`
   - `dataset/heart.csv`
   - `requirements.txt`
   - `.streamlit/config.toml`

   > The `.env` file is gitignored and should not be pushed. It is only needed locally for retraining the model.

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.

3. **Click "New app"** and fill in the fields:

   | Field | Value |
   |---|---|
   | Repository | `your-github-username/Heart-Disease-Prediction-System` |
   | Branch | `main` |
   | Main file path | `frontend/app.py` |

4. **Click "Deploy"** — Streamlit Cloud will install dependencies from `requirements.txt` and launch the app automatically.

The app will be live at a public URL like:
```
https://your-github-username-heart-disease-prediction-system-frontend-app-xxxx.streamlit.app
```

## Run Locally

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Train the model** (only needed if you don't have the saved model):

Copy `env_template.txt` to `.env` and fill in your project root path, then run:

```bash
python -m backend.training
```

**Start the Streamlit frontend:**

```bash
streamlit run frontend/app.py
```

**Start the FastAPI backend** (optional — only needed if using the REST API):

```bash
uvicorn backend.main:app --reload
```

Open `http://localhost:8501` for the Streamlit app, or `http://localhost:8000/docs` for the FastAPI interactive docs.

## Author

**Taniya Sharma**

B.Tech Student | Machine Learning & Full-Stack Development

GitHub: `<YOUR_GITHUB_PROFILE>`

LinkedIn: `<YOUR_LINKEDIN_PROFILE>`
