import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================================================
# FEATURE COLUMNS
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

TARGET_COLUMN = "target"


def train_model():

    # =====================================================
    # LOAD ENVIRONMENT VARIABLES
    # =====================================================

    load_dotenv()

    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

    DATASET_PATH = (
        PROJECT_ROOT
        / os.getenv("DATASET_DIR")
        / os.getenv("DATASET_NAME")
    )

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

    TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))


    # =====================================================
    # CREATE DIRECTORIES
    # =====================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # LOGGING
    # =====================================================

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH)
        ],
        force=True
    )


    try:

        # =================================================
        # LOAD DATASET
        # =================================================

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}"
            )

        df = pd.read_csv(DATASET_PATH)

        logging.info(
            f"Dataset loaded successfully: {df.shape}"
        )


        # =================================================
        # CHECK REQUIRED COLUMNS
        # =================================================

        required_columns = FEATURE_COLUMNS + [
            TARGET_COLUMN
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )


        # =================================================
        # SELECT FEATURES IN EXACT ORDER
        # =================================================

        X = df[FEATURE_COLUMNS].copy()

        y = df[TARGET_COLUMN].copy()


        # =================================================
        # DISPLAY TARGET DISTRIBUTION
        # =================================================

        print("\nTarget Distribution:")
        print(y.value_counts().sort_index())


        # =================================================
        # TRAIN TEST SPLIT
        # =================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )


        logging.info(
            f"Training samples: {len(X_train)}"
        )

        logging.info(
            f"Testing samples: {len(X_test)}"
        )


        # =================================================
        # RANDOM FOREST MODEL
        # =================================================

        random_forest = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None
        )


        # =================================================
        # PIPELINE
        # =================================================

        pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    random_forest
                )
            ]
        )


        # =================================================
        # TRAIN
        # =================================================

        pipeline.fit(
            X_train,
            y_train
        )

        logging.info(
            "Model training completed."
        )


        # =================================================
        # PREDICTIONS
        # =================================================

        y_train_pred = pipeline.predict(
            X_train
        )

        y_test_pred = pipeline.predict(
            X_test
        )


        # =================================================
        # PROBABILITY
        # =================================================

        y_test_probability = pipeline.predict_proba(
            X_test
        )[:, 1]


        # =================================================
        # METRICS
        # =================================================

        train_accuracy = accuracy_score(
            y_train,
            y_train_pred
        )

        test_accuracy = accuracy_score(
            y_test,
            y_test_pred
        )

        precision = precision_score(
            y_test,
            y_test_pred
        )

        recall = recall_score(
            y_test,
            y_test_pred
        )

        f1 = f1_score(
            y_test,
            y_test_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_test_probability
        )


        # =================================================
        # PRINT RESULTS
        # =================================================

        print("\n================================")
        print("MODEL PERFORMANCE")
        print("================================")

        print(
            f"Train Accuracy : {train_accuracy:.4f}"
        )

        print(
            f"Test Accuracy  : {test_accuracy:.4f}"
        )

        print(
            f"Precision       : {precision:.4f}"
        )

        print(
            f"Recall          : {recall:.4f}"
        )

        print(
            f"F1 Score        : {f1:.4f}"
        )

        print(
            f"ROC-AUC         : {roc_auc:.4f}"
        )


        # =================================================
        # CONFUSION MATRIX
        # =================================================

        print("\n================================")
        print("CONFUSION MATRIX")
        print("================================")

        print(
            confusion_matrix(
                y_test,
                y_test_pred
            )
        )


        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        print("\n================================")
        print("CLASSIFICATION REPORT")
        print("================================")

        print(
            classification_report(
                y_test,
                y_test_pred
            )
        )


        # =================================================
        # SAVE MODEL
        # =================================================

        dump(
            pipeline,
            MODEL_PATH
        )

        logging.info(
            f"Model saved at: {MODEL_PATH}"
        )


        # =================================================
        # SAVE FEATURE ORDER
        # =================================================

        print("\nModel Feature Order:")

        for index, feature in enumerate(
            FEATURE_COLUMNS,
            start=1
        ):
            print(
                f"{index}. {feature}"
            )


        print(
            "\nModel successfully trained and saved."
        )


    except Exception as e:

        logging.exception(
            f"Training failed: {e}"
        )

        raise


if __name__ == "__main__":
    train_model()