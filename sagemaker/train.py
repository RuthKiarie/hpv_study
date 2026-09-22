"""
SageMaker script-mode training entry point (state 3 of the pipeline).

Runs inside the AWS-managed scikit-learn training container. SageMaker sets
SM_CHANNEL_TRAIN / SM_CHANNEL_TEST / SM_MODEL_DIR / SM_OUTPUT_DATA_DIR as
environment variables automatically based on the channels we configure when
launching the job -- argparse just reads them with sane local-testing
defaults.

IMPORTANT -- excludes high_risk_flag from features: it is defined as
(hiv_positive AND NOT screened_last_3yrs), which directly encodes the label.
Including it would be label leakage.
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)

FEATURE_COLS = [
    "age", "education_ord", "wealth_ord", "is_urban", "health_insurance",
    "hiv_positive", "hiv_unknown", "distance_barrier",
    "heard_of_cervical_cancer", "parity_capped",
]
TARGET_COL = "screened_last_3yrs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "local_test/train"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "local_test/test"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "local_test/model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "local_test/output"))
    return parser.parse_args()


def main():
    args = parse_args()

    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.test, "test.csv"))

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # P(screened_last_3yrs = 1)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    print("Evaluation metrics:", json.dumps(metrics, indent=2))

    # feature importance is directly interpretable for logistic regression --
    # worth surfacing for the report 
    coefficients = dict(zip(FEATURE_COLS, [round(float(c), 4) for c in model.coef_[0]]))
    print("Feature coefficients (log-odds):", json.dumps(coefficients, indent=2))

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, "evaluation.json"), "w") as f:
        json.dump({"metrics": metrics, "coefficients": coefficients}, f, indent=2)


if __name__ == "__main__":
    main()