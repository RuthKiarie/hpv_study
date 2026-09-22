"""
Runs INSIDE the SageMaker Processing container (AWS-managed scikit-learn
image -- no custom Docker build). SageMaker mounts S3 input at
/opt/ml/processing/input and uploads anything written to
/opt/ml/processing/output/* back to S3 automatically.

This is state (2) of the pipeline: validate data, split train/test,
compute basic drift/summary stats.
"""

import glob
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = "/opt/ml/processing/input"
TRAIN_OUT = "/opt/ml/processing/output/train"
TEST_OUT = "/opt/ml/processing/output/test"
REPORT_OUT = "/opt/ml/processing/output/report"

EXPECTED_COLUMNS = [
    "person_id", "age", "education", "residence", "county", "wealth_quintile",
    "health_insurance", "hiv_status", "parity", "distance_problem",
    "heard_of_cervical_cancer", "screened_last_3yrs",
    "education_ord", "wealth_ord", "is_urban", "hiv_positive", "hiv_unknown",
    "distance_barrier", "age_group", "parity_capped", "high_risk_flag",
]


def load_feature_data(input_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def validate(df: pd.DataFrame) -> dict:
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    null_counts = df[EXPECTED_COLUMNS].isnull().sum() if not missing_cols else {}
    return {
        "row_count": int(len(df)),
        "missing_columns": missing_cols,
        "null_counts": {k: int(v) for k, v in null_counts.items()} if not missing_cols else {},
        "passed": len(missing_cols) == 0 and len(df) > 0,
    }


def compute_drift_stats(df: pd.DataFrame) -> dict:
    return {
        "screened_last_3yrs_prevalence": round(float(df["screened_last_3yrs"].mean()), 4),
        "high_risk_flag_count": int(df["high_risk_flag"].sum()),
        "high_risk_flag_share": round(float(df["high_risk_flag"].mean()), 4),
        "mean_age": round(float(df["age"].mean()), 2),
        "hiv_positive_share": round(float(df["hiv_positive"].mean()), 4),
        "is_urban_share": round(float(df["is_urban"].mean()), 4),
        "mean_education_ord": round(float(df["education_ord"].mean()), 3),
        "mean_wealth_ord": round(float(df["wealth_ord"].mean()), 3),
    }


def main():
    os.makedirs(TRAIN_OUT, exist_ok=True)
    os.makedirs(TEST_OUT, exist_ok=True)
    os.makedirs(REPORT_OUT, exist_ok=True)

    df = load_feature_data(INPUT_DIR)
    print(f"Loaded {len(df)} rows from feature zone")

    validation = validate(df)
    print("Validation result:", validation)
    if not validation["passed"]:
        raise ValueError(f"Data validation failed: {validation}")

    drift_stats = compute_drift_stats(df)
    print("Drift/summary stats:", drift_stats)

    # stratified split to preserve the ~17% positive class balance in both sets
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["screened_last_3yrs"]
    )

    train_df.to_csv(os.path.join(TRAIN_OUT, "train.csv"), index=False)
    test_df.to_csv(os.path.join(TEST_OUT, "test.csv"), index=False)

    report = {
        "validation": validation,
        "drift_stats": drift_stats,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_prevalence": round(float(train_df["screened_last_3yrs"].mean()), 4),
        "test_prevalence": round(float(test_df["screened_last_3yrs"].mean()), 4),
    }
    with open(os.path.join(REPORT_OUT, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Report:", json.dumps(report, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()