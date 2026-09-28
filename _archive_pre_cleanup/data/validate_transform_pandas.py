"""
Local validation of the feature-engineering logic that will later be ported
into the real AWS Glue PySpark ETL job.

Why this exists: no AWS access / no PySpark in this dev environment, so we
prove the transformation logic is correct here (fast, easy to inspect with
pandas) BEFORE porting it 1:1 into the actual Glue job script. The Glue
script should produce the same output columns/values as this.
"""

import glob
import pandas as pd

RAW_GLOB = "s3_sim/raw/kenya_hpv_screening/dt=*/kenya_hpv_screening_synthetic.csv"

EDU_ORDER = {"none": 0, "primary": 1, "secondary": 2, "higher": 3}
WEALTH_ORDER = {"poorest": 0, "poorer": 1, "middle": 2, "richer": 3, "richest": 4}


def age_to_group(age: int) -> str:
    if age < 20:
        return "15-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    else:
        return "40-49"


def transform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["education_ord"] = out["education"].map(EDU_ORDER)
    out["wealth_ord"] = out["wealth_quintile"].map(WEALTH_ORDER)
    out["is_urban"] = (out["residence"] == "urban").astype(int)
    out["hiv_positive"] = (out["hiv_status"] == "positive").astype(int)
    out["hiv_unknown"] = (out["hiv_status"] == "unknown").astype(int)
    out["distance_barrier"] = (out["distance_problem"] == "big_problem").astype(int)
    out["age_group"] = out["age"].apply(age_to_group)
    out["parity_capped"] = out["parity"].clip(upper=6)

    # Deliberate feature: HIV+ women NOT yet screened -- the population a
    # risk-scoring model should actively flag for outreach.
    out["high_risk_flag"] = (
        (out["hiv_positive"] == 1) & (out["screened_last_3yrs"] == 0)
    ).astype(int)

    return out


if __name__ == "__main__":
    files = glob.glob(RAW_GLOB)
    assert files, f"No raw files found matching {RAW_GLOB}"
    print(f"Reading raw file(s): {files}")

    df_raw = pd.read_csv(files[0])
    print(f"Raw shape: {df_raw.shape}")

    df_feat = transform(df_raw)
    print(f"Feature shape: {df_feat.shape}  (+{df_feat.shape[1] - df_raw.shape[1]} engineered columns)")

    print("\n--- new/engineered columns, first 10 rows ---")
    new_cols = ["education_ord", "wealth_ord", "is_urban", "hiv_positive",
                "hiv_unknown", "distance_barrier", "age_group", "parity_capped",
                "high_risk_flag"]
    print(df_feat[new_cols].head(10).to_string())

    print("\n--- sanity checks ---")
    print("education_ord value counts:\n", df_feat["education_ord"].value_counts().sort_index())
    print("\nage_group value counts:\n", df_feat["age_group"].value_counts())
    print("\nhigh_risk_flag count (HIV+ AND unscreened):", df_feat["high_risk_flag"].sum())
    print("  as share of all rows:", round(df_feat["high_risk_flag"].mean(), 4))
    print("  as share of HIV+ rows:",
          round(df_feat.loc[df_feat["hiv_positive"] == 1, "high_risk_flag"].mean(), 4))

    # write to a local stand-in for the feature zone, partitioned by dt
    # (mirrors what the Glue job will do in S3, just on local disk)
    out_dir = "s3_sim/features/kenya_hpv_screening/dt=2026-09-19"
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/part-00000.parquet"
    df_feat.to_parquet(out_path, index=False)
    print(f"\nWrote feature-zone preview to {out_path}")