"""
STEP 3 of the build: final, clean export.
 
Pulls the calibrated dataset from step 2, drops the debug-only
`screening_probability` column (that was only there so we could sanity-check
calibration) and writes the final CSV that the rest of the pipeline
(Glue / SageMaker / etc.) will actually consume.
"""

import pandas as pd
from synthetic_dataset.outcome import build_dataset

OUTPUT_PATH = "kenya_hpv_screening_synthetic.csv"

def finalize(target_prevalence: float=0.17):
    df, intercept = build_dataset(target_prevalence=target_prevalence)
    df_final = df.drop(columns=["screening_probability"])
    df_final.to_csv(OUTPUT_PATH, index=False)
    return df_final, intercept


if __name__ == "__main__":
    df_final, intercept = finalize()
    print(f"Wrote {OUTPUT_PATH}  |  shape={df_final.shape}  |  calibrated intercept={intercept:.4f}")
    print("\nColumns:")
    for c in df_final.columns:
        print(f"  - {c}: {df_final[c].dtype}")
    print("\nFirst 5 rows:")
    print(df_final.head().to_string())