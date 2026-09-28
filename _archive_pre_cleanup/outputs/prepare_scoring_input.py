"""
Prepares the Batch Transform input file: person_id + FEATURE_COLS, in that
order, NO header row (the SageMaker sklearn container's default CSV parser
assumes headerless, positional columns).

Score the held-out test set here as a first validation pass -- the real
pipeline will eventually score the full current feature set instead.
"""

import pandas as pd
from train import FEATURE_COLS

SOURCE = "local_test/test/test.csv"
DEST = "local_test/scoring_input/test_for_scoring.csv"

if __name__ == "__main__":
    df = pd.read_csv(SOURCE)
    scoring_df = df[["person_id"] + FEATURE_COLS]

    import os
    os.makedirs("local_test/scoring_input", exist_ok=True)
    scoring_df.to_csv(DEST, index=False, header=False)

    print(f"Wrote {DEST}  |  shape={scoring_df.shape}  |  columns={scoring_df.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(scoring_df.head(3).to_string(index=False, header=False))