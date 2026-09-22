"""
STEP 2 of the build: outcome variable (screened_last_3yrs).
 
Modeled as a logistic function of the predictors from step 1. Coefficients
are set to be DIRECTIONALLY and RELATIVELY consistent with the pattern
reported across Kenyan cervical-cancer-screening literature (education,
wealth, urban residence, insurance and knowledge all raise uptake; distance
lowers it), with one deliberately strong effect: HIV-positive status, which
in Kenya is associated with routine linkage into cervical screening through
HIV care programs -- reported as a notably larger effect than a "generic"
predictor bump.
 
The intercept is calibrated (via bisection) so the population-level
prevalence lands near the ~15-20% range reported nationally. This is a
simulation for pipeline development, not a re-estimation of any specific
paper's coefficients.
"""

import numpy as np
import pandas as pd
from synthetic_dataset.predictors import build_predictors, RNG_SEED

rng = np.random.default_rng(RNG_SEED + 1)   # separate stream from step 1


def compute_logit_no_intercept(df: pd.DataFrame) -> np.ndarray :
    edu_coef = {"none": 0.0, "primary": 0.30, "secondary": 0.60, "higher": 1.00}
    wealth_coef = {"poorest": 0.0, "poorer": 0.15, "middle": 0.35, "richer": 0.55, "richest": 0.85}
    hiv_coef = {"negative": 0.0, "unknown": 0.10, "positive": 1.60}  # <- strong linkage-to-care effect
 
    logit = np.zeros(len(df))
    logit += df["education"].map(edu_coef).to_numpy()
    logit += df["wealth_quintile"].astype(str).map(wealth_coef).to_numpy()
    logit += np.where(df["residence"] == "urban", 0.35, 0.0)
    logit += df["health_insurance"].to_numpy() * 0.50
    logit += df["hiv_status"].map(hiv_coef).to_numpy()
    logit += df["heard_of_cervical_cancer"].to_numpy() * 0.80
    logit += np.where(df["distance_problem"] == "big_problem", -0.45, 0.0)
    logit += 0.02 * (df["age"].to_numpy() - 15)          # slight rise with age
    logit += 0.06 * np.clip(df["parity"].to_numpy(), 0, 6)  # slight rise with health-system contact

    # unobserved individual heterogeneity, so the outcome isn't perfectly
    # deterministic given the observed predictors (as in real data)
    logit += rng.normal(loc=0.0, scale=0.6, size=len(df))
    return logit


def calibrate_intercept(logit_no_intercept: np.ndarray, target_prevalence: 
                        float,
                                lo=-4.0, hi=2.0, tol=1e-4, max_iter=60) -> float:
    def mean_p(b0):
        return 1/(1 + np.exp(-(logit_no_intercept + b0)))
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        m = mean_p(mid).mean()
        if abs(m - target_prevalence) < tol:
            return mid
        if m < target_prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_dataset(target_prevalence: float = 0.17):
    df = build_predictors()
    logit0 = compute_logit_no_intercept(df)
    intercept = calibrate_intercept(logit0, target_prevalence)
    p = 1 / (1 + np.exp(-(logit0 + intercept)))
    outcome_rng = np.random.default_rng(RNG_SEED + 2)
    screened = (outcome_rng.random(len(df)) < p).astype(int)
    df["screening_probability"] = p.round(4)  # kept for transparency/debugging
    df["screened_last_3yrs"] = screened
    return df, intercept



if __name__ == "__main__":
    df, intercept = build_dataset(target_prevalence=0.17)
    print(f"Calibrated intercept: {intercept:.4f}")
    print(f"Overall screening prevalence: {df['screened_last_3yrs'].mean():.3f}")
 
    print("\n--- prevalence by subgroup (sanity checks) ---")
    for col in ["hiv_status", "education", "wealth_quintile", "residence",
                "health_insurance", "heard_of_cervical_cancer", "distance_problem"]:
        print(f"\nby {col}:")
        print(df.groupby(col, observed=True)["screened_last_3yrs"].mean().round(3))
 
    df.drop(columns=["screening_probability"]).to_csv("step2_full_dataset_preview.csv", index=False)
    print("\nSaved step2_full_dataset_preview.csv, shape:", df.shape)
        