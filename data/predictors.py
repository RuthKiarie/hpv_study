"""
Synthetic Kenya HPV-preventive-care (cervical cancer screening) dataset.

IMPORTANT: This is a SIMULATED dataset. No real individuals' data is used.
Marginal distributions and predictor relationships are set to approximate,
directionally-correct values based on the pattern reported across published
Kenyan cervical-cancer-screening literature (KDHS-based studies). Exact
figures are NOT lifted verbatim from any single table — see README for
sourcing notes and caveats.

STEP 1 of the build: demographic/predictor sampling only.
The outcome variable (screened_last_3yrs) is added in step 2, once we've
checked these distributions look sane.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ROWS = 17_000  # roughly matches KDHS 2022 IR sample size of women 15-49

rng = np.random.default_rng(RNG_SEED)

# A sample of Kenyan counties (mix of urban-heavy, rural, and mixed counties)
# Not all 47 included for now -- easy to extend later.
COUNTIES = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Uasin Gishu",
    "Kiambu", "Machakos", "Kilifi", "Kakamega", "Bungoma",
    "Meru", "Turkana", "Garissa", "Wajir", "Mandera",
    "Kitui", "Makueni", "Kajiado", "Narok", "Bomet",
]

def sample_age(n):
    # Women 15-49, roughly matching DHS age-group weighting (younger-skewed
    # population pyramid typical of Kenya)
    age_groups = [15, 20, 25, 30, 35, 40, 45]
    group_weights = [0.19, 0.18, 0.16, 0.14, 0.13, 0.11, 0.09]
    base = rng.choice(age_groups, size=n, p=group_weights)
    jitter = rng.integers(0, 5, size=n)
    return np.clip(base + jitter, 15, 49)

def sample_education(n):
    cats = ["none", "primary", "secondary", "higher"]
    weights = [0.08, 0.38, 0.40, 0.14]
    return rng.choice(cats, size=n, p=weights)

def sample_residence(n):
    cats = ["urban", "rural"]
    weights = [0.31, 0.69]  # Kenya is still majority-rural
    return rng.choice(cats, size=n, p=weights)

def sample_county(n, residence):
    # crude urban/rural-aware county sampling: nudge urban rows toward
    # urban-heavy counties, rural rows toward the rest
    urban_heavy = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Uasin Gishu", "Kiambu"]
    out = np.empty(n, dtype=object)
    for i in range(n):
        if residence[i] == "urban" and rng.random() < 0.6:
            out[i] = rng.choice(urban_heavy)
        else:
            out[i] = rng.choice(COUNTIES)
    return out

def sample_wealth_quintile(n, residence):
    cats = ["poorest", "poorer", "middle", "richer", "richest"]
    out = np.empty(n, dtype=object)
    urban_w = [0.06, 0.12, 0.18, 0.28, 0.36]
    rural_w = [0.27, 0.24, 0.20, 0.16, 0.13]
    for i in range(n):
        w = urban_w if residence[i] == "urban" else rural_w
        out[i] = rng.choice(cats, p=w)
    return out

def sample_health_insurance(n, wealth):
    # NHIF coverage rises with wealth quintile
    wealth_to_p = {"poorest": 0.08, "poorer": 0.13, "middle": 0.20, "richer": 0.32, "richest": 0.48}
    p = np.array([wealth_to_p[w] for w in wealth])
    return (rng.random(n) < p).astype(int)

def sample_hiv_status(n):
    cats = ["negative", "positive", "unknown"]
    weights = [0.92, 0.052, 0.028]  # ~5.2% adult HIV prevalence order of magnitude
    return rng.choice(cats, size=n, p=weights)

def sample_parity(n, age):
    # parity roughly increases with age; poisson-ish
    lam = np.clip((age - 15) / 8, 0.1, 5.5)
    return rng.poisson(lam)

def sample_distance_problem(n, residence):
    p_big_problem = np.where(residence == "urban", 0.18, 0.42)
    draws = rng.random(n) < p_big_problem
    return np.where(draws, "big_problem",  "not_big_problem")

def sample_heard_of_cc(n, education):
    edu_to_p = {"none": 0.35, "primary": 0.55, "secondary": 0.74, "higher": 0.88}
    p = np.array([edu_to_p[e] for e in education])
    return (rng.random(n) < p).astype(int)


def build_predictors(n=N_ROWS):
    age = sample_age(n)
    education = sample_education(n)
    residence = sample_residence(n)
    county = sample_county(n, residence)
    wealth = sample_wealth_quintile(n, residence)
    insurance = sample_health_insurance(n, wealth)
    hiv = sample_hiv_status(n)
    parity = sample_parity(n, age)
    distance = sample_distance_problem(n, residence)
    heard_cc = sample_heard_of_cc(n, education)

    df = pd.DataFrame({
        "person_id": [f"KE{100000+i}" for i in range(n)],
        "age": age,
        "education": education,
        "residence": residence,
        "county": county,
        "wealth_quintile": pd.Categorical(
            wealth, categories=["poorest", "poorer", "middle", "richer", "richest"], ordered=True
        ),
        "health_insurance": insurance,
        "hiv_status": hiv,
        "parity": parity,
        "distance_problem": distance,
        "heard_of_cervical_cancer": heard_cc,
    })
    return df


if __name__ == "__main__":
    df = build_predictors()
    print(df.shape)
    print(df.head(10).to_string())
    print("\n--- quick distribution checks ---")
    print(df["education"].value_counts(normalize=True).round(3))
    print(df["residence"].value_counts(normalize=True).round(3))
    print(df["wealth_quintile"].value_counts(normalize=True).round(3))
    print(df["health_insurance"].mean().round(3), "= share with insurance")
    print(df["hiv_status"].value_counts(normalize=True).round(3))
    print(df["heard_of_cervical_cancer"].mean().round(3), "= share who've heard of cervical cancer")
    df.to_csv("step1_predictors_preview.csv", index=False)