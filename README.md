# Kenya Cervical Cancer Screening — Synthetic Dataset

## What this is

A **fully synthetic** dataset of 17,000 simulated Kenyan women (ages 15–49),
generated to develop and test an MLOps pipeline (Step Functions + Glue +
SageMaker + DynamoDB) before real survey data access is available.

**No real individual's data is used anywhere in this file.** Every row is
produced by a random-number generator. Nothing here was extracted, scraped,
or derived from any specific person's record.


## How it was generated

Three-stage script, in order:

1. **`step1_predictors.py`** — samples demographic/access predictors
   (age, education, residence, county, wealth quintile, health insurance,
   HIV status, parity, distance-to-facility barrier, cervical-cancer
   awareness) from distributions approximating Kenya's population
   structure and KDHS-reported marginals.
2. **`step2_outcome.py`** — generates the outcome (`screened_last_3yrs`)
   from a logistic model over the step-1 predictors. Coefficients are set
   to be directionally and relatively consistent with the published
   literature (higher education/wealth/urban/insurance/knowledge raise
   uptake; distance barriers lower it), with a deliberately **strong**
   effect for HIV-positive status, reflecting Kenya's documented practice
   of linking HIV-positive women into cervical screening through HIV care
   programs. The intercept is calibrated by bisection so the
   population-level prevalence lands at ~17%, within the ~15–20% range
   reported nationally.
3. **`step3_finalize.py`** — drops the debug-only probability column and
   writes the final `kenya_hpv_screening_synthetic.csv`.

Random seed is fixed (42 for predictors, offset for outcome/noise), so the
dataset is fully reproducible by re-running the scripts.

## Schema

| Column | Type | Description |
|---|---|---|
| `person_id` | string | Synthetic ID (KE100000, KE100001, ...) |
| `age` | int | 15–49 |
| `education` | string | none / primary / secondary / higher |
| `residence` | string | urban / rural |
| `county` | string | One of 20 sampled Kenyan counties |
| `wealth_quintile` | ordered category | poorest → richest |
| `health_insurance` | int (0/1) | NHIF coverage proxy |
| `hiv_status` | string | negative / positive / unknown |
| `parity` | int | Number of births |
| `distance_problem` | string | big_problem / not_big_problem (access barrier) |
| `heard_of_cervical_cancer` | int (0/1) | Awareness/knowledge variable |
| `screened_last_3yrs` | int (0/1) | **Target variable** — screening uptake proxy |

## Validated subgroup patterns (from generation run)

- Overall prevalence: 17.0%
- HIV-positive: 40.3% vs. HIV-negative: 15.8%
- Education: 9.0% (none) → 25.1% (higher)
- Wealth: 9.9% (poorest) → 27.0% (richest)
- Urban 24.0% vs. rural 13.8%
- Insured 25.6% vs. uninsured 14.3%
- Aware of cervical cancer 21.0% vs. unaware 9.1%
- Distance a big problem: 11.7% vs. not a big problem: 19.8%

## Limitations

- This is a simulation for pipeline-development purposes, not a
  re-estimation of any single published study's exact coefficients.
- No geographic/county-level clustering effects beyond the urban/rural
  county-sampling nudge — real DHS data would show much richer
  county-to-county heterogeneity.
- Intended to be swapped for real, approved data (KDHS or otherwise) once
  available — this dataset should not be cited as real-world evidence.
