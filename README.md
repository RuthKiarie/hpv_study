# HPV Screening Risk Pipeline (Kenya) - MLOps + DevOps Portfolio Project

An end-to-end, orchestrated machine learning pipeline that predicts cervical-cancer-screening
risk from synthetic Kenyan demographic/health data; built to demonstrate real MLOps and DevOps
engineering practice, not just a trained model in a notebook.

**Everything in this repository runs for real in AWS.** Every component listed below was built,
deployed and independently verified against live AWS resources during development. Built in 2 weeks.

---

## What this project actually does

1. Generates a **fully synthetic** dataset modeling cervical-cancer-screening uptake in Kenya
   (see [`data/README.md`](data/README.md) for exactly how and why. This is not real patient
   or government data; see *Why synthetic data* below).
2. Cleans and feature-engineers it with an **AWS Glue** ETL job.
3. Validates the data and splits it for training with a **SageMaker Processing** job.
4. Conditionally retrains a **logistic regression** risk model with **SageMaker Training**,
   registering the current model's location in **SSM Parameter Store** as a lightweight model
   registry; so a "skip retraining, use the existing model" run always knows which model to use.
5. Scores records with **SageMaker Batch Transform**.
6. Writes risk scores to **DynamoDB** via a **Lambda** function.
7. Serves scores through a **FastAPI Lambda** and a **Streamlit dashboard**.
8. All of the above is orchestrated end-to-end by a single **Step Functions** state machine,
   triggered nightly by **EventBridge**.
9. The entire stack is also defined as code in **AWS CDK** (Python), with a **GitHub Actions**
   CI/CD pipeline that lints, tests, builds, synthesizes, shows a `cdk diff` and deploys behind
   a manual approval gate.

## Architecture

```
S3 (raw) → Glue ETL → S3 (features) → Glue Data Catalog / Athena
                                            │
                                            ▼
EventBridge (nightly) → Step Functions ─────┼─────────────────────────────┐
                            │                │                             │
                            ▼                ▼                             ▼
                    SageMaker Processing → Choice: retrain?         (skip) │
                            │              yes │            no             │
                            │                  ▼             ▼             │
                            │         SageMaker Training   SSM Parameter Store
                            │                  │            (current model) │
                            │                  └──────┬──────────┘         │
                            │                         ▼                    │
                            │                 SageMaker Model               │
                            │                         │                    │
                            └─────────────────────────┼────────────────────┘
                                                       ▼
                                          SageMaker Batch Transform
                                                       │
                                                       ▼
                                       Lambda (write_results) → DynamoDB
                                                       │
                                        ┌──────────────┴──────────────┐
                                        ▼                              ▼
                              Lambda (FastAPI) + API Gateway    Streamlit dashboard
                               / Function URL (see caveat)      (direct DynamoDB read)
```

## Tech stack

| Layer | Technology |
|---|---|
| Data engineering | AWS Glue (PySpark), S3, Glue Data Catalog, Athena |
| ML | SageMaker Processing / Training / Batch Transform, scikit-learn (AWS-managed containers — no custom Docker images) |
| Orchestration | Step Functions, EventBridge, SSM Parameter Store (model registry) |
| Serving | DynamoDB, Lambda, FastAPI, Mangum, Streamlit |
| Infrastructure as Code | AWS CDK (Python) |
| CI/CD | GitHub Actions, pytest, moto, ruff, actionlint |

## Repository structure

```
data/               Synthetic dataset generation (step1–step3) + methodology README
glue/               Glue ETL job source
sagemaker/          Processing, training (incl. inference functions), scoring-input prep
step_functions/      The orchestration state machine (statemachine.json)
lambda/              write_results Lambda (scores → DynamoDB)
api/                 FastAPI read-side app + Streamlit dashboard
cdk/                 Full infrastructure-as-code stack (parallel "v2" resources — see below)
deploy_assets/       Code/data files CDK deploys into S3 (built from source, not hand-edited)
scripts/             build_assets.sh, build_lambda_package.sh, setup_github_oidc.sh
tests/               Unit, API, state-machine-structure, and cross-file contract tests
.github/workflows/   CI/CD pipeline (lint → test → build → synth → diff → approve → deploy)
```

---

## Honest design decisions and trade-offs

This section exists on purpose. A project like this involves real engineering trade-offs and
documenting them accurately is more useful and more honest, than hiding them.

### No Docker, anywhere

Docker was deliberately excluded from this project from the start (difficult to install in the
target environment). Every place Docker would normally be the default solution has a Docker-free
alternative instead:

- **SageMaker Processing/Training/Batch Transform** use AWS's **pre-built, managed containers**
  (`sagemaker-scikit-learn`), resolved via the SDK or a manually-verified ECR image URI — never a
  custom-built image.
- **The FastAPI Lambda's compiled dependencies** (notably `pydantic_core`, which has a native Rust
  extension) are installed with `pip install --platform manylinux2014_x86_64 --python-version 3.12
  --implementation cp --abi cp312 --only-binary=:all:` — this forces pip to fetch
  Lambda-compatible Linux wheels on any host OS, without needing a matching container to build in.
  This exact approach is what `scripts/build_lambda_package.sh` automates, with an import-time
  check that catches a platform mismatch at build time instead of at runtime (this bug was hit
  once during development and cost real debugging time; the check exists specifically because
  of that).

### Public API access is blocked at the account level - documented, not hidden

The FastAPI read-side Lambda works correctly, verified with a full automated test suite and
direct `aws lambda invoke` calls returning correct data. **Public HTTPS access to it does not
currently work.**

Both routes to public access were tried and both failed identically:
- **API Gateway** (HTTP API, `AWS_PROXY` integration) - rebuilt from scratch twice, with every
  component (route, integration, stage, Lambda resource policy) independently verified correct.
  Consistently returned `500 Internal Server Error`.
- **Lambda Function URL** (`AuthType: NONE`) - resource policy verified correct via
  `aws lambda get-policy`. Returns `403 Forbidden` with `x-amzn-ErrorType: AccessDeniedException`
  directly from AWS Lambda's own authorization layer (confirmed via `curl -v`, not a network or
  client-side issue).

The consistent failure across two independent, unrelated AWS mechanisms; both specifically on
the *public/unauthenticated* access path, with everything else in this project (SageMaker, Glue,
DynamoDB, IAM) working correctly, points to an account-level restriction on publicly-invocable
Lambda endpoints, a known anti-abuse measure some AWS accounts have applied by default. This
wasn't conclusively provable from my side and is documented here as a
precisely diagnosed, evidence-backed constraint rather than an unexplained gap.

**Practical consequence**: the Streamlit dashboard connects to DynamoDB **directly via `boto3`**
using the my AWS credentials, rather than through the public API, a legitimate
architectural choice for an internal tool (not a workaround), documented explicitly in
`api/dashboard_data.py`.

### DynamoDB `Scan` vs. a Global Secondary Index

The `/scores?risk_tier=high` endpoint and the dashboard's tier filter both use a table `Scan`
with a filter expression, not an indexed `Query`, there's no GSI on `risk_tier`. At this
project's scale (~3,400 rows) this is fast and inexpensive. At real production scale (millions of
rows), a GSI on `risk_tier` would be the correct fix. Also worth knowing: DynamoDB's `Limit`
parameter caps rows *scanned*, not rows *returned after filtering*, so a tier-filtered request
can return fewer rows than requested even when more matches exist. Documented in code comments
at the point of use.

### Label leakage was deliberately excluded from training

`high_risk_flag` (HIV-positive AND not yet screened) is *derived from* the target variable
(`screened_last_3yrs`); including it as a model feature would leak the answer directly into the
input. It's excluded from `FEATURE_COLS` in `sagemaker/train.py` and `tests/test_train.py`
enforces this exclusion so it can't silently regress.

### Two parallel infrastructure generations (v1 manual, v2 CDK)

The pipeline was first built by hand (AWS CLI, one resource at a time) to validate the
architecture end-to-end before investing in infrastructure-as-code. The CDK stack in `cdk/`
deploys a **separate, independently-named set of resources** (distinct bucket, table, roles,
state machine) rather than adopting the original hand-built ones, deliberately, to prove the
whole architecture is reproducible as code without risking the already-working manual deployment.
`cdk/hpv_mlops_stack.py` includes explicit `assert` checks that fail the study loudly if any
old, hardcoded resource reference ever leaks into the new stack.

---

## How to reproduce this

### Prerequisites
- AWS account with CLI access configured
- Python 3.12, Node.js 20+ (for the CDK CLI)
- No Docker required, anywhere in this project

### One-time setup
```bash
pip install -r requirements-dev.txt
npm install -g aws-cdk
pip install -r cdk/requirements.txt

# Build the deployable assets from source
./scripts/build_assets.sh
./scripts/build_lambda_package.sh

# One-time per AWS account/region
cd cdk && cdk bootstrap && cd ..
```

### Deploy
```bash
cd cdk && cdk deploy && cd ..
```
Or push to `main`; GitHub Actions will lint, test, build, synthesize, show a `cdk diff` and
wait for manual approval before deploying (see *CI/CD*, below). One-time setup for that path:
```bash
GH_REPO=<your-username>/<your-repo> ./scripts/setup_github_oidc.sh
```
then add the printed role ARN as the `AWS_DEPLOY_ROLE_ARN` repository variable and create a
`production` environment with required reviewers, in GitHub repo Settings.

### Run the pipeline
```bash
aws stepfunctions start-execution \
  --state-machine-arn <StateMachineArn from CDK output> \
  --input '{"retrain_requested": true}'
```

### View results
```bash
cd api && streamlit run dashboard.py
```

## Testing & CI/CD

`pytest` (unit, API via `moto`-mocked DynamoDB, state-machine structural checks and cross-file
contract tests catching the exact seams that broke during development, e.g. training expecting
a column the ETL job never produces) all run in CI on every push/PR, with no AWS credentials
needed. Deploys are gated behind a manual approval on a protected GitHub environment, and the
post-deploy step smoke-tests the live API Lambda before considering the deploy complete.

## Known limitations

- Public HTTPS access to the API is blocked (see above); data access is via direct AWS
  credentials (CLI, `boto3`, or the Streamlit dashboard).
- No GSI on `risk_tier`; fine at current scale, documented as a production-scale improvement.
- The synthetic dataset models *screening uptake* as a proxy for HPV-preventive-care engagement,
  not literal HPV vaccination status; see `data/README.md` for the full reasoning.
- `ml.m5.large` is used throughout for simplicity; not cost-optimized for production traffic
  patterns.