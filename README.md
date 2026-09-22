# Kenya Cervical Cancer Screening and Cloud MLOps Pipeline
## 1. Overview
Cervical cancer remains a major health challenge in sub-Saharan Africa, yet preventive screening uptake varies widely due to access, education and health-system links. To build and test a robust production-grade MLOps platform prior to accessing real-world clinical data, I designed an end-to-end cloud pipeline orchestrating AWS Step Functions, AWS Glue and Amazon SageMaker. The platform is powered by a synthetic dataset of 17,000 simulated Kenyan women (ages 15–49) calibrated closely against national demographic benchmarks.

## 2. Why This Project

- Synthetic Data Engineering: Building a rigorous, reproducible 3-stage generation pipeline (predictors.py, outcome.py, finalize.py) that models complex demographic interactions, wealth disparities and clinical linkages (such as HIV care integration) matching Kenya Demographic and Health Survey (KDHS) marginals.

- Serverless ETL and Data Processing: Utilising AWS Glue for scalable, serverless data transformation and validation using Pandas.

- Cloud Orchestration and Automation: Designing AWS Step Functions state machines (statemachine.json) to automate the handoff between data extraction, Glue processing and SageMaker model training.

- Scalable Machine Learning: Training predictive models on Amazon SageMaker with secure IAM least-privilege trust policies.

- DevOps and Infrastructure-as-Code: Managing clean repository architectures, secure JSON access policies and modular pipeline scripts.

## 3. Tech Stack
- Languages & Core Libraries: Python | Pandas | Scikit-learn | Boto3

- Cloud & Orchestration: AWS Step Functions | AWS Glue | Amazon SageMaker | Amazon S3 | AWS IAM

- DevOps and Version Control: Git | GitHub | JSON Infrastructure Policies

## 4. Data Source
- Dataset Overview: A fully synthetic dataset of 17,000 simulated Kenyan women (ages 15–49). No real individual's data is used; every row is generated via a reproducible random-number generator.

## Generation Design:

    - Stage 1 (predictors.py): Samples demographics (age, education, residence, county, wealth quintile, insurance, HIV status, parity, distance barriers and cervical cancer awareness) approximating Kenya's population structure.

    - Stage 2 (outcome.py): Simulates screening uptake (screened_last_3yrs) using a logistic model where coefficients reflect published literature (e.g., strong uplift for HIV-positive status due to integrated care programs, calibrated to a ~17% national prevalence).

    - Stage 3 (finalize.py): Cleans debug metadata and outputs the final production-ready CSV.

Validated Subgroup Highlights: Overall prevalence 17.0% | HIV-positive uptake 40.3% vs. negative 15.8% | Higher education 25.1% vs. none 9.0%.

## 5. Pipeline Access and Repository
GitHub Repository: RuthKiarie/hpv_study

Cloud Infrastructure Status: Automated via AWS Step Functions state machine (hpv-mlops-pipeline), integrating Glue ETL jobs and SageMaker training steps.

## 6. Architecture and Pipeline
The system operates on a fully automated, event-triggered cloud architecture:

- Data Ingestion & Glue ETL: Raw synthetic records stored in Amazon S3 are processed and cleaned using serverless AWS Glue jobs.

- Orchestration: AWS Step Functions coordinates the pipeline sequence from data preparation to model training.

- SageMaker Training: Processed data is fed into Amazon SageMaker training jobs to build predictive classification models for screening uptake.

- Security & Governance: Strict execution roles and security boundaries are enforced using dedicated S3 and IAM trust policies (glue/, sagemaker/ and step_functions/).
