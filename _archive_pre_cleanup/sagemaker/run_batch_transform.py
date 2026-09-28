"""
Runs LOCALLY -- launches a SageMaker Batch Transform job using the model
artifact from our completed training job.

Key mechanism: input_filter="$[1:]" strips column 0 (person_id) before the
row reaches predict_fn in train.py -- the model never sees person_id.
join_source="Input" then re-attaches the FULL original input row (including
person_id) to the prediction in the output file. This means train.py's
predict_fn never has to know person_id exists at all.
"""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

ROLE_ARN = "arn:aws:iam::041904915024:role/sm-hpv-transform-role"
REGION = "us-east-1"
BUCKET = "rwk-hpv-mlops"

# from the completed training job -- update if you retrain and get a new path
MODEL_DATA = "s3://rwk-hpv-mlops/training-output/hpv-risk-training-2026-09-22-12-38-21-529/output/model.tar.gz"

boto_session = boto3.Session(region_name=REGION)
sm_session = sagemaker.Session(boto_session=boto_session)

model = SKLearnModel(
    model_data=MODEL_DATA,
    role=ROLE_ARN,
    entry_point="train.py",
    framework_version="1.2-1",
    sagemaker_session=sm_session,
    code_location=f"s3://{BUCKET}/training-code",
)

transformer = model.transformer(
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{BUCKET}/batch-transform-output/",
    accept="text/csv",
    assemble_with="Line",
)

transformer.transform(
    data=f"s3://{BUCKET}/scoring-input/test_for_scoring.csv",
    content_type="text/csv",
    split_type="Line",
    input_filter="$[1:]",   # drop column 0 (person_id) before scoring
    join_source="Input",    # reattach full original row to the prediction
    wait=True,
    logs=True,
)

print("Batch transform finished. Output location:", transformer.output_path)