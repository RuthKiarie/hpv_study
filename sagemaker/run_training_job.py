"""
Runs LOCALLY -- launches train.py inside a SageMaker-managed scikit-learn
training container.

code_location is set explicitly so the SDK uploads train.py to the bucket
(which the IAM role has permissions on) rather than the SageMaker-managed
default bucket that caused the AccessDenied error during the Processing
job step.
"""

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

ROLE_ARN = "arn:aws:iam::041904915024:role/sm-hpv-training-role"
REGION = "us-east-1"
BUCKET = "rwk-hpv-mlops"

boto_session = boto3.Session(region_name=REGION)
sm_session = sagemaker.Session(boto_session=boto_session)

estimator = SKLearn(
    entry_point="train.py",
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    base_job_name="hpv-risk-training",
    sagemaker_session=sm_session,
    code_location=f"s3://{BUCKET}/training-code",
    output_path=f"s3://{BUCKET}/training-output",
)

estimator.fit({
    "train": f"s3://{BUCKET}/processing-output/train/",
    "test": f"s3://{BUCKET}/processing-output/test/",
})

print("Training job finished. Model artifact location:", estimator.model_data)