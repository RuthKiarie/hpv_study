"""
Runs LOCALLY -- launches processing_script.py inside a
SageMaker-managed scikit-learn container. Uses the sagemaker SDK so not to
have to hand-resolve the AWS-managed container's ECR image URI.
"""

import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

ROLE_ARN = "arn:aws:iam::041904915024:role/sm-hpv-processing-role"
REGION = "us-east-1"
BUCKET = "rwk-hpv-mlops"

boto_session = boto3.Session(region_name=REGION)
sm_session = sagemaker.Session(boto_session=boto_session)

processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="hpv-feature-processing",
    sagemaker_session=sm_session,
)

processor.run(
    code=f"s3://{BUCKET}/processing-code/processing_script.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{BUCKET}/features/kenya_hpv_screening/",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/train",
            destination=f"s3://{BUCKET}/processing-output/train/",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/test",
            destination=f"s3://{BUCKET}/processing-output/test/",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/report",
            destination=f"s3://{BUCKET}/processing-output/report/",
        ),
    ],
)

print("Processing job submitted and finished (or failed) -- see logs above.")