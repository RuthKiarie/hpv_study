import os
import sys
from pathlib import Path
from decimal import Decimal
import boto3
import pytest
from moto import mock_aws

# ---------------------------------------------------------------------------
# Path configuration: Add source subdirectories to Python's import path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
for sub in ("api", "lambda/write_results", "sagemaker"):
    path_to_add = ROOT / sub
    if str(path_to_add) not in sys.path:
        sys.path.insert(0, str(path_to_add))

# ---------------------------------------------------------------------------
# AWS / Moto environment configuration
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

@pytest.fixture(autouse=True)
def mock_dynamodb(aws_credentials):
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="hpv-risk-scores",
            KeySchema=[{"AttributeName": "person_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "person_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        # Seed all 4 items expected by test_api.py
        seed_items = [
            {
                "person_id": "KE1",
                "risk_score": Decimal("0.86"),
                "risk_tier": "high",
                "scored_at": "2026-01-01T00:00:00Z",
                "model_artifact_path": "s3://bucket/model.tar.gz",
            },
            {
                "person_id": "KE2",
                "risk_score": Decimal("0.91"),
                "risk_tier": "high",
                "scored_at": "2026-01-01T00:00:00Z",
                "model_artifact_path": "s3://bucket/model.tar.gz",
            },
            {
                "person_id": "KE3",
                "risk_score": Decimal("0.20"),
                "risk_tier": "low",
                "scored_at": "2026-01-01T00:00:00Z",
                "model_artifact_path": "s3://bucket/model.tar.gz",
            },
            {
                "person_id": "KE4",
                "risk_score": Decimal("0.45"),
                "risk_tier": "medium",
                "scored_at": "2026-01-01T00:00:00Z",
                "model_artifact_path": "s3://bucket/model.tar.gz",
            },
        ]
        for item in seed_items:
            table.put_item(Item=item)
        yield
