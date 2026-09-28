import os
import boto3
from moto import mock_aws
import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture(autouse=True)
def mock_aws_services():
    """Mock AWS services and create the DynamoDB table for testing."""
    with mock_aws():
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
        
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="hpv-risk-scores",
            KeySchema=[{"AttributeName": "person_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "person_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        
        # Insert sample test data if needed
        table.put_item(
            Item={
                "person_id": "KE1",
                "risk_score": 0.85,
                "risk_tier": "high"
            }
        )
        yield

@pytest.fixture
def client():
    return TestClient(app)