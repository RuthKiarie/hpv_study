"""
Local validation harness -- NOT deployed. Uses moto to mock DynamoDB
in-process, seeds it with sample data matching our real schema, then hits
the FastAPI app directly via TestClient. Validates route logic before we
touch Lambda packaging at all.
"""

import os
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

from decimal import Decimal
from moto import mock_aws
import boto3

with mock_aws():
    # seed a mock table matching our real schema before importing app.py,
    # since app.py connects to the table at import time
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.create_table(
        TableName="hpv-risk-scores",
        KeySchema=[{"AttributeName": "person_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "person_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )

    sample_items = [
        {"person_id": "KE114061", "risk_score": Decimal("0.8616"), "risk_tier": "high",
         "scored_at": "2026-09-25T07:40:32+00:00", "model_artifact_path": "s3://bucket/model.tar.gz"},
        {"person_id": "KE109685", "risk_score": Decimal("0.6802"), "risk_tier": "high",
         "scored_at": "2026-09-25T07:40:32+00:00", "model_artifact_path": "s3://bucket/model.tar.gz"},
        {"person_id": "KE108740", "risk_score": Decimal("0.0587"), "risk_tier": "low",
         "scored_at": "2026-09-25T07:40:32+00:00", "model_artifact_path": "s3://bucket/model.tar.gz"},
        {"person_id": "KE100001", "risk_score": Decimal("0.4500"), "risk_tier": "medium",
         "scored_at": "2026-09-25T07:40:32+00:00", "model_artifact_path": "s3://bucket/model.tar.gz"},
    ]
    for item in sample_items:
        table.put_item(Item=item)

    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)

    print("=== GET /health ===")
    r = client.get("/health")
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("\n=== GET /score/KE114061 (exists) ===")
    r = client.get("/score/KE114061")
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json()["risk_tier"] == "high"

    print("\n=== GET /score/DOES_NOT_EXIST (404 case) ===")
    r = client.get("/score/DOES_NOT_EXIST")
    print(r.status_code, r.json())
    assert r.status_code == 404

    print("\n=== GET /scores (no filter) ===")
    r = client.get("/scores")
    print(r.status_code, "count:", r.json()["count"])
    assert r.status_code == 200
    assert r.json()["count"] == 4

    print("\n=== GET /scores?risk_tier=high ===")
    r = client.get("/scores", params={"risk_tier": "high"})
    print(r.status_code, "count:", r.json()["count"], "person_ids:", [i["person_id"] for i in r.json()["items"]])
    assert r.status_code == 200
    assert r.json()["count"] == 2

    print("\nAll local tests passed.")