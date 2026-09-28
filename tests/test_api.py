import importlib
from decimal import Decimal
import boto3
import pytest
from moto import mock_aws
TABLE = "hpv-risk-scores-test"
ITEMS = [
    ("KE1", "0.86", "high"),
    ("KE2", "0.68", "high"),
    ("KE3", "0.45", "medium"),
    ("KE4", "0.05", "low"),
]

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TABLE_NAME", TABLE)
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "person_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "person_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        for pid, score, tier in ITEMS:
            table.put_item(
                Item={
                    "person_id": pid,
                    "risk_score": Decimal(score),
                    "risk_tier": tier,
                    "scored_at": "2026-01-01T00:00:00+00:00",
                    "model_artifact_path": "s3://m/model.tar.gz",
                }
            )
        import app as api_app
        importlib.reload(api_app)
        from fastapi.testclient import TestClient
        yield TestClient(api_app.app)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}

def test_get_score_found(client):
    r = client.get("/score/KE1")
    assert r.status_code == 200
    body = r.json()
    assert body["risk_tier"] == "high" and body["risk_score"] == 0.86

def test_get_score_missing_is_404(client):
    assert client.get("/score/NOPE").status_code == 404

def test_list_all(client):
    assert client.get("/scores").json()["count"] == 4

def test_list_filtered_by_tier(client):
    body = client.get("/scores", params={"risk_tier": "high"}).json()
    assert body["count"] == 2
    assert {i["person_id"] for i in body["items"]} == {"KE1", "KE2"}

@pytest.mark.parametrize("limit", [0, 501])
def test_limit_bounds_are_validated(client, limit):
    assert client.get("/scores", params={"limit": limit}).status_code == 422
