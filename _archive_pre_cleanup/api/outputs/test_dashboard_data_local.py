import os
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

from decimal import Decimal
from moto import mock_aws
import boto3

with mock_aws():
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    table = ddb.create_table(
        TableName="hpv-risk-scores",
        KeySchema=[{"AttributeName": "person_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "person_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    sample_items = [
        {"person_id": "KE1", "risk_score": Decimal("0.86"), "risk_tier": "high", "scored_at": "t", "model_artifact_path": "p"},
        {"person_id": "KE2", "risk_score": Decimal("0.68"), "risk_tier": "high", "scored_at": "t", "model_artifact_path": "p"},
        {"person_id": "KE3", "risk_score": Decimal("0.45"), "risk_tier": "medium", "scored_at": "t", "model_artifact_path": "p"},
        {"person_id": "KE4", "risk_score": Decimal("0.05"), "risk_tier": "low", "scored_at": "t", "model_artifact_path": "p"},
    ]
    for item in sample_items:
        table.put_item(Item=item)

    from api.dashboard_data import get_score, list_scores, get_tier_counts

    print("=== get_score('KE1') ===")
    r = get_score("KE1")
    print(r)
    assert r["risk_tier"] == "high"

    print("\n=== get_score('MISSING') ===")
    r = get_score("MISSING")
    print(r)
    assert r is None

    print("\n=== list_scores() no filter ===")
    r = list_scores()
    print(f"{len(r)} items")
    assert len(r) == 4

    print("\n=== list_scores(risk_tier='high') ===")
    r = list_scores(risk_tier="high")
    print(f"{len(r)} items:", [i["person_id"] for i in r])
    assert len(r) == 2

    print("\n=== get_tier_counts() ===")
    r = get_tier_counts()
    print(r)
    assert r == {"high": 2, "medium": 1, "low": 1}

    print("\nAll dashboard data-layer tests passed.")