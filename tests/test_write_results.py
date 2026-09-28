import importlib
import boto3
import pytest
from moto import mock_aws
BUCKET = "test-bucket"
KEY = "batch-transform-output/test_for_scoring.csv.out"
TABLE = "hpv-risk-scores-test"

@pytest.fixture
def wr(monkeypatch):
    monkeypatch.setenv("TABLE_NAME", TABLE)
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "person_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "person_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        import write_results
        importlib.reload(write_results)
        yield write_results, table, s3

@pytest.mark.parametrize(
    "score,tier",
    [(0.0, "low"), (0.2999, "low"), (0.3, "medium"), (0.5999, "medium"), (0.6, "high"), (1.0, "high")],
)
def test_risk_tier_thresholds(wr, score, tier):
    assert wr[0].compute_risk_tier(score) == tier

def test_handler_writes_scores_and_tiers(wr):
    mod, table, s3 = wr
    body = "KE1,22,1,0,0,0,0,0,0,0,0.86\nKE2,17,2,0,0,0,0,0,0,1,0.45\n\nKE3,15,3,1,0,0,0,0,0,1,0.05\n\n"
    s3.put_object(Bucket=BUCKET, Key=KEY, Body=body)
    result = mod.handler({"bucket": BUCKET, "key": KEY, "model_artifact_path": "s3://m/model.tar.gz"}, None)
    assert result["records_written"] == 3
    assert result["tier_counts"] == {"high": 1, "medium": 1, "low": 1}
    item = table.get_item(Key={"person_id": "KE2"})["Item"]
    assert item["risk_tier"] == "medium"
    assert float(item["risk_score"]) == 0.45
    assert item["model_artifact_path"] == "s3://m/model.tar.gz"
