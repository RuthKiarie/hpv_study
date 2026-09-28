"""
Lambda function: reads the Batch Transform output CSV from S3, computes
risk_tier from risk_score using the agreed thresholds, and writes each
record to the hpv-risk-scores DynamoDB table.

Expected input event (passed from Step Functions):
{
  "bucket": "rwk-hpv-mlops",
  "key": "batch-transform-output/test_for_scoring.csv.out",
  "model_artifact_path": "s3://.../model.tar.gz"   (for traceability)
}

Expected S3 file format: no header, each row is
  person_id, <10 feature columns>, risk_score
(this is exactly what join_source="Input" produces -- the original input
row with risk_score appended as the last column).
"""

import csv
import io
from datetime import datetime, timezone
from decimal import Decimal
import os

import boto3

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

TABLE_NAME = os.environ.get("TABLE_NAME", "hpv-risk-scores")


def compute_risk_tier(risk_score: float) -> str:
    if risk_score >= 0.6:
        return "high"
    elif risk_score >= 0.3:
        return "medium"
    else:
        return "low"


def handler(event, context):
    bucket = event["bucket"]
    key = event["key"]
    model_artifact_path = event.get("model_artifact_path", "unknown")

    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read().decode("utf-8")
    reader = csv.reader(io.StringIO(content))

    scored_at = datetime.now(timezone.utc).isoformat()
    table = dynamodb.Table(TABLE_NAME)

    written = 0
    tier_counts = {"high": 0, "medium": 0, "low": 0}

    with table.batch_writer() as batch:
        for row in reader:
            if not row:
                continue
            person_id = row[0]
            risk_score = float(row[-1])  # last column = joined prediction
            risk_tier = compute_risk_tier(risk_score)
            tier_counts[risk_tier] += 1

            batch.put_item(Item={
                "person_id": person_id,
                "risk_score": Decimal(str(round(risk_score, 4))),
                "risk_tier": risk_tier,
                "scored_at": scored_at,
                "model_artifact_path": model_artifact_path,
            })
            written += 1

    result = {"records_written": written, "scored_at": scored_at, "tier_counts": tier_counts}
    print("Write summary:", result)
    return result


# ---------------------------------------------------------------------------
# Local test harness -- NOT run in Lambda, only when this file is executed
# directly, so we can validate the parsing/tier logic before deploying.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open("sample_output.csv") as f:
        content = f.read()
    reader = csv.reader(io.StringIO(content))
    for row in reader:
        person_id = row[0]
        risk_score = float(row[-1])
        risk_tier = compute_risk_tier(risk_score)
        print(f"{person_id}: risk_score={risk_score:.4f} -> risk_tier={risk_tier}")