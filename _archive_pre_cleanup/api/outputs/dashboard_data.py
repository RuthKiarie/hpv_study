"""
Data-access layer for the Streamlit dashboard -- direct boto3 calls to
DynamoDB, deliberately NOT going through the Lambda/API Gateway layer.

Design note: this dashboard is intended to run with the my own AWS
credentials already configured (same as every other script in this
project), so it queries the data store directly rather than through a
public HTTP API. If/when the public API access issue is resolved, this
could be swapped for HTTP calls -- but direct access is the right call for
project, not a workaround.
"""

from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Attr

TABLE_NAME = "hpv-risk-scores"
REGION = "us-east-1"


def _get_table():
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return dynamodb.Table(TABLE_NAME)


def _clean(item: dict) -> dict:
    return {
        "person_id": item["person_id"],
        "risk_score": float(item["risk_score"]) if isinstance(item["risk_score"], Decimal) else item["risk_score"],
        "risk_tier": item["risk_tier"],
        "scored_at": item["scored_at"],
        "model_artifact_path": item["model_artifact_path"],
    }


def get_score(person_id: str) -> dict | None:
    table = _get_table()
    result = table.get_item(Key={"person_id": person_id})
    item = result.get("Item")
    return _clean(item) if item else None


def list_scores(risk_tier: str | None = None, limit: int = 200) -> list[dict]:
    table = _get_table()
    scan_kwargs = {"Limit": limit}
    if risk_tier:
        scan_kwargs["FilterExpression"] = Attr("risk_tier").eq(risk_tier)
    result = table.scan(**scan_kwargs)
    return [_clean(i) for i in result.get("Items", [])]


def get_tier_counts() -> dict:
    """Full-table scan to compute tier distribution for the summary chart.
    Fine at our ~3.4k row scale; would need a smarter approach (e.g. a
    maintained counter, or a GSI + query) at real production scale."""
    table = _get_table()
    counts = {"high": 0, "medium": 0, "low": 0}
    response = table.scan(ProjectionExpression="risk_tier")
    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = table.scan(ProjectionExpression="risk_tier", ExclusiveStartKey=response["LastEvaluatedKey"])
        items += response.get("Items", [])
    for item in items:
        tier = item.get("risk_tier")
        if tier in counts:
            counts[tier] += 1
    return counts