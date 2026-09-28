"""
Read-side API for the HPV risk-scoring pipeline. Runs as a Lambda function
behind API Gateway (via Mangum), but is a completely normal FastAPI app --
can also be run locally with `uvicorn app:app --reload` for testing.

Endpoints:
  GET /health                        - basic health check
  GET /score/{person_id}             - single lookup, 404 if not found
  GET /scores?risk_tier=high&limit=50 - filtered list (table Scan -- see
                                         README note on GSI as a future
                                         improvement at real scale)
"""

from decimal import Decimal
from typing import List, Optional

import boto3
from boto3.dynamodb.conditions import Attr
from fastapi import FastAPI, HTTPException, Query
from mangum import Mangum
from pydantic import BaseModel

TABLE_NAME = "hpv-risk-scores"
REGION = "us-east-1"

app = FastAPI(title="HPV Risk Scoring API")

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)


class ScoreResponse(BaseModel):
    person_id: str
    risk_score: float
    risk_tier: str
    scored_at: str
    model_artifact_path: str


class ScoreListResponse(BaseModel):
    count: int
    items: List[ScoreResponse]


def _to_score_response(item: dict) -> ScoreResponse:
    """DynamoDB returns Decimal for numbers -- convert before Pydantic sees it."""
    return ScoreResponse(
        person_id=item["person_id"],
        risk_score=float(item["risk_score"]) if isinstance(item["risk_score"], Decimal) else item["risk_score"],
        risk_tier=item["risk_tier"],
        scored_at=item["scored_at"],
        model_artifact_path=item["model_artifact_path"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/score/{person_id}", response_model=ScoreResponse)
def get_score(person_id: str):
    result = table.get_item(Key={"person_id": person_id})
    item = result.get("Item")
    if item is None:
        raise HTTPException(status_code=404, detail=f"No score found for person_id={person_id}")
    return _to_score_response(item)


@app.get("/scores", response_model=ScoreListResponse)
def list_scores(
    risk_tier: Optional[str] = Query(default=None, description="Filter by 'high', 'medium', or 'low'"),
    limit: int = Query(default=50, ge=1, le=500),
):
    # NOTE: DynamoDB's Limit caps items SCANNED, not items returned after
    # FilterExpression is applied -- so with a risk_tier filter, the actual
    # returned count can be less than `limit` even if more matches exist.
    # Fine at our current scale (~3.4k rows); a GSI on risk_tier would be
    # the right fix at real production scale.
    scan_kwargs = {"Limit": limit}
    if risk_tier is not None:
        scan_kwargs["FilterExpression"] = Attr("risk_tier").eq(risk_tier)

    result = table.scan(**scan_kwargs)
    items = [_to_score_response(i) for i in result.get("Items", [])]
    return ScoreListResponse(count=len(items), items=items)


handler = Mangum(app)