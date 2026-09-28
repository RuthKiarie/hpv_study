"""
Streamlit dashboard for the HPV risk-scoring pipeline.

Connects DIRECTLY to DynamoDB via boto3 (see dashboard_data.py) rather than
through the public API -- see that file's docstring for why.

Run locally with:  streamlit run dashboard.py
Requires AWS credentials configured (same as every other script here).
"""

import pandas as pd
import streamlit as st

from dashboard_data import get_score, list_scores, get_tier_counts

st.set_page_config(page_title="HPV Screening Risk Dashboard", layout="wide")

st.title("HPV Screening Risk Dashboard")
st.caption(
    "Cervical-cancer-screening risk scores from the synthetic Kenya pipeline. "
    "Connects directly to DynamoDB with the operator's AWS credentials."
)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
with st.spinner("Loading tier summary..."):
    tier_counts = get_tier_counts()

total = sum(tier_counts.values())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Scored", total)
col2.metric("High Risk", tier_counts["high"], help="risk_score \u2265 0.6")
col3.metric("Medium Risk", tier_counts["medium"], help="0.3 \u2264 risk_score < 0.6")
col4.metric("Low Risk", tier_counts["low"], help="risk_score < 0.3")

if total > 0:
    chart_df = pd.DataFrame({
        "Tier": ["High", "Medium", "Low"],
        "Count": [tier_counts["high"], tier_counts["medium"], tier_counts["low"]],
    })
    st.bar_chart(chart_df.set_index("Tier"))

st.divider()

# ---------------------------------------------------------------------------
# Single lookup
# ---------------------------------------------------------------------------
st.subheader("Look up a single person")
person_id = st.text_input("person_id", placeholder="e.g. KE112491")
if person_id:
    result = get_score(person_id.strip())
    if result is None:
        st.warning(f"No score found for person_id={person_id}")
    else:
        st.json(result)

st.divider()

# ---------------------------------------------------------------------------
# Filtered list
# ---------------------------------------------------------------------------
st.subheader("Browse scores")
filter_col, limit_col = st.columns([1, 1])
with filter_col:
    tier_filter = st.selectbox("Filter by risk tier", options=["All", "high", "medium", "low"])
with limit_col:
    limit = st.slider("Max rows", min_value=10, max_value=500, value=50, step=10)

st.caption(
    "Note: DynamoDB's row limit applies to rows *scanned*, not rows *returned after filtering* -- "
    "so a tier filter can return fewer rows than the slider value even when more matches exist. "
    "Fine at this dataset's scale; a GSI on risk_tier would be the production-scale fix."
)

with st.spinner("Querying DynamoDB..."):
    rows = list_scores(risk_tier=None if tier_filter == "All" else tier_filter, limit=limit)

if rows:
    df = pd.DataFrame(rows).sort_values("risk_score", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No matching records.")