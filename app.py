"""
HPV Screening Risk Dashboard (Kenya) - standalone version for Streamlit Community Cloud.

Mirrors the logic of the AWS pipeline in the hpv_study repo (same features, same
model settings, same stratified split, same risk tiers) but reads the bundled
synthetic CSV instead of DynamoDB, so it needs no AWS credentials.

All data is SYNTHETIC. It is not real patient or government data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).parent
_CSV = "kenya_hpv_screening_synthetic.csv"
# Works whether app.py sits at the repo root, in dashboard/, or next to the CSV.
DATA_PATH = next(
    (p for p in [_HERE / _CSV, _HERE / "data" / _CSV, _HERE.parent / "data" / _CSV] if p.exists()),
    _HERE / _CSV,
)

FEATURE_COLS = [
    "age", "education_ord", "wealth_ord", "is_urban", "health_insurance",
    "hiv_positive", "hiv_unknown", "distance_barrier",
    "heard_of_cervical_cancer", "parity_capped",
]
TARGET_COL = "screened_last_3yrs"
EDU_MAP = {"none": 0, "primary": 1, "secondary": 2, "higher": 3}
WEALTH_MAP = {"poorest": 0, "poorer": 1, "middle": 2, "richer": 3, "richest": 4}
WEALTH_ORDER = list(WEALTH_MAP)
TIER_ORDER = ["low", "medium", "high"]
TIER_COLORS = {"low": "#2E7D32", "medium": "#F9A825", "high": "#C62828"}

st.set_page_config(page_title="HPV Screening Risk - Kenya", page_icon="🩺", layout="wide")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same encodings as glue/glue_etl_job.py."""
    df = df.copy()
    df["education_ord"] = df["education"].map(EDU_MAP)
    df["wealth_ord"] = df["wealth_quintile"].map(WEALTH_MAP)
    df["is_urban"] = (df["residence"] == "urban").astype(int)
    df["hiv_positive"] = (df["hiv_status"] == "positive").astype(int)
    df["hiv_unknown"] = (df["hiv_status"] == "unknown").astype(int)
    df["distance_barrier"] = (df["distance_problem"] == "big_problem").astype(int)
    df["age_group"] = pd.cut(
        df["age"], bins=[-1, 19, 29, 39, 200], labels=["15-19", "20-29", "30-39", "40-49"]
    ).astype(str)
    df["parity_capped"] = df["parity"].clip(upper=6)
    return df


def risk_tier(score: float) -> str:
    """Same thresholds as lambda/write_results/write_results.py."""
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


@st.cache_resource(show_spinner="Training model...")
def load_and_train():
    df = add_features(pd.read_csv(DATA_PATH))

    # Same split and model settings as processing_script.py and train.py.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_COL]
    )
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    proba = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    pred = model.predict(test_df[FEATURE_COLS])
    metrics = {
        "Accuracy": accuracy_score(test_df[TARGET_COL], pred),
        "Precision": precision_score(test_df[TARGET_COL], pred),
        "Recall": recall_score(test_df[TARGET_COL], pred),
        "AUC-ROC": roc_auc_score(test_df[TARGET_COL], proba),
    }
    cm = confusion_matrix(test_df[TARGET_COL], pred)
    coefs = pd.Series(model.coef_[0], index=FEATURE_COLS).sort_values()

    # risk_score = probability of NOT being screened (as in train.py predict_fn)
    df["risk_score"] = 1 - model.predict_proba(df[FEATURE_COLS])[:, 1]
    df["risk_tier"] = df["risk_score"].apply(risk_tier)
    return df, model, metrics, cm, coefs, len(train_df), len(test_df)


df, model, metrics, cm, coefs, n_train, n_test = load_and_train()

# ----------------------------------------------------------------- header
st.title("🩺 HPV Screening Risk Dashboard - Kenya")
st.caption(
    "Predicts which women are least likely to have been screened for cervical cancer "
    "in the last 3 years, so outreach can reach them first."
)
st.warning(
    "**All data here is synthetic**, modelled on Kenyan demographic and health patterns. "
    "It is not real patient data, and nothing on this page is a finding about real women.",
    icon="⚠️",
)

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.header("Filters")
    counties = st.multiselect("County", sorted(df["county"].unique()))
    residence = st.multiselect("Residence", sorted(df["residence"].unique()))
    age_groups = st.multiselect("Age group", ["15-19", "20-29", "30-39", "40-49"])
    hiv = st.multiselect("HIV status", ["negative", "positive", "unknown"])
    wealth = st.multiselect("Wealth quintile", WEALTH_ORDER)
    st.caption("Leave a filter empty to include everyone.")

view = df
for col, chosen in [
    ("county", counties), ("residence", residence), ("age_group", age_groups),
    ("hiv_status", hiv), ("wealth_quintile", wealth),
]:
    if chosen:
        view = view[view[col].isin(chosen)]

if view.empty:
    st.info("No records match these filters.")
    st.stop()

tab_overview, tab_who, tab_model, tab_score = st.tabs(
    ["Overview", "Who is being missed", "Model", "Score a profile"]
)

# --------------------------------------------------------------- overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Women in view", f"{len(view):,}")
    c2.metric("Screened in last 3 years", f"{view[TARGET_COL].mean():.1%}")
    c3.metric("High-risk tier", f"{(view['risk_tier'] == 'high').mean():.1%}")
    c4.metric("Mean risk score", f"{view['risk_score'].mean():.2f}")

    left, right = st.columns(2)
    tier_counts = (
        view["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0).reset_index()
    )
    tier_counts.columns = ["risk_tier", "women"]
    fig = px.bar(
        tier_counts, x="risk_tier", y="women", color="risk_tier",
        color_discrete_map=TIER_COLORS, title="Women by risk tier",
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    left.plotly_chart(fig, width="stretch")

    fig = px.histogram(
        view, x="risk_score", nbins=40, title="Distribution of risk scores",
        color_discrete_sequence=["#1F4E79"],
    )
    fig.add_vline(x=0.3, line_dash="dash", annotation_text="medium")
    fig.add_vline(x=0.6, line_dash="dash", annotation_text="high")
    right.plotly_chart(fig, width="stretch")

    st.caption(
        "Risk score = probability of NOT having been screened. "
        "Tiers: low below 0.30, medium 0.30 to 0.59, high 0.60 and above."
    )

# ----------------------------------------------------------- who is missed
with tab_who:
    st.subheader("Screening rate by group")
    st.caption("Lower bars show groups where fewer women have been screened.")

    dim_labels = {
        "County": "county",
        "Age group": "age_group",
        "Residence": "residence",
        "HIV status": "hiv_status",
        "Wealth quintile": "wealth_quintile",
        "Education": "education",
        "Distance to facility": "distance_problem",
        "Health insurance": "health_insurance",
        "Heard of cervical cancer": "heard_of_cervical_cancer",
    }
    choice = st.selectbox("Group by", list(dim_labels))
    col = dim_labels[choice]
    grouped = (
        view.groupby(col)
        .agg(women=(TARGET_COL, "size"), screened_rate=(TARGET_COL, "mean"),
             mean_risk=("risk_score", "mean"))
        .reset_index()
    )
    if col == "wealth_quintile":
        grouped[col] = pd.Categorical(grouped[col], WEALTH_ORDER, ordered=True)
    grouped = grouped.sort_values("screened_rate" if col == "county" else col)
    grouped[col] = grouped[col].astype(str)

    fig = px.bar(
        grouped, x=col, y="screened_rate", hover_data=["women", "mean_risk"],
        color_discrete_sequence=["#1F4E79"],
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title="Screened rate")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Highest-risk women in view")
    high = view[view["risk_tier"] == "high"].sort_values("risk_score", ascending=False)
    show_cols = ["person_id", "age", "county", "residence", "hiv_status",
                 "wealth_quintile", "risk_score", "risk_tier"]
    st.dataframe(
        high[show_cols].head(200).style.format({"risk_score": "{:.3f}"}),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "Download all high-risk records (CSV)",
        high[show_cols].to_csv(index=False).encode(),
        file_name="high_risk_women_synthetic.csv", mime="text/csv",
    )

# ------------------------------------------------------------------ model
with tab_model:
    st.subheader("Logistic regression")
    st.caption(
        f"Trained on {n_train:,} records, evaluated on {n_test:,} held-out records "
        "(stratified 80/20 split, class-weight balanced), same settings as the AWS pipeline."
    )
    m1, m2, m3, m4 = st.columns(4)
    for box, (name, val) in zip([m1, m2, m3, m4], metrics.items()):
        box.metric(name, f"{val:.3f}")

    left, right = st.columns(2)
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted not screened", "Predicted screened"],
        y=["Actually not screened", "Actually screened"],
        title="Confusion matrix (test set)",
    )
    cm_fig.update_coloraxes(showscale=False)
    left.plotly_chart(cm_fig, width="stretch")

    coef_df = coefs.reset_index()
    coef_df.columns = ["feature", "coefficient"]
    coef_fig = px.bar(
        coef_df, x="coefficient", y="feature", orientation="h",
        title="Feature coefficients (log-odds of being screened)",
        color_discrete_sequence=["#1F4E79"],
    )
    right.plotly_chart(coef_fig, width="stretch")

    st.info(
        "**Label leakage:** the pipeline also builds a `high_risk_flag` "
        "(HIV-positive AND not screened). It is derived from the answer, so it is "
        "deliberately excluded from the model's features.",
        icon="🔒",
    )

# ------------------------------------------------------------ score a profile
with tab_score:
    st.subheader("Score a hypothetical profile")
    st.caption("Change the inputs to see how each factor moves the risk of not being screened.")
    a, b, c = st.columns(3)
    age = a.slider("Age", 15, 49, 30)
    education = a.selectbox("Education", list(EDU_MAP), index=2)
    wealth_q = a.selectbox("Wealth quintile", WEALTH_ORDER, index=2)
    resid = b.selectbox("Residence", ["urban", "rural"])
    hiv_status = b.selectbox("HIV status", ["negative", "positive", "unknown"])
    parity = b.slider("Number of births", 0, 8, 1)
    insured = c.checkbox("Has health insurance")
    far = c.checkbox("Distance to facility is a big problem")
    heard = c.checkbox("Has heard of cervical cancer", value=True)

    row = pd.DataFrame([{
        "age": age,
        "education_ord": EDU_MAP[education],
        "wealth_ord": WEALTH_MAP[wealth_q],
        "is_urban": int(resid == "urban"),
        "health_insurance": int(insured),
        "hiv_positive": int(hiv_status == "positive"),
        "hiv_unknown": int(hiv_status == "unknown"),
        "distance_barrier": int(far),
        "heard_of_cervical_cancer": int(heard),
        "parity_capped": min(parity, 6),
    }])[FEATURE_COLS]
    score = float(1 - model.predict_proba(row)[0, 1])
    tier = risk_tier(score)
    st.metric("Risk of not being screened", f"{score:.0%}")
    st.markdown(
        f"Risk tier: <span style='color:{TIER_COLORS[tier]};font-weight:700'>{tier.upper()}</span>",
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Built by Ruth Kiarie · Synthetic data only · "
    "Pipeline code: github.com/RuthKiarie/hpv_study"
)
