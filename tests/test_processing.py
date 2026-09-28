import pandas as pd
import processing_script as ps

def _df(n=10):
    return pd.DataFrame({c: [0] * n for c in ps.EXPECTED_COLUMNS})

def test_validate_passes_on_complete_data():
    v = ps.validate(_df())
    assert v["passed"] and v["row_count"] == 10 and v["missing_columns"] == []

def test_validate_flags_missing_column():
    v = ps.validate(_df().drop(columns=["age"]))
    assert not v["passed"] and v["missing_columns"] == ["age"]

def test_validate_rejects_empty_data():
    assert not ps.validate(_df(0))["passed"]

def test_drift_stats():
    df = _df(4)
    df["screened_last_3yrs"] = [1, 0, 0, 0]
    df["high_risk_flag"] = [0, 1, 0, 0]
    stats = ps.compute_drift_stats(df)
    assert stats["screened_last_3yrs_prevalence"] == 0.25
    assert stats["high_risk_flag_count"] == 1
