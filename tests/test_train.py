import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import train

def test_no_label_leakage_in_features():
    assert "high_risk_flag" not in train.FEATURE_COLS
    assert train.TARGET_COL not in train.FEATURE_COLS
    assert "person_id" not in train.FEATURE_COLS
    assert len(set(train.FEATURE_COLS)) == len(train.FEATURE_COLS)

def _toy_model():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]] * 5)
    y = np.array([0, 1, 1, 0] * 5)
    return X, LogisticRegression().fit(X, y)

def test_predict_fn_returns_risk_of_not_screened():
    X, model = _toy_model()
    risk = train.predict_fn(X, model)
    assert np.allclose(risk, 1 - model.predict_proba(X)[:, 1])
    assert ((risk >= 0) & (risk <= 1)).all()

def test_model_fn_roundtrip(tmp_path):
    X, model = _toy_model()
    joblib.dump(model, tmp_path / "model.joblib")
    loaded = train.model_fn(str(tmp_path))
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
