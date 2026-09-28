import re
from pathlib import Path
import processing_script as ps
import train
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "deploy_assets/raw/kenya_hpv_screening/dt=2026-09-19/kenya_hpv_screening_synthetic.csv"
SCORING_CSV = ROOT / "deploy_assets/scoring-input/test_for_scoring.csv"
GLUE = ROOT / "glue" / "glue_etl_job.py"

def _available_columns():
    raw = RAW_CSV.open().readline().strip().split(",")
    engineered = set(re.findall(r'withColumn\(\s*"(\w+)"', GLUE.read_text()))
    return set(raw) | engineered

def test_training_features_are_produced_by_the_etl():
    missing = set(train.FEATURE_COLS) - _available_columns()
    assert not missing, f"train.py expects columns the Glue job never produces: {missing}"

def test_processing_expected_columns_are_produced_by_the_etl():
    missing = set(ps.EXPECTED_COLUMNS) - _available_columns()
    assert not missing, f"processing_script expects columns the Glue job never produces: {missing}"

def test_scoring_input_matches_the_model_contract():
    width = 1 + len(train.FEATURE_COLS)
    with SCORING_CSV.open() as f:
        first = f.readline().strip().split(",")
        assert first[0] != "person_id", "scoring file has a header row"
        f.seek(0)
        for i, line in enumerate(f, 1):
            fields = line.strip().split(",")
            assert len(fields) == width, f"line {i}: {len(fields)} fields, expected {width}"
            [float(x) for x in fields[1:]]
