import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from generate_data import generate
from train import net_saving, time_split
from features import add_features, NUMERIC, CATEGORICAL, SEGMENTS


def small():
    return add_features(generate(n_txn=20_000, n_customers=1_000, seed=1))


def test_time_split_has_no_overlap_and_is_chronological():
    tr, va, te = time_split(small())
    assert tr.timestamp.max() < va.timestamp.min() <= va.timestamp.max() < te.timestamp.min()


def test_protected_attributes_not_model_inputs():
    assert not set(SEGMENTS) & set(NUMERIC + CATEGORICAL)


def test_fraud_is_rare_but_present():
    r = small().is_fraud.mean()
    assert 0.002 < r < 0.05


def test_net_saving_logic():
    y = pd.Series([1, 0, 1, 0]).values
    amt = pd.Series([100.0, 50.0, 200.0, 10.0]).values
    flag = pd.Series([1, 1, 0, 0]).values
    assert net_saving(y, amt, flag) == 100.0 - 2 * 5.0


def test_api_root_redirects_to_docs():
    from fastapi.testclient import TestClient
    import api
    response = TestClient(api.app).get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_api_predict():
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    assert c.get("/health").status_code == 200
    body = dict(amount=900, amount_ratio=8, hour=2, category="electronics", channel="online",
                distance_from_home_km=250, txn_count_1h=4, is_new_merchant=1, device_change=1, foreign_txn=1)
    r = c.post("/predict", json=body).json()
    assert r["decision"] in ("REVIEW", "APPROVE") and len(r["top_reasons"]) == 3
    safe = dict(body, amount=12, amount_ratio=0.5, hour=14, channel="pos", category="grocery",
                distance_from_home_km=2, txn_count_1h=0, is_new_merchant=0, device_change=0, foreign_txn=0)
    assert c.post("/predict", json=safe).json()["fraud_score"] < r["fraud_score"]
