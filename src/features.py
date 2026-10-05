"""Shared feature definitions (used by training and the API)."""
import numpy as np
import pandas as pd

NUMERIC = ["amount", "amount_ratio", "distance_from_home_km", "txn_count_1h", "is_new_merchant",
           "device_change", "foreign_txn", "hour", "is_night", "log_amount"]
CATEGORICAL = ["category", "channel"]
# Protected / segment attributes: NOT used as model inputs, only for fairness monitoring.
SEGMENTS = ["age_band", "region"]
TARGET = "is_fraud"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    out["log_amount"] = np.log1p(out["amount"])
    return out


def model_columns():
    return NUMERIC + CATEGORICAL
