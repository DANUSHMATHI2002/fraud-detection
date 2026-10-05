"""FastAPI scoring service with per-transaction explanations.  Run: uvicorn api:app --app-dir src"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from features import add_features

ART = joblib.load(Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib")
MODEL, THRESHOLD, COLS = ART["model"], ART["threshold"], ART["columns"]
PREP, CLF = MODEL.named_steps["prep"], MODEL.named_steps["clf"]
NAMES = [n.split("__", 1)[1] for n in PREP.get_feature_names_out()]
EXPLAINER = shap.TreeExplainer(CLF) if ART["model_name"] != "Logistic Regression" else None

app = FastAPI(title="Fraud Detection API", version="1.0")


class Txn(BaseModel):
    amount: float = Field(gt=0)
    amount_ratio: float = Field(ge=0, description="amount / customer's average spend")
    hour: int = Field(ge=0, le=23)
    category: str
    channel: str
    distance_from_home_km: float = Field(ge=0)
    txn_count_1h: int = Field(ge=0)
    is_new_merchant: int = Field(ge=0, le=1)
    device_change: int = Field(ge=0, le=1)
    foreign_txn: int = Field(ge=0, le=1)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": ART["model_name"], "threshold": THRESHOLD}


@app.post("/predict")
def predict(txn: Txn):
    X = add_features(pd.DataFrame([txn.model_dump()]))[COLS]
    score = float(MODEL.predict_proba(X)[0, 1])
    reasons = []
    if EXPLAINER is not None:
        Xt = PREP.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, 'toarray') else Xt
        sv = EXPLAINER.shap_values(Xt)[0]
        idx = np.argsort(-np.abs(sv))[:3]
        reasons = [{"feature": NAMES[i], "impact": round(float(sv[i]), 3),
                    "direction": "raises risk" if sv[i] > 0 else "lowers risk"} for i in idx]
    return {"fraud_score": round(score, 4), "flagged": score >= THRESHOLD,
            "decision": "REVIEW" if score >= THRESHOLD else "APPROVE", "top_reasons": reasons}
