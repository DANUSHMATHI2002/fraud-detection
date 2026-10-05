"""Train, evaluate and explain fraud models; export results for Tableau.

Key design choices (see README):
  * time-based split (no random shuffling -> no look-ahead leakage)
  * PR-AUC as headline metric (fraud is ~1% of transactions)
  * decision threshold chosen on VALIDATION data by business cost, then frozen for TEST
  * protected attributes excluded from the model; recall/FPR monitored by segment
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CATEGORICAL, NUMERIC, SEGMENTS, TARGET, add_features, model_columns

ROOT = Path(__file__).resolve().parents[1]
OUT, TAB, MODELS = ROOT / "outputs", ROOT / "tableau", ROOT / "models"
REVIEW_COST = 5.0      # GBP cost of an analyst reviewing / customer friction per flagged transaction
SEED = 42


def time_split(df):
    t = df["timestamp"]
    d0 = t.min().normalize()
    train = df[t < d0 + pd.Timedelta(days=60)]
    val = df[(t >= d0 + pd.Timedelta(days=60)) & (t < d0 + pd.Timedelta(days=75))]
    test = df[t >= d0 + pd.Timedelta(days=75)]
    return train, val, test


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def build_models(pos_weight):
    return {
        "Logistic Regression": Pipeline([("prep", preprocessor()),
                                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("prep", preprocessor()),
                                   ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1,
                                                                  class_weight="balanced_subsample", random_state=SEED))]),
        "XGBoost": Pipeline([("prep", preprocessor()),
                             ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8,
                                                   colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                                   eval_metric="aucpr", n_jobs=-1, random_state=SEED))]),
    }


def net_saving(y, amount, flag):
    """Fraud value caught minus fraud value missed minus review cost of every flag (vs. doing nothing)."""
    caught = amount[(flag == 1) & (y == 1)].sum()
    cost = REVIEW_COST * flag.sum()
    return float(caught - cost)


def best_threshold(y, amount, score):
    grid = np.linspace(0.02, 0.98, 97)
    rows = [{"threshold": float(t), "net_saving": net_saving(y, amount, (score >= t).astype(int)),
             "flagged": int((score >= t).sum()),
             "recall": float(((score >= t) & (y == 1)).sum() / max(y.sum(), 1)),
             "precision": float(((score >= t) & (y == 1)).sum() / max((score >= t).sum(), 1))} for t in grid]
    curve = pd.DataFrame(rows)
    return float(curve.loc[curve["net_saving"].idxmax(), "threshold"]), curve


def main():
    for d in (OUT, TAB, MODELS):
        d.mkdir(exist_ok=True)
    df = add_features(pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["timestamp"]))
    train, val, test = time_split(df)
    cols = model_columns()
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | fraud rate "
          f"{train[TARGET].mean():.2%}/{val[TARGET].mean():.2%}/{test[TARGET].mean():.2%}")

    pos_weight = (train[TARGET] == 0).sum() / (train[TARGET] == 1).sum()
    results, fitted = [], {}
    for name, pipe in build_models(pos_weight).items():
        pipe.fit(train[cols], train[TARGET])
        fitted[name] = pipe
        for split_name, part in (("validation", val), ("test", test)):
            s = pipe.predict_proba(part[cols])[:, 1]
            results.append({"model": name, "split": split_name,
                            "roc_auc": roc_auc_score(part[TARGET], s),
                            "pr_auc": average_precision_score(part[TARGET], s)})
    res = pd.DataFrame(results)
    res.to_csv(TAB / "model_comparison.csv", index=False)
    print(res.round(4).to_string(index=False))

    best_name = res[res.split == "validation"].sort_values("pr_auc", ascending=False).iloc[0]["model"]
    model = fitted[best_name]
    print("selected on validation PR-AUC:", best_name)

    # Threshold tuned on validation only, then applied unchanged to the untouched test period
    v_score = model.predict_proba(val[cols])[:, 1]
    thr, _ = best_threshold(val[TARGET].values, val["amount"].values, v_score)
    t_score = model.predict_proba(test[cols])[:, 1]
    _, test_curve = best_threshold(test[TARGET].values, test["amount"].values, t_score)
    test_curve.to_csv(TAB / "threshold_curve_test.csv", index=False)

    test = test.copy()
    test["fraud_score"] = t_score
    test["flagged"] = (t_score >= thr).astype(int)
    test["outcome"] = np.select(
        [(test.flagged == 1) & (test[TARGET] == 1), (test.flagged == 1) & (test[TARGET] == 0),
         (test.flagged == 0) & (test[TARGET] == 1)], ["True Positive", "False Positive", "False Negative"], "True Negative")
    y, f, amt = test[TARGET].values, test["flagged"].values, test["amount"].values
    summary = {
        "selected_model": best_name, "threshold": thr, "review_cost_gbp": REVIEW_COST,
        "test_rows": int(len(test)), "test_fraud_cases": int(y.sum()),
        "test_roc_auc": float(roc_auc_score(y, t_score)), "test_pr_auc": float(average_precision_score(y, t_score)),
        "precision": float(((f == 1) & (y == 1)).sum() / max(f.sum(), 1)),
        "recall": float(((f == 1) & (y == 1)).sum() / max(y.sum(), 1)),
        "fraud_value_total_gbp": float(amt[y == 1].sum()),
        "fraud_value_caught_gbp": float(amt[(f == 1) & (y == 1)].sum()),
        "review_cost_total_gbp": float(REVIEW_COST * f.sum()),
        "net_saving_gbp": net_saving(y, amt, f),
        "alerts_per_1000_txn": float(1000 * f.mean()),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Segment / fairness monitoring (protected attributes were NOT model inputs)
    seg_rows = []
    for seg in SEGMENTS:
        for val_, g in test.groupby(seg):
            gy, gf = g[TARGET].values, g["flagged"].values
            seg_rows.append({"segment_type": seg, "segment": val_, "transactions": len(g),
                             "fraud_cases": int(gy.sum()),
                             "recall": float(((gf == 1) & (gy == 1)).sum() / max(gy.sum(), 1)),
                             "false_positive_rate": float(((gf == 1) & (gy == 0)).sum() / max((gy == 0).sum(), 1)),
                             "alert_rate": float(gf.mean())})
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(TAB / "segment_performance.csv", index=False)

    # Explainability (SHAP on the selected model if tree-based, else linear)
    sample = test.sample(min(3000, len(test)), random_state=SEED)
    Xs = model.named_steps["prep"].transform(sample[cols])
    names = list(model.named_steps["prep"].get_feature_names_out())
    clf = model.named_steps["clf"]
    explainer = shap.TreeExplainer(clf) if best_name != "Logistic Regression" else shap.LinearExplainer(clf, Xs)
    sv = explainer.shap_values(Xs.toarray() if hasattr(Xs, "toarray") else Xs)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    imp = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "mean_abs_shap": np.abs(sv).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(TAB / "shap_importance.csv", index=False)

    # Plots
    top = imp.head(12).iloc[::-1]
    plt.figure(figsize=(7, 5)); plt.barh(top.feature, top.mean_abs_shap, color="#1F3864")
    plt.title(f"Global feature importance (mean |SHAP|) - {best_name}"); plt.tight_layout()
    plt.savefig(OUT / "shap_importance.png", dpi=130); plt.close()

    p, r, _ = precision_recall_curve(y, t_score)
    plt.figure(figsize=(6, 4.5)); plt.plot(r, p, color="#1F3864")
    plt.axhline(y.mean(), ls="--", color="grey", label=f"baseline = {y.mean():.2%}")
    plt.scatter([summary["recall"]], [summary["precision"]], color="red", zorder=3, label="chosen threshold")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Precision-recall curve (test)")
    plt.tight_layout(); plt.savefig(OUT / "pr_curve.png", dpi=130); plt.close()

    # Tableau-ready transaction table + daily rollup
    keep = ["transaction_id", "timestamp", "customer_id", "region", "age_band", "category", "channel", "amount",
            "hour", "distance_from_home_km", "txn_count_1h", "foreign_txn", "device_change", "is_new_merchant",
            "is_fraud", "fraud_score", "flagged", "outcome"]
    out_tab = test[keep].copy()
    out_tab["timestamp"] = out_tab["timestamp"].dt.floor("s")
    out_tab[["fraud_score"]] = out_tab[["fraud_score"]].round(4)
    out_tab.to_csv(TAB / "test_predictions.csv", index=False)
    daily = test.assign(date=test.timestamp.dt.date).groupby("date").apply(
        lambda g: pd.Series({"transactions": len(g), "fraud_cases": int(g.is_fraud.sum()),
                             "alerts": int(g.flagged.sum()),
                             "fraud_value_gbp": g.loc[g.is_fraud == 1, "amount"].sum(),
                             "fraud_caught_gbp": g.loc[(g.is_fraud == 1) & (g.flagged == 1), "amount"].sum()}),
        include_groups=False).reset_index()
    daily.to_csv(TAB / "daily_summary.csv", index=False)

    joblib.dump({"model": model, "threshold": thr, "model_name": best_name, "columns": cols}, MODELS / "fraud_model.joblib")
    print("saved model + exports")


if __name__ == "__main__":
    main()
