# Card Fraud Detection: from model to business decision

End-to-end fraud detection project: data -> time-aware validation -> cost-based decision threshold ->
explainability -> fairness monitoring -> scoring API -> Tableau dashboard.

> **Data note:** transactions are **synthetic** (seeded generator in `src/generate_data.py`) because real bank data is
> confidential. Results demonstrate the method, not real-world fraud performance. The pipeline runs unchanged on any
> table with the same columns.

## Results (held-out last 15 days, 33,314 transactions, 273 fraud cases)

| Model | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| Logistic Regression | 0.845 | 0.155 |
| Random Forest | 0.818 | 0.162 |
| XGBoost (selected on validation PR-AUC) | 0.823 | 0.159 |

The three models are close; XGBoost was chosen on validation PR-AUC by a small margin, and logistic regression has the
highest ROC-AUC. A simpler model would be easier to defend in a regulated setting, which is worth discussing.

At the cost-optimal threshold (0.83, tuned on validation, frozen for test): recall 36.6%, precision 16.2%,
18.6 alerts per 1,000 transactions, GBP 6.9k of GBP 17.2k fraud value caught, **net saving about GBP 3.8k** after a GBP 5
review cost per alert (cost assumption, adjustable in `train.py`).

## Design choices
- **Time-based split** (train days 1-60, validation 61-75, test 76-90): no look-ahead leakage.
- **PR-AUC as headline metric**: fraud is under 1% of transactions, so accuracy and ROC-AUC mislead.
- **Threshold chosen by business cost** (value of fraud caught minus review cost), not by default 0.5.
- **Explainability:** SHAP global importance and per-transaction reasons from the API.
- **Fairness monitoring:** `age_band` and `region` are excluded from the model; recall and false-positive rate are
  tracked by segment (`tableau/segment_performance.csv`). Some segments have few fraud cases, so rates are noisy. For
  example Wales has 21 cases and lower recall, which is flagged as something to investigate, not a conclusion.
- **Auditability:** the API returns the score, decision and top 3 reasons for each transaction.

## Run it
```bash
pip install -r requirements.txt
python src/generate_data.py          # creates data/transactions.csv
cd src && python train.py && cd ..   # trains, evaluates, writes outputs/ and tableau/
pytest -q tests                      # 5 tests
uvicorn api:app --app-dir src        # http://127.0.0.1:8000/docs
```

## Limitations
Synthetic data with hand-built fraud patterns; no concept-drift or adversarial adaptation; flat GBP 5 review cost;
no real customer-impact modelling (declined genuine customers); would need privacy review, monitoring and model-risk
governance before production use.
