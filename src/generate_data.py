"""Generate a SYNTHETIC card-transaction dataset with realistic fraud patterns.

Real bank data is confidential, so this project uses a seeded synthetic table.
The rest of the pipeline works unchanged on any table with the same columns.
"""
import numpy as np
import pandas as pd

SEED = 42
CATEGORIES = ["grocery", "fuel", "travel", "electronics", "restaurants", "online_retail", "utilities", "atm_cash"]
CHANNELS = ["pos", "online", "atm", "contactless"]
REGIONS = ["London", "South East", "North West", "Scotland", "Wales", "Midlands"]
AGE_BANDS = ["18-25", "26-40", "41-60", "61+"]


def generate(n_txn: int = 200_000, n_customers: int = 8_000, days: int = 90, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cust = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "region": rng.choice(REGIONS, n_customers, p=[.25, .2, .17, .1, .08, .2]),
        "age_band": rng.choice(AGE_BANDS, n_customers, p=[.15, .35, .33, .17]),
        "avg_spend": rng.lognormal(3.4, 0.5, n_customers),
    })
    df = pd.DataFrame({"customer_id": rng.integers(0, n_customers, n_txn)}).merge(cust, on="customer_id")
    n = len(df)
    df["timestamp"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.uniform(0, days * 86400, n), unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["category"] = rng.choice(CATEGORIES, n, p=[.24, .1, .05, .07, .18, .2, .1, .06])
    df["channel"] = rng.choice(CHANNELS, n, p=[.35, .3, .07, .28])
    df["amount"] = (df["avg_spend"] * rng.lognormal(0, 0.7, n)).round(2).clip(lower=0.5)
    df["amount_ratio"] = (df["amount"] / df["avg_spend"]).round(3)
    df["distance_from_home_km"] = rng.gamma(1.5, 8, n).round(1)
    df["txn_count_1h"] = rng.poisson(0.4, n)
    df["is_new_merchant"] = rng.binomial(1, 0.12, n)
    df["device_change"] = rng.binomial(1, 0.04, n)
    df["foreign_txn"] = rng.binomial(1, 0.05, n)

    z = (-7.9
         + 1.1 * np.log1p(df["amount_ratio"])
         + 0.9 * df["hour"].isin([0, 1, 2, 3, 4]).astype(int)
         + 0.55 * np.minimum(df["txn_count_1h"], 6)
         + 0.012 * np.minimum(df["distance_from_home_km"], 300)
         + 1.3 * df["is_new_merchant"]
         + 2.0 * df["device_change"]
         + 1.7 * df["foreign_txn"]
         + 1.0 * (df["channel"] == "online").astype(int)
         + 0.6 * df["category"].isin(["electronics", "travel"]).astype(int)
         # non-linear interaction: late-night online spend at a new merchant is disproportionately risky
         + 2.2 * ((df['channel'] == 'online') & (df['is_new_merchant'] == 1) & df['hour'].isin([0, 1, 2, 3, 4])).astype(int)
         + 0.9 * ((df['amount_ratio'] > 3) & (df['txn_count_1h'] >= 2)).astype(int)
         + rng.normal(0, 0.5, n))
    df["is_fraud"] = rng.binomial(1, 1 / (1 + np.exp(-z)))
    df["transaction_id"] = np.arange(n)
    return df.drop(columns=["avg_spend"])


if __name__ == "__main__":
    d = generate()
    d.to_csv("data/transactions.csv", index=False)
    print(f"{len(d):,} rows | fraud rate {d.is_fraud.mean():.3%} | {d.timestamp.min()} -> {d.timestamp.max()}")
