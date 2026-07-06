"""
Generates a synthetic fashion-retail transactions dataset with the SAME schema
as the H&M Personalized Fashion Recommendations Kaggle dataset:

  transactions: customer_id, article_id, price, date, sales_channel_id
  customers:    customer_id, age, club_member_status
  articles:     article_id, category

Real H&M data can be dropped in later with zero pipeline changes, since the
column names/types match.

We bake in 4 hidden customer "archetypes" so clustering has real structure to find:
  1. Loyal high-spenders   - frequent, high monetary, recent
  2. Seasonal bargain hunters - moderate frequency, low monetary, bursty timing
  3. One-time buyers       - single purchase, low monetary, old recency
  4. Category loyalists    - frequent but narrow category focus, mid monetary
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
import argparse

CATEGORIES = ["Garment Upper Body", "Garment Lower Body", "Accessories",
              "Footwear", "Outerwear", "Underwear", "Nightwear"]

REF_DATE = date(2025, 12, 31)
START_DATE = date(2024, 1, 1)
TOTAL_DAYS = (REF_DATE - START_DATE).days


def gen_customers(n_customers, rng):
    archetypes = rng.choice(
        ["loyal_high_spender", "seasonal_bargain", "one_time", "category_loyalist"],
        size=n_customers, p=[0.15, 0.35, 0.30, 0.20]
    )
    customer_ids = np.arange(1, n_customers + 1)
    ages = rng.integers(18, 70, size=n_customers)
    club_status = rng.choice(["ACTIVE", "PRE-CREATE", "LEFT CLUB"], size=n_customers, p=[0.7, 0.2, 0.1])
    return pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "club_member_status": club_status,
        "archetype": archetypes,  # kept for validation only, not used by pipeline
    })


def gen_transactions_for_customer(cid, archetype, rng):
    rows = []
    if archetype == "loyal_high_spender":
        n_tx = rng.integers(25, 60)
        days = np.sort(rng.choice(range(TOTAL_DAYS - 30, TOTAL_DAYS), size=n_tx, replace=True))
        price_mu = 45
        fav_cats = rng.choice(CATEGORIES, size=rng.integers(2, 4), replace=False)
    elif archetype == "seasonal_bargain":
        n_tx = rng.integers(6, 15)
        burst_start = rng.integers(0, TOTAL_DAYS - 40)
        days = np.sort(rng.integers(burst_start, burst_start + 30, size=n_tx))
        price_mu = 15
        fav_cats = rng.choice(CATEGORIES, size=rng.integers(1, 3), replace=False)
    elif archetype == "one_time":
        n_tx = 1
        days = rng.integers(0, TOTAL_DAYS - 200, size=n_tx)  # old purchase -> high recency
        price_mu = 25
        fav_cats = rng.choice(CATEGORIES, size=1, replace=False)
    else:  # category_loyalist
        n_tx = rng.integers(15, 30)
        days = np.sort(rng.choice(range(TOTAL_DAYS), size=n_tx, replace=True))
        price_mu = 35
        fav_cats = rng.choice(CATEGORIES, size=1, replace=False)  # narrow focus

    for d in days:
        cat = rng.choice(fav_cats)
        price = max(3.0, rng.normal(price_mu, price_mu * 0.35))
        rows.append((cid, int(d), round(float(price), 2), cat))
    return rows


def main(n_customers, out_path, seed=42):
    rng = np.random.default_rng(seed)
    customers = gen_customers(n_customers, rng)

    all_rows = []
    for cid, archetype in zip(customers.customer_id, customers.archetype):
        all_rows.extend(gen_transactions_for_customer(cid, archetype, rng))

    tx = pd.DataFrame(all_rows, columns=["customer_id", "day_offset", "price", "category"])
    tx["date"] = tx["day_offset"].apply(lambda d: START_DATE + timedelta(days=int(d)))
    tx["article_id"] = (tx["category"].astype("category").cat.codes.astype("int64") * 100000
                         + rng.integers(0, 99999, size=len(tx)))
    tx["sales_channel_id"] = rng.choice([1, 2], size=len(tx))
    tx = tx[["customer_id", "article_id", "price", "date", "category", "sales_channel_id"]]
    tx = tx.sort_values("date").reset_index(drop=True)

    tx.to_parquet(f"{out_path}/transactions.parquet", index=False)
    customers.drop(columns=["archetype"]).to_parquet(f"{out_path}/customers.parquet", index=False)
    customers[["customer_id", "archetype"]].to_parquet(f"{out_path}/_archetypes_groundtruth.parquet", index=False)

    print(f"Customers: {len(customers):,}")
    print(f"Transactions: {len(tx):,}")
    print(f"Date range: {tx.date.min()} -> {tx.date.max()}")
    print(f"Saved to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_customers", type=int, default=20000)
    parser.add_argument("--out", type=str, default="/home/claude/retail-segmentation/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n_customers, args.out, args.seed)
