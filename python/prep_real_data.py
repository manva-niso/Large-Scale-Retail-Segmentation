"""
Converts the REAL H&M Kaggle CSVs into the same parquet schema the pipeline
already uses (transactions.parquet, customers.parquet).
"""
import pandas as pd
import argparse

def main(raw_dir, out_dir):
    print("Loading raw CSVs (this may take a few minutes, transactions file is large)...")
    tx = pd.read_csv(f"{raw_dir}/transactions_train.csv",
                      dtype={"article_id": str})
    customers = pd.read_csv(f"{raw_dir}/customers.csv")
    articles = pd.read_csv(f"{raw_dir}/articles.csv", dtype={"article_id": str})

    print(f"Raw transactions: {len(tx):,}")

    tx = tx.rename(columns={"t_dat": "date"})
    tx["date"] = pd.to_datetime(tx["date"])

    articles_small = articles[["article_id", "product_group_name"]].rename(
        columns={"product_group_name": "category"}
    )
    tx = tx.merge(articles_small, on="article_id", how="left")

    tx = tx[["customer_id", "article_id", "price", "date", "category", "sales_channel_id"]]
    customers = customers[["customer_id", "age", "club_member_status"]]

    tx.to_parquet(f"{out_dir}/transactions.parquet", index=False)
    customers.to_parquet(f"{out_dir}/customers.parquet", index=False)

    print(f"Saved {len(tx):,} transactions and {len(customers):,} customers to {out_dir}/")
    print(f"Date range: {tx['date'].min().date()} -> {tx['date'].max().date()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
