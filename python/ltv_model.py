"""
Predicts customer lifetime value (future-period spend) from RFM + segment features.
Splits transactions by TIME (not randomly) into a feature window and a target window,
so this is a genuine forward-looking prediction task, not leakage.
Compares Linear Regression vs Gradient Boosting.
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import retail_cpp

DATA_DIR = "data"

FEATURE_WINDOW_END = pd.Timestamp("2020-06-30")   # features computed using data up to here
TARGET_WINDOW_END = pd.Timestamp("2020-09-22")    # predict spend in the following period


def load_transactions():
    tx = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    tx["day_offset"] = (tx["date"] - pd.Timestamp("2018-01-01")).dt.days.astype("int32")
    return tx


def build_feature_and_target(tx: pd.DataFrame):
    feature_tx = tx[tx["date"] <= FEATURE_WINDOW_END].copy()
    target_tx = tx[(tx["date"] > FEATURE_WINDOW_END) & (tx["date"] <= TARGET_WINDOW_END)].copy()

    reference_day = (FEATURE_WINDOW_END - pd.Timestamp("2018-01-01")).days
    codes, uniques = pd.factorize(feature_tx["customer_id"], sort=True)
    idx = codes.astype(np.int32)
    days = feature_tx["day_offset"].to_numpy(dtype=np.int32)
    prices = feature_tx["price"].to_numpy(dtype=np.float64)
    n_customers = len(uniques)

    res = retail_cpp.compute_rfm(idx, days, prices, n_customers, reference_day, num_threads=8)
    rfm = pd.DataFrame({
        "customer_id": uniques,
        "recency": res["recency"],
        "frequency": res["frequency"],
        "monetary": res["monetary"],
    })
    cat_diversity = feature_tx.groupby("customer_id")["category"].nunique().rename("category_diversity")
    rfm = rfm.set_index("customer_id").join(cat_diversity).fillna(0).reset_index()

    # target: total spend in the following window (0 if no purchases -> churn)
    future_spend = target_tx.groupby("customer_id")["price"].sum().rename("future_spend")
    data = rfm.set_index("customer_id").join(future_spend).fillna({"future_spend": 0.0}).reset_index()
    return data


def main():
    print("Loading transactions and building time-split feature/target windows...")
    tx = load_transactions()
    data = build_feature_and_target(tx)
    print(f"  {len(data):,} customers with features; predicting spend in "
          f"{FEATURE_WINDOW_END.date()} -> {TARGET_WINDOW_END.date()}")

    feature_cols = ["recency", "frequency", "monetary", "category_diversity"]
    X = data[feature_cols].to_numpy()
    y = data["future_spend"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in [
        ("Linear Regression", LinearRegression()),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        results[name] = {"rmse": rmse, "r2": r2}
        print(f"  {name:20s}  RMSE={rmse:8.2f}   R2={r2:.3f}")

    best_model_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nBest model: {best_model_name}")

    import json
    with open(f"{DATA_DIR}/ltv_model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # refit best model on ALL data and save predictions for the dashboard
    best_model = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42) \
        if best_model_name == "Gradient Boosting" else LinearRegression()
    best_model.fit(X, y)
    data["predicted_ltv"] = best_model.predict(X)
    data.to_parquet(f"{DATA_DIR}/ltv_predictions.parquet", index=False)
    print(f"Saved predictions to {DATA_DIR}/ltv_predictions.parquet")


if __name__ == "__main__":
    main()
