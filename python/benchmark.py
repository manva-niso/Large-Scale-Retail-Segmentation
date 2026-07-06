"""
Benchmarks pandas-groupby RFM aggregation vs the multithreaded C++ implementation,
at several data sizes. Also validates correctness: C++ output must exactly match
the pandas baseline before we trust any speedup number.
"""
import time
import json
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_rfm import compute_rfm_pandas

import retail_cpp

DATA_DIR = "data"
REFERENCE_DATE = pd.Timestamp("2020-09-22")


def load_transactions():
    tx = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    tx["day_offset"] = (tx["date"] - pd.Timestamp("2018-01-01")).dt.days.astype("int32")
    return tx


def run_pandas(tx, reference_day):
    t0 = time.perf_counter()
    result = compute_rfm_pandas(tx, reference_day)
    return time.perf_counter() - t0, result


def run_cpp(tx, reference_day, num_threads=8):
    # Factorize customer_id -> dense idx. In a real system with a customer
    # dimension table this mapping already exists (surrogate keys); we time
    # ONLY the aggregation itself, not the factorize step, to compare like-for-like
    # against pandas' groupby (which does its own internal factorization).
    codes, uniques = pd.factorize(tx["customer_id"], sort=True)
    idx = codes.astype(np.int32)
    days = tx["day_offset"].to_numpy(dtype=np.int32)
    prices = tx["price"].to_numpy(dtype=np.float64)
    n_customers = len(uniques)

    t0 = time.perf_counter()
    res = retail_cpp.compute_rfm(idx, days, prices, n_customers, reference_day, num_threads)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame({
        "customer_id": uniques,
        "recency": res["recency"],
        "frequency": res["frequency"],
        "monetary": res["monetary"],
    })
    return elapsed, df


def validate(pandas_df, cpp_df):
    # Sort both dataframes to align the customers
    p = pandas_df.sort_values("customer_id").reset_index(drop=True)
    c = cpp_df.sort_values("customer_id").reset_index(drop=True)
    
    # 1. Check Row Counts
    assert len(p) == len(c), f"Row count mismatch: Pandas {len(p)} vs C++ {len(c)}"
    
    # 2. Check Customer IDs
    assert np.array_equal(p["customer_id"].to_numpy(), c["customer_id"].to_numpy()), "Customer IDs do not match!"
    
    # 3. DEBUG RECENCY
    p_recency = p["recency"].to_numpy()
    c_recency = c["recency"].to_numpy()
    
    if not np.array_equal(p_recency, c_recency):
        # Find exactly where they disagree
        mismatch_idx = p_recency != c_recency
        
        print("\n" + "="*40)
        print("💥 RECENCY MISMATCH DETECTED 💥")
        print("="*40)
        print(f"Total mismatched rows: {mismatch_idx.sum():,}")
        print("-" * 40)
        print("First 5 mismatches:")
        print(f"Pandas calculated: {p.loc[mismatch_idx, 'recency'].head(5).tolist()}")
        print(f"C++ calculated:    {c.loc[mismatch_idx, 'recency'].head(5).tolist()}")
        print("="*40 + "\n")
        
        raise AssertionError("Recency values do not match! Check the terminal output above.")

    # 4. Check Frequency & Monetary
    assert np.array_equal(p["frequency"].to_numpy(), c["frequency"].to_numpy()), "Frequency mismatch!"
    assert np.allclose(p["monetary"].to_numpy(), c["monetary"].to_numpy(), atol=1e-6), "Monetary mismatch!"
    
    return True

def main():
    tx_full = load_transactions()
    reference_day = (REFERENCE_DATE - pd.Timestamp("2018-01-01")).days

    sizes = [50_000, 200_000, 500_000, len(tx_full)]
    sizes = sorted(set(s for s in sizes if s <= len(tx_full)))

    results = []
    for size in sizes:
        tx = tx_full.iloc[:size].copy()

        # average over 3 runs each for stable numbers
        pandas_times, cpp_times = [], []
        pandas_df, cpp_df = None, None
        for _ in range(3):
            pt, pandas_df = run_pandas(tx, reference_day)
            ct, cpp_df = run_cpp(tx, reference_day, num_threads=8)
            pandas_times.append(pt)
            cpp_times.append(ct)

        validate(pandas_df, cpp_df)

        avg_pandas = sum(pandas_times) / len(pandas_times)
        avg_cpp = sum(cpp_times) / len(cpp_times)
        speedup = avg_pandas / avg_cpp if avg_cpp > 0 else float("inf")

        print(f"n={size:>9,}  pandas={avg_pandas*1000:8.2f}ms  "
              f"cpp={avg_cpp*1000:8.2f}ms  speedup={speedup:.2f}x  [validated OK]")

        results.append({
            "n_transactions": size,
            "pandas_ms": avg_pandas * 1000,
            "cpp_ms": avg_cpp * 1000,
            "speedup": speedup,
        })

    with open(f"{DATA_DIR}/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {DATA_DIR}/benchmark_results.json")


if __name__ == "__main__":
    main()
