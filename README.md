# Fashion Retail Customer Segmentation & LTV Engine

A customer segmentation and lifetime-value prediction pipeline for fashion retail
transactions, with a multithreaded C++ acceleration layer for the feature
engineering bottleneck.

## Problem

Fashion retailers need to segment customers and predict lifetime value across
millions of transactions, but standard Python tooling (pandas groupby) becomes
a real bottleneck at that scale — especially when this needs to run repeatedly
(e.g. nightly, or per-campaign).

## Approach

1. **Feature engineering** — RFM (Recency, Frequency, Monetary) per customer,
   plus category diversity and average basket size.
2. **Clustering** — K-Means (k=4) on standardized features, PCA for 2D
   visualization, clusters manually labeled by profile (e.g. "Loyal
   High-Spenders", "One-Time Buyers").
3. **Lifetime value prediction** — regression models (Linear Regression vs
   Gradient Boosting) predicting next-quarter spend from a customer's purchase
   history, split by **time** (not randomly) to avoid leakage.

## The optimization: C++ acceleration layer

The RFM aggregation step was identified as the bottleneck. Two implementations
were built and validated against each other:

- **Baseline:** `pandas.groupby().agg()` — vectorized but not thread-parallel,
  and pays hashing overhead per group.
- **C++ layer (via pybind11):** since customer IDs are dense integers, each
  thread accumulates into its own full-size local array (no hashmap, no
  locking), and partial arrays are summed. This is a standard technique when
  the key space is dense rather than sparse/arbitrary.

**Every C++ result is validated element-for-element against the pandas
baseline before any benchmark number is trusted** — a fast wrong answer isn't
useful.

### Benchmark results (measured, `-O3`, 8 threads, averaged over 3 runs)

| Transactions | pandas (ms) | C++ (ms) | Speedup |
|---|---|---|---|
| 50,000 | 10.03 | 1.40 | **7.16x** |
| 200,000 | 15.30 | 3.02 | **5.06x** |
| 500,000 | 26.82 | 5.29 | **5.07x** |
| 1,009,839 | 45.81 | 9.81 | **4.67x** |

## Results

- **Clustering:** 4 segments identified with a silhouette score of 0.49
  (Loyal High-Spenders, Category Loyalists, Seasonal Bargain Hunters,
  One-Time Buyers).
- **LTV model comparison:**

  | Model | RMSE | R² |
  |---|---|---|
  | Linear Regression | 33.61 | 0.562 |
  | Gradient Boosting | 29.34 | **0.666** |

  Gradient Boosting was selected as the production model and refit on the
  full dataset for the dashboard.

## Data note

This build uses a **synthetic dataset generated with the identical schema**
to the H&M Personalized Fashion Recommendations Kaggle dataset (customer_id,
article_id, price, date, category, age, membership status), with 4 built-in
customer archetypes so clustering has real structure to find. Swapping in the
real H&M CSVs requires no pipeline changes — only `python/generate_data.py`
would be skipped in favor of loading the real files into the same schema.

## What I'd do with more time

- Distributed processing (Spark/Ray) for datasets beyond single-machine memory
- A real-time/streaming version (segment updates as transactions arrive)
- Churn prediction as a complementary model to LTV
- Replace manual cluster labeling with an automated profile-to-name mapping

## How to run

```bash
# 1. Build the C++ extension
pip install pybind11 --break-system-packages
python3 setup.py build_ext --inplace

# 2. Generate data (or swap in real H&M CSVs with matching schema)
python3 python/generate_data.py --n_customers 70000 --out data

# 3. Run the pipeline
python3 python/benchmark.py       # C++ vs pandas benchmark + validation
python3 python/clustering.py      # segmentation
python3 python/ltv_model.py       # LTV prediction

# 4. Launch the dashboard
pip install streamlit plotly --break-system-packages
streamlit run dashboard/app.py
```

## Stack

Python (pandas, numpy, scikit-learn), C++17 (`std::thread`), pybind11,
Streamlit, Plotly.
