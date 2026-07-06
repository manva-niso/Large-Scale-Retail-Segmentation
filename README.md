# Fashion Retail Customer Segmentation & LTV Engine

A customer segmentation and lifetime-value prediction pipeline for fashion retail transactions, featuring a multithreaded C++ acceleration layer built to handle a 31.7+ million row feature engineering bottleneck.

## Problem

Fashion retailers need to segment customers and predict lifetime value across millions of transactions, but standard Python tooling (like `pandas.groupby`) becomes a severe bottleneck at that scale — especially when memory constraints cause Out-Of-Memory (OOM) crashes during nightly or per-campaign runs.

## Approach

1. **Feature engineering** — RFM (Recency, Frequency, Monetary) per customer computed via custom C++, plus highly optimized, vectorized pandas aggregations for category diversity and average basket size.
2. **Clustering** — `MiniBatchKMeans` (k=4) on standardized features, PCA for 2D visualization, with clusters mapped to actionable profiles (e.g., "Loyal High-Spenders", "One-Time Buyers").
3. **Lifetime value prediction** — Gradient Boosting regression predicting next-quarter spend from a customer's purchase history, split by **time** (not randomly) to prevent data leakage and reliably rank users into future spending tiers.

## The optimization: C++ acceleration layer

The RFM aggregation step was identified as the primary memory and speed bottleneck. Python's Global Interpreter Lock (GIL) and memory reallocation severely hindered scaling.

- **Baseline:** `pandas.groupby().agg()` — vectorized but single-threaded, suffering from massive memory overhead on 30M+ rows.
- **C++ layer (via pybind11):** Customer IDs are factored into dense integers. Each thread accumulates into its own full-size local array (bypassing hashmap overhead and thread locking), summing partial arrays at the end.

**Every C++ result is validated element-for-element against the pandas baseline before any benchmark number is trusted.**

### Benchmark results (Measured on Kaggle H&M Dataset)

| Transactions | pandas (ms) | C++ (ms) | Speedup |
|---|---|---|---|
| 50,000 | 79.82 | 5.33 | **14.9x** |
| 200,000 | 50.51 | 5.29 | **9.5x** |
| 500,000 | 126.10 | 7.99 | **15.7x** |
| **31,788,324** | **8055.27** | **273.37** | **29.4x** |

*At full scale, the C++ engine achieves near $O(N)$ time complexity, executing 29x faster than standard Python pipelines while bypassing Python memory constraints entirely.*

## Results

- **Clustering:** 1.36 million unique customers successfully segmented into 4 profiles (Loyal High-Spenders, Category Loyalists, Seasonal Bargain Hunters, One-Time Buyers).
- **LTV Prediction:** Gradient Boosting model deployed to accurately rank and categorize customers by projected future value, providing actionable targeting segments for marketing focus.
- **Dashboard:** Full pipeline insights and interactive PCA visualizations served via a web dashboard.

## What I'd do with more time

- Distributed processing (Spark/Ray) for datasets beyond single-machine memory.
- A real-time/streaming version (segment updates as transactions arrive).
- Churn prediction as a complementary model to LTV.
- Replace manual cluster labeling with an automated profile-to-name mapping.

## How to run

```bash
# 1. Build the C++ extension
pip install pybind11
python setup.py build_ext --inplace

# 2. Run the pipeline (Requires dataset in /data)
python python/benchmark.py       # C++ vs pandas benchmark + validation
python python/clustering.py      # segmentation
python python/ltv_model.py       # LTV prediction

# 3. Launch the dashboard
pip install streamlit plotly
streamlit run dashboard/app.py