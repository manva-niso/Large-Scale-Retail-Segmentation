import sys, os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

st.set_page_config(page_title="Fashion Retail Segmentation & LTV", layout="wide")

st.title("Fashion Retail Customer Segmentation & LTV Engine")
st.caption(
    "Customer segmentation + lifetime-value prediction on fashion retail transactions, "
    "with a multithreaded C++ layer (via pybind11) accelerating the RFM feature aggregation."
)

# ---------- Load data ----------
@st.cache_data
def load_all():
    segments = pd.read_parquet(f"{DATA_DIR}/customer_segments.parquet")
    ltv = pd.read_parquet(f"{DATA_DIR}/ltv_predictions.parquet")
    with open(f"{DATA_DIR}/benchmark_results.json") as f:
        bench = json.load(f)
    with open(f"{DATA_DIR}/ltv_model_results.json") as f:
        ltv_results = json.load(f)
    return segments, ltv, bench, ltv_results

segments, ltv, bench, ltv_results = load_all()

# ---------- Benchmark panel ----------
st.header("1. C++ vs Python Performance Benchmark")
bench_df = pd.DataFrame(bench)
col1, col2 = st.columns([2, 1])
with col1:
    fig = px.bar(
        bench_df.melt(id_vars="n_transactions", value_vars=["pandas_ms", "cpp_ms"],
                       var_name="method", value_name="milliseconds"),
        x="n_transactions", y="milliseconds", color="method", barmode="group",
        labels={"n_transactions": "Number of Transactions", "milliseconds": "Time (ms)"},
        title="RFM Aggregation: pandas groupby vs multithreaded C++ (dense-array)",
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.metric("Speedup at full dataset size",
               f"{bench_df.iloc[-1]['speedup']:.2f}x",
               help="C++ dense-array aggregation vs pandas groupby, validated for exact correctness")
    st.dataframe(bench_df.set_index("n_transactions").round(2))
st.caption(
    "Correctness note: every C++ result is validated element-for-element against the pandas "
    "baseline before being trusted (see benchmark.py) — this is a genuine, verified speedup, "
    "not just a faster wrong answer."
)

# ---------- Cluster visualization ----------
st.header("2. Customer Segments (K-Means + PCA)")
col1, col2 = st.columns([2, 1])
with col1:
    fig2 = px.scatter(
        segments, x="pca_x", y="pca_y", color="segment_name",
        hover_data=["recency", "frequency", "monetary", "category_diversity"],
        title="Customer segments projected onto 2 principal components",
        labels={"pca_x": "PC1", "pca_y": "PC2"},
    )
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    st.subheader("Segment sizes")
    st.dataframe(segments["segment_name"].value_counts().rename("customers"))

st.subheader("Segment profiles")
profile_cols = ["recency", "frequency", "monetary", "category_diversity", "avg_transaction_value"]
profile = segments.groupby("segment_name")[profile_cols].mean().round(1)
st.dataframe(profile)

# ---------- LTV model ----------
st.header("3. Lifetime Value Prediction")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Model comparison")
    ltv_res_df = pd.DataFrame(ltv_results).T.round(3)
    ltv_res_df.index.name = "model"
    st.dataframe(ltv_res_df)
    st.caption("Predicting next-quarter spend from RFM features, time-split (no leakage): "
               "features computed through Sept 30 2025, target is spend in Oct-Dec 2025.")
with col2:
    st.subheader("Predicted LTV distribution")
    fig3 = px.histogram(ltv, x="predicted_ltv", nbins=40,
                         title="Distribution of predicted next-quarter spend")
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Look up a customer")
customer_ids = ltv["customer_id"].sort_values().tolist()
selected = st.selectbox("Customer ID", customer_ids)
row = ltv[ltv["customer_id"] == selected].iloc[0]
seg_row = segments[segments["customer_id"] == selected]
seg_name = seg_row["segment_name"].iloc[0] if len(seg_row) else "N/A"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Segment", seg_name)
c2.metric("Recency (days)", f"{row['recency']:.0f}")
c3.metric("Frequency", f"{row['frequency']:.0f}")
c4.metric("Predicted next-Q spend", f"${row['predicted_ltv']:.2f}")

st.divider()
st.caption(
    "Data: synthetic fashion-retail transactions generated with the same schema as the "
    "H&M Personalized Fashion Recommendations dataset (real H&M data can be substituted "
    "with no pipeline changes). Built with pandas, scikit-learn, C++17/pybind11, and Streamlit."
)
