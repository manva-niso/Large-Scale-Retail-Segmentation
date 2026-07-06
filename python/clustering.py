"""
Builds the full customer feature table (RFM via C++ + category diversity/basket
size via pandas), runs K-Means clustering, PCA for visualization, and assigns
human-readable segment names based on cluster centroid characteristics.
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import retail_cpp

DATA_DIR = "data"
REFERENCE_DATE = pd.Timestamp("2020-09-22")


def load_transactions():
    tx = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    tx["day_offset"] = (tx["date"] - pd.Timestamp("2018-01-01")).dt.days.astype("int32")
    return tx


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    reference_day = (REFERENCE_DATE - pd.Timestamp("2018-01-01")).days

    # --- RFM via the C++ layer ---
    codes, uniques = pd.factorize(tx["customer_id"], sort=True)
    idx = codes.astype(np.int32)
    days = tx["day_offset"].to_numpy(dtype=np.int32)
    prices = tx["price"].to_numpy(dtype=np.float64)
    n_customers = len(uniques)

    res = retail_cpp.compute_rfm(idx, days, prices, n_customers, reference_day, num_threads=8)
    rfm = pd.DataFrame({
        "customer_id": uniques,
        "recency": res["recency"],
        "frequency": res["frequency"],
        "monetary": res["monetary"],
    })

    # --- extra features via pandas (OPTIMIZED) ---
    print("  Calculating category diversity and average spend...")
    stats = tx.groupby("customer_id", as_index=False).agg(
        category_diversity=("category", "nunique"),
        avg_transaction_value=("price", "mean")
    )
    
    print("  Calculating top category...")
    top_cat = (
        tx.groupby(["customer_id", "category"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .drop_duplicates(subset=["customer_id"])
        .rename(columns={"category": "top_category"})
        [["customer_id", "top_category"]]
    )
    
    # Merge them all together
    features = rfm.merge(stats, on="customer_id", how="left")
    features = features.merge(top_cat, on="customer_id", how="left")
    
    return features


def run_clustering(features: pd.DataFrame, k=4):
    feature_cols = ["recency", "frequency", "monetary", "category_diversity", "avg_transaction_value"]
    X = features[feature_cols].to_numpy()
    
    print("Scaling features...")
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Running MiniBatchKMeans for {k} clusters...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000, n_init="auto")
    labels = km.fit_predict(X_scaled)
    
    print("Calculating silhouette score...")
    sil = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)

    print("Running PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    # Attach results back to the dataframe
    features = features.copy()
    features["cluster"] = labels
    features["pca_x"] = coords[:, 0]
    features["pca_y"] = coords[:, 1]

    return features, km, sil, feature_cols


def name_segments(features: pd.DataFrame, feature_cols) -> dict:
    """Rank clusters by monetary value & frequency to assign readable names."""
    profile = features.groupby("cluster")[feature_cols].mean().sort_values("monetary", ascending=False)
    names = {}
    ranked = profile.index.tolist()
    labels_pool = [
        "Loyal High-Spenders",
        "Category Loyalists",
        "Seasonal Bargain Hunters",
        "One-Time Buyers",
    ]
    for i, cluster_id in enumerate(ranked):
        names[cluster_id] = labels_pool[i] if i < len(labels_pool) else f"Segment {cluster_id}"
    return names


def main():
    print("Loading transactions...")
    tx = load_transactions()

    print("Building features (RFM via C++ + category stats via pandas)...")
    features = build_features(tx)
    print(f"  {len(features):,} customers, columns: {list(features.columns)}")

    print("Running K-Means + PCA...")
    features, km, sil, feature_cols = run_clustering(features, k=4)
    print(f"  Silhouette score: {sil:.3f}")

    names = name_segments(features, feature_cols)
    features["segment_name"] = features["cluster"].map(names)

    print("\nSegment profile summary:")
    profile = features.groupby("segment_name")[feature_cols + ["customer_id"]].agg(
        {**{c: "mean" for c in feature_cols}, "customer_id": "count"}
    ).rename(columns={"customer_id": "n_customers"})
    print(profile.round(1))

    features.to_parquet(f"{DATA_DIR}/customer_segments.parquet", index=False)
    print(f"\nSaved to {DATA_DIR}/customer_segments.parquet")


if __name__ == "__main__":
    main()