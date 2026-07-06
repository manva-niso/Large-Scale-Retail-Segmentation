"""Pure-pandas RFM aggregation -- the 'before' baseline we benchmark C++ against."""
import pandas as pd
import numpy as np


def compute_rfm_pandas(tx: pd.DataFrame, reference_day: int) -> pd.DataFrame:
    """
    tx must have columns: customer_id, day_offset (int), price (float)
    Returns a DataFrame indexed by customer_id with recency/frequency/monetary.
    """
    grouped = tx.groupby("customer_id").agg(
        last_day=("day_offset", "max"),
        frequency=("day_offset", "count"),
        monetary=("price", "sum"),
    )
    grouped["recency"] = reference_day - grouped["last_day"]
    return grouped[["recency", "frequency", "monetary"]].reset_index()
