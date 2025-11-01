# reporting.py — display-only, simple, and robust with timestamp as index

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import polars as pl
from data_loader import load_data_pd, load_data_polar
from parallel import metrics_threaded, metrics_multiprocess

DATA_PATH = "data/market_data-1.csv"


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def time_block(fn, *args, **kwargs):
    m0 = rss_mb()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    m1 = rss_mb()
    return out, wall, max(m0, m1)

def plot_bar(df, metric):
    sub = df[df["Metric"] == metric]
    plt.figure()
    plt.bar(sub["Library"], sub["Value"])
    plt.title(metric); plt.ylabel("Value"); plt.tight_layout()
    plt.show()

# ---------- rolling (PANDAS) ----------
def rolling_pd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().copy()
    g = df.groupby("symbol", sort=False)

    df["ret"]   = g["price"].pct_change()
    df["ma20"]  = g["price"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["std20"] = g["price"].transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    df["sharpe20"] = g["ret"].transform(
        lambda s: s.rolling(20, min_periods=20).mean() /
                  s.rolling(20, min_periods=20).std(ddof=0)
    )
    return df

# ---------- rolling (POLARS) ----------
def rolling_pl(df: pl.DataFrame) -> pl.DataFrame:
    # df has columns: timestamp, symbol, price
    df = df.sort(["symbol", "timestamp"])
    df = df.with_columns(((pl.col("price")/pl.col("price").shift(1)-1).over("symbol")).alias("ret"))
    df = df.with_columns([
        pl.col("price").rolling_mean(20).over("symbol").alias("ma20"),
        pl.col("price").rolling_std(20).over("symbol").alias("std20"),
        (pl.col("ret").rolling_mean(20) / pl.col("ret").rolling_std(20)).over("symbol").alias("sharpe20"),
    ])
    return df

# ---------- main ----------
def main():
    rows = []

    _, t_pd_in, _ = time_block(load_data_pd, DATA_PATH)
    _, t_pl_in, _ = time_block(load_data_polar, DATA_PATH)
    rows += [["Ingestion time (s)", "pandas", t_pd_in],
             ["Ingestion time (s)", "polars", t_pl_in]]

    df_pd = load_data_pd(DATA_PATH) 
    df_pl = load_data_polar(DATA_PATH)

    _, t_pd_roll, mem_pd = time_block(rolling_pd, df_pd)
    _, t_pl_roll, mem_pl = time_block(rolling_pl, df_pl)
    rows += [["Rolling time (s)", "pandas", t_pd_roll],
             ["Rolling time (s)", "polars", t_pl_roll],
             ["Memory (MB)", "pandas", mem_pd],
             ["Memory (MB)", "polars", mem_pl]]

    _, t_thread, _ = time_block(metrics_threaded, df_pd.reset_index())
    _, t_mp, _     = time_block(metrics_multiprocess, df_pd.reset_index())
    rows += [["Parallel time (s)", "pandas(threaded)", t_thread],
             ["Parallel time (s)", "pandas(multiproc)", t_mp]]

    summary = pd.DataFrame(rows, columns=["Metric", "Library", "Value"])
    
    
    
        # Pretty print: order columns and hide NaNs
    order_cols = ["pandas", "polars", "pandas(threaded)", "pandas(multiproc)"]
    order_rows = ["Ingestion time (s)", "Rolling time (s)", "Parallel time (s)", "Memory (MB)"]

    pivot = (
        summary.pivot(index="Metric", columns="Library", values="Value")
            .reindex(index=order_rows)
            .reindex(columns=order_cols)
    )

    print("\n=== PERFORMANCE SUMMARY ===\n")
    print(pivot.fillna("—").round(4))

    for m in summary["Metric"].unique():
        plot_bar(summary, m)

if __name__ == "__main__":
    main()
