# parallel_metrics.py
import os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from data_loader import load_data_pd


def compute_metrics_one(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma20"] = out["price"].rolling(20, min_periods=20).mean()
    out["std20"] = out["price"].rolling(20, min_periods=20).std(ddof=0)
    out["ret"] = out["price"].pct_change()
    out["sharpe20"] = (
        out["ret"].rolling(20, min_periods=20).mean() /
        out["ret"].rolling(20, min_periods=20).std(ddof=0)
    )
    return out


def split_by_symbol(df: pd.DataFrame):
    return [g for _, g in df.groupby("symbol", sort=False)]


def metrics_threaded(df: pd.DataFrame) -> pd.DataFrame:
    parts = split_by_symbol(df)
    results = []
    max_workers = len(parts)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(compute_metrics_one, part) for part in parts]
        for f in as_completed(futures):
            results.append(f.result())
    return pd.concat(results, ignore_index=True)

def compute_metrics_one_wrapper(df_sym: pd.DataFrame) -> pd.DataFrame:
    return compute_metrics_one(df_sym)

def metrics_multiprocess(df: pd.DataFrame) -> pd.DataFrame:
    parts = split_by_symbol(df)
    max_workers = len(parts)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(compute_metrics_one_wrapper, parts, chunksize=1):
            results.append(res)
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    df = load_data_pd("data/market_data-1.csv")
    #print(split_by_symbol(df))
    #print(len(split_by_symbol(df)))
    print(metrics_multiprocess(df))




