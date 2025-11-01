# parallel_metrics.py
import os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from data_loader import load_data_pd
import os, time, sys, resource

'''
In Python, the Global Interpreter Lock (GIL) lets only one thread execute Python bytecode at a time, 
so CPU-bound work in pure Python won't speed up with threads. threads are good mainly for I/O-bound tasks or 
when heavy vectorized C/NumPy/Pandas kernels release the GIL (not all do, and rolling/groupby chains can 
still contend and incur Python-level overhead). Multiprocessing avoids the GIL by using separate processes, 
giving true parallelism for per-symbol, CPU-heavy computations like your rolling stats—often faster on 
multi-core machines—at the cost of higher overhead (process startup), data serialization/copying between processes, 
and more memory usage. So prefer multiprocessing when the workload is compute-intensive and relatively coarse, 
and prefer threading for I/O waits or pipelines dominated by GIL-releasing C code. 
'''

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




def _peak_rss_mb() -> float:
    # ru_maxrss: KB on Linux, bytes on macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / (1024**2)) if sys.platform == "darwin" else (r / 1024)


def run_benchmark(label, func, df):
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    c0 = time.process_time()

    _ = func(df)

    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    rss_after = _peak_rss_mb()

    print(f"{label:12} | wall: {wall:7.3f}s | cpu: {cpu:7.3f}s | peak RSS≈ {max(rss_before, rss_after):7.1f} MB")


if __name__ == "__main__":
    df = load_data_pd("data/market_data-1.csv")


    print("\nBenchmark results:")
    run_benchmark("threaded", metrics_threaded, df)
    run_benchmark("multiprocess", metrics_multiprocess, df)




