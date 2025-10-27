import pandas as pd
import polars as pl
import time
import tracemalloc


def load_data_pd(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df[['symbol', 'price']]
    return df


def load_data_polar(file_path):
    df = pl.read_csv(file_path)
    df = df.with_columns([
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("timestamp")
    ])
    df = df.with_columns([
        pl.col("timestamp").set_sorted()
    ])
    df = df.select(["timestamp", "symbol", "price"])
    #df = df.sort("timestamp")
    df = df.set_sorted("timestamp")
    return df


def profile_function(func, file_path):
    tracemalloc.start()
    start_time = time.perf_counter()
    df = func(file_path)
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "result": df,
        "exec_time_sec": end_time - start_time,
        "mem_peak_mb": peak / (1024 * 1024)
    }

if __name__ == "__main__":
    file_path = "data/market_data-1.csv"
    print("Profiling load_data_pd:")
    pd_profile = profile_function(load_data_pd, file_path)
    print(f"Time: {pd_profile['exec_time_sec']:.6f} sec")
    print(f"Peak memory: {pd_profile['mem_peak_mb']:.3f} MB")

    print("\nProfiling load_data_polar:")
    polar_profile = profile_function(load_data_polar, file_path)
    print(f"Time: {polar_profile['exec_time_sec']:.6f} sec")
    print(f"Peak memory: {polar_profile['mem_peak_mb']:.3f} MB")



