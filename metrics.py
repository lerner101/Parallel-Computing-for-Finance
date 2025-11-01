from data_loader import load_data_polar, load_data_pd
import polars as pl
import pandas as pd
import time
import matplotlib.pyplot as plt

'''
The main difference between the Pandas and Polars versions is syntax and performance. In Pandas, 
calculations use groupby with transform and rolling, which run on a single core and 
can be slower for large datasets. Polars uses expressions like pl.col("price").rolling_mean(...).over("symbol"), 
which are more compact and run in parallel using a faster backend (Rust). Both produce the same results, but Polars 
generally finishes much faster and uses less memory, while Pandas is easier to integrate with existing Python tools like Matplotlib.
'''


def metrics_pd():
    df = load_data_pd("data/market_data-1.csv")

    df["ma20"] = df.groupby("symbol")["price"].transform(
        lambda x: x.rolling(window=20).mean()
    )
    df["std20"] = df.groupby("symbol")["price"].transform(
        lambda x: x.rolling(window=20).std()
    )
    df["ret"] = df.groupby("symbol")["price"].pct_change()
    df["sharpe20"] = df.groupby("symbol")["ret"].transform(
        lambda x: x.rolling(window=20).mean() / x.rolling(window=20).std()
    )

    return df
    


def metrics_pl():
    df = load_data_polar("data/market_data-1.csv")


    # percent returns per symbol
    df = df.with_columns(
        (pl.col("price") / pl.col("price").shift(1) - 1)
        .over("symbol")
        .alias("ret")
    )

    df = df.with_columns([
        pl.col("price").rolling_mean(window_size=20).over("symbol").alias("ma20"),

        pl.col("price").rolling_std(window_size=20).over("symbol").alias("std20"),

        (
            pl.col("ret").rolling_mean(window_size=20) / pl.col("ret").rolling_std(window_size=20)
        ).over("symbol").alias("sharpe20")
    ])

    return df



def timeit(fn, label):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label} runtime: {dt:.4f} s")
    return out, dt

def visualize_one_symbol_pd(df_pd, symbol="AAPL"):
    sub = df_pd[df_pd["symbol"] == symbol].dropna(subset=["ma20", "sharpe20"])
    plt.figure()
    plt.plot(sub["price"], label="price")
    plt.plot(sub["ma20"], label="ma20")
    plt.title(f"{symbol}: Price & 20-Period MA (Pandas)")
    plt.xlabel("time")
    plt.ylabel("price")
    plt.legend()
    plt.tight_layout()
    # Sharpe
    plt.figure()
    plt.plot(sub["sharpe20"], label="sharpe20")
    plt.title(f"{symbol}: 20-Period Rolling Sharpe (Pandas)")
    plt.xlabel("time")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("Computing metrics and timing…")
    df_pd, t_pd = timeit(metrics_pd, "Pandas")
    df_pl, t_pl = timeit(metrics_pl, "Polars")

    visualize_one_symbol_pd(df_pd, symbol="AAPL")