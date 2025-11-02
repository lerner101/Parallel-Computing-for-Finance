import os
import sys
import pandas as pd
import numpy as np

# --- Ensure we can import project modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics import metrics_pd, metrics_pl


def test_metrics_correctness():
    
    df_pd = metrics_pd()
    df_pl = metrics_pl().to_pandas()
    df_pd = df_pd.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df_pl = df_pl.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    df_pl = df_pl[df_pd.columns]

    for col in ["ma20", "std20", "ret", "sharpe20"]:
        pd.testing.assert_series_equal(
            df_pd[col].round(6),
            df_pl[col].round(6),
            check_names=False,
            check_dtype=False,
            atol=1e-6,
            obj=f"Column {col}",
        )

    for col in ["ma20", "std20", "ret", "sharpe20"]:
        assert not df_pd[col].isna().all(), f"{col} in Pandas is all NaN"
        assert not df_pl[col].isna().all(), f"{col} in Polars is all NaN"