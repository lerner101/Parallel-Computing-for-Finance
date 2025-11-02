import sys
from pathlib import Path
import pandas as pd
import polars as pl

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from data_loader import load_data_pd, load_data_polar

def test_pd_and_polar_equivalence():
    x = load_data_pd("data/market_data-1.csv")
    y = load_data_polar("data/market_data-1.csv")
    y = y.to_pandas().astype({"symbol": str}).reset_index(drop=True)
    x = x.astype({"symbol": str}).reset_index(drop=True)
    y = y.sort_values(["timestamp", "symbol", "price"]).reset_index(drop=True)
    x = x.sort_values(["timestamp", "symbol", "price"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(y, x, check_exact=False, rtol=1e-9, check_dtype=False)