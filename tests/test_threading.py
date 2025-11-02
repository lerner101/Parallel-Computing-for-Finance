import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import load_data_pd
from parallel import metrics_threaded, metrics_multiprocess


def test_parallel_consistency():
    df = load_data_pd("data/market_data-1.csv")

    threaded = metrics_threaded(df)
    multiprocess = metrics_multiprocess(df)

    threaded = threaded.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    multiprocess = multiprocess.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        threaded,
        multiprocess,
        check_exact=False,
        rtol=1e-9,
        check_dtype=False
    )