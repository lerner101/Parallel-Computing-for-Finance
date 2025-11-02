import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from portfolio import aggregate_portfolio

def test_aggregate_portfolio_totals():
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="D").repeat(2),
        "symbol": ["AAPL", "MSFT"] * 3,
        "price": [100, 200, 110, 210, 120, 220],
    }
    df = pd.DataFrame(data)

    portfolio = {
        "name": "Main Portfolio",
        "owner": "TestUser",
        "positions": [
            {"symbol": "AAPL", "quantity": 10},
            {"symbol": "MSFT", "quantity": 5}
        ]
    }

    expected_total = 2300

    result = aggregate_portfolio(df, portfolio)

    assert abs(result["total_value"] - expected_total) < 1e-6, (
        f"Expected total {expected_total}, got {result['total_value']}"
    )