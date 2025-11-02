import pandas as pd
import metrics
import portfolio
import parallel
import json
import data_loader
from pathlib import Path
import reporting

# ============= Main Function ====================
def main():
    df = pd.read_csv("data/market_data-1.csv")
    path = Path('data/portfolio_structure-1.json')
    data = json.loads(path.read_text())
    result = portfolio.aggregate_portfolio(df,data)
    print(json.dumps(result,indent=2))

    # Ingestion
    t_ing = data_loader.profile_function(data_loader.load_data_pd, "data/market_data-1.csv")
    t_ing_pl = data_loader.profile_function(data_loader.load_data_polar, "data/market_data-1.csv")
    print(f"Ingestion: {t_ing['exec_time_sec']:.3f} seconds")
    print(f"Ingestion Polars: {t_ing_pl['exec_time_sec']:.3f} seconds")

    # Reporting
    reporting.main()





if __name__ == '__main__':
    main()