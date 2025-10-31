from unittest import result
import pandas as pd
from pathlib import Path
from multiprocessing import Pool,cpu_count
import numpy as np
import json

# Compute Metric for Each Portfolio
def compute_metrics(df, pos):
    name = pos['symbol']
    ret = df[df['symbol']==name]['price'].pct_change()
    prices =  df[df['symbol']==name]['price']

    value = prices.iloc[-1]*pos['quantity']
    std20 =  ret.rolling(20).std()
    vol = std20.iloc[-1]

    #Draw Down
    cummax = prices.cummax()
    drawdowns = (prices - cummax) / cummax
    max_drawdown = drawdowns.min()

    return {'symbol':name,
            'total_value': value,
            'volatility': vol,
            'max_drawdown': max_drawdown,
            }


# ============= Compute Portoflio Paralelly =============
def compute_pos(df,positions):
    n = len(positions)
    results = None
    with Pool(cpu_count()) as pool:
        results = pool.starmap(compute_metrics, [(df, pos) for pos in positions])
    return results

# =============  Recursive aggregation ====================
def aggregate_portfolio(df, portfolio):
    positions = portfolio.get('positions', [])
    if positions:
        pos_metrics = compute_pos(df,positions)
    else:
        pos_metrics = []
    
    #Recursively compute sub-portf metrics
    sub_portfolios = []
    for sub in portfolio.get('sub_portfolios', []):
        sub_result = aggregate_portfolio(df, sub)
        sub_portfolios.append(sub_result)
        pos_metrics.extend(sub_result.get('positions', []))
    
    #Agregate Metrics
    valid_positions = [p for p in pos_metrics if p['total_value'] is not None]
    if valid_positions:
        print(pos_metrics)
        tot_value = np.sum([p['total_value'] for p in pos_metrics if p['total_value'] is not None])
        weights = np.array([p['total_value'] / tot_value for p in valid_positions])
        aggregate_volatility = np.sum(np.array([p['volatility']*w for p,w in zip(valid_positions,weights)]))
        max_drawdown = min(p['max_drawdown'] for p in valid_positions)
    else:
        tot_value = aggregate_volatility = max_drawdown = None
    
    return {
        'name': portfolio['name'],
        'owner': portfolio.get('owner'),
        'total_value': tot_value,
        'aggregate_volatility': aggregate_volatility,
        'max_drawdown': max_drawdown,
        'positions': valid_positions,
        'sub_portfolios': sub_portfolios
    }


# =============  Sequential version   =====================
def compute_all_positions_sequential(positions):
    """Compute all position metrics sequentially (for comparison)."""
    return [compute_metrics(p) for p in positions]



if __name__ == '__main__':
    df = pd.read_csv("data/market_data-1.csv")
    path = Path('data/portfolio_structure-1.json')
    data = json.loads(path.read_text())
    result = aggregate_portfolio(df,data)
    print(json.dumps(result,indent=2))