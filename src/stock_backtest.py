"""
个股级回测引擎

SMA250制度择时框架 + 多因子个股精选
RISKON/NEUTRAL/RISKOFF时用个股，CRISIS时用ETF防御
"""
import pandas as pd
import numpy as np
from config import START_DATE
from src.stock_data import compute_stock_factors, select_top_stocks


def run_stock_backtest(
    df_close,          # ETF对齐价格（用于防御资产）
    regime_series,     # 制度分 Series
    stock_data,        # {code: DataFrame} 个股数据
    st_filter=None,    # ST股票集合
    top_n=10,          # 选股数量
    verbose=True,
):
    """
    个股增强回测

    RISKON:   95% 个股（多因子选 Top-N）
    NEUTRAL:  60% 个股 + 30% 国债 + 10% 黄金
    RISKOFF:  30% 个股 + 50% 国债 + 10% 黄金 + 10% 货币
    CRISIS:   0% 个股 + 65% 国债 + 15% 黄金 + 20% 货币
    """
    df_close = df_close.sort_index()
    start_mask = df_close.index >= START_DATE
    df_close = df_close[start_mask]

    equity_etfs = ['sh510300', 'sh510500', 'sz159915']
    bond_etf = 'sh511010'
    gold_etf = 'sh518880'
    cash_etf = 'sh511880'

    # Ensure stock data is sorted
    for code in stock_data:
        stock_data[code] = stock_data[code].sort_index()

    allocation = {
        'RISKON':  {'equity': 0.95, 'bond': 0.00, 'gold': 0.00, 'cash': 0.05},
        'NEUTRAL': {'equity': 0.60, 'bond': 0.30, 'gold': 0.05, 'cash': 0.05},
        'RISKOFF': {'equity': 0.30, 'bond': 0.50, 'gold': 0.10, 'cash': 0.10},
        'CRISIS':  {'equity': 0.00, 'bond': 0.65, 'gold': 0.15, 'cash': 0.20},
    }

    initial_capital = 1_000_000
    cash = initial_capital
    positions = {}  # {symbol: shares}
    portfolio_values = []
    rebalance_dates = []

    monthly_dates = df_close.groupby(df_close.index.to_period('M')).apply(lambda x: x.index[-1])
    monthly_dates = pd.DatetimeIndex(monthly_dates)

    if verbose:
        print(f"Stock Backtest: {df_close.index[0].date()} ~ {df_close.index[-1].date()}")
        print(f"Stocks available: {len(stock_data)}, Top-N: {top_n}")
        print(f"Rebalance months: {len(monthly_dates)}")

    warmup_days = 252

    for i, date in enumerate(df_close.index):
        today_prices = df_close.loc[date]

        # Calculate portfolio value
        position_value = 0
        for sym, shares in positions.items():
            # Check stock data for individual stocks
            if sym in stock_data:
                sdf = stock_data[sym]
                if date in sdf.index and not pd.isna(sdf.loc[date, 'close']):
                    position_value += shares * sdf.loc[date, 'close']
            elif sym in today_prices and not pd.isna(today_prices[sym]):
                position_value += shares * today_prices[sym]

        portfolio_value = cash + position_value
        portfolio_values.append({"date": date, "value": portfolio_value})

        # Monthly rebalance
        if date in monthly_dates and i >= warmup_days:
            rebalance_dates.append(date)

            # Get regime
            if regime_series is not None and date in regime_series.index:
                s_val = float(regime_series.loc[date])
                if s_val >= 0.70: regime = 'RISKON'
                elif s_val >= 0.50: regime = 'NEUTRAL'
                elif s_val >= 0.30: regime = 'RISKOFF'
                else: regime = 'CRISIS'
            else:
                regime = 'NEUTRAL'

            if verbose and len(rebalance_dates) <= 3:
                print(f"  {date.date()}: regime={regime}")

            alloc = allocation[regime]

            # === Stock Selection (during RISKON/NEUTRAL/RISKOFF) ===
            selected_stocks = []
            if alloc['equity'] > 0 and stock_data:
                # Filter ST stocks
                valid_stocks = {}
                for code, sdf in stock_data.items():
                    if st_filter and code in st_filter:
                        continue
                    if date in sdf.index and len(sdf[sdf.index <= date]) >= 250:
                        valid_stocks[code] = sdf

                if valid_stocks:
                    scores = compute_stock_factors(valid_stocks, date)
                    selected_stocks = select_top_stocks(scores, top_n)

            # === Build Target Weights ===
            target_weights = {}

            if regime == 'CRISIS' or not selected_stocks:
                # CRISIS or no stocks selected: use ETFs
                target_weights[bond_etf] = alloc['bond']
                target_weights[gold_etf] = alloc['gold']
                target_weights[cash_etf] = alloc['cash']
            else:
                # Equity allocation to individual stocks
                n_stocks = len(selected_stocks)
                per_stock_weight = alloc['equity'] / n_stocks

                for code in selected_stocks:
                    target_weights[code] = per_stock_weight

                # Bond/Gold/Cash from ETFs
                if alloc['bond'] > 0:
                    target_weights[bond_etf] = alloc['bond']
                if alloc['gold'] > 0:
                    target_weights[gold_etf] = alloc['gold']
                if alloc['cash'] > 0:
                    target_weights[cash_etf] = alloc['cash']

            # === Execute ===
            # Sell non-target positions
            for sym in list(positions.keys()):
                if sym not in target_weights or target_weights[sym] == 0:
                    # Get price
                    price = _get_price(sym, date, stock_data, df_close)
                    if price is not None and price > 0:
                        cash += positions[sym] * price * (1 - 0.00125)
                        del positions[sym]

            # Buy/Adjust target positions
            for sym, weight in target_weights.items():
                if weight <= 0:
                    continue

                price = _get_price(sym, date, stock_data, df_close)
                if price is None or price <= 0:
                    continue

                target_value = portfolio_value * weight
                # Individual stocks: 100 shares per lot; ETFs: 100 shares
                lot_size = 100
                target_shares = int(target_value / price / lot_size) * lot_size
                current_shares = positions.get(sym, 0)
                diff = target_shares - current_shares

                if diff > 0:
                    cost = diff * price * 1.00125  # 0.025% comm + 0.1% slip
                    if cost <= cash:
                        cash -= cost
                        positions[sym] = target_shares
                elif diff < 0:
                    cash += abs(diff) * price * 0.99875
                    positions[sym] = target_shares

    # === Metrics ===
    df_values = pd.DataFrame(portfolio_values).set_index("date")
    df_values["return"] = df_values["value"].pct_change()
    rets = df_values["return"].dropna()

    total_ret = df_values["value"].iloc[-1] / initial_capital - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.expanding().max() - 1).min()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0

    annual_returns = rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1) * 100

    return {
        "values": df_values,
        "metrics": {
            "annual_return": ann_ret * 100,
            "max_drawdown": max_dd * 100,
            "sharpe_ratio": sharpe,
            "total_return": total_ret * 100,
            "volatility": rets.std() * np.sqrt(252) * 100,
            "annual_returns": annual_returns,
        },
        "rebalance_dates": rebalance_dates,
    }


def _get_price(sym, date, stock_data, df_close):
    """获取任意symbol在指定日期的价格"""
    if sym in stock_data:
        sdf = stock_data[sym]
        if date in sdf.index:
            return float(sdf.loc[date, 'close'])
        # Find nearest previous date
        prev = sdf[sdf.index < date]
        if len(prev) > 0:
            return float(prev.iloc[-1]['close'])
    elif sym in df_close.columns:
        if date in df_close.index:
            val = df_close.loc[date, sym]
            if not pd.isna(val):
                return float(val)
    return None
