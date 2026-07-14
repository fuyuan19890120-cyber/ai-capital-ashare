"""
风控增强版回测 — 对比基线 vs 四层风控
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import START_DATE
from src.stock_data import compute_stock_factors, select_top_stocks
from src.risk_manager import RiskManager, RiskLimits, DrawdownController, classify_sector


def run_risk_backtest(
    df_close,
    regime_series,
    stock_data,
    top_n=15,
    verbose=True,
    enable_risk=True,
):
    """
    个股增强回测 + 四层风控

    风控层（仅在 enable_risk=True 时生效）：
    1. 单票上限 15%
    2. 板块上限 40%
    3. 回撤熔断 -15%/-20%/-25%
    4. 个股止损 -15%
    """
    df_close = df_close.sort_index()
    start_mask = df_close.index >= START_DATE
    df_close = df_close[start_mask]

    bond_etf = 'sh511010'
    gold_etf = 'sh518880'
    cash_etf = 'sh511880'

    for code in stock_data:
        stock_data[code] = stock_data[code].sort_index()

    allocation = {
        'RISKON':  {'equity': 0.95, 'bond': 0.00, 'gold': 0.00, 'cash': 0.05},
        'NEUTRAL': {'equity': 0.60, 'bond': 0.30, 'gold': 0.05, 'cash': 0.05},
        'RISKOFF': {'equity': 0.30, 'bond': 0.50, 'gold': 0.10, 'cash': 0.10},
        'CRISIS':  {'equity': 0.00, 'bond': 0.65, 'gold': 0.15, 'cash': 0.20},
    }

    risk_limits = RiskLimits(
        # 针对 A 股调参：高波动市场放宽止损，板块上限放宽
        stop_loss_pct=0.25,        # 15%→25%，A股波动大
        max_sector_pct=0.50,       # 40%→50%，减少误过滤
        drawdown_warning=0.15,
        drawdown_halt=0.20,
        drawdown_liquidate=0.25,
    )
    dc = DrawdownController(risk_limits) if enable_risk else None
    entry_prices = {}

    initial_capital = 1_000_000
    cash = initial_capital
    positions = {}
    portfolio_values = []
    rebalance_dates = []

    stop_loss_events = []
    drawdown_events = []
    sector_skip_events = []

    monthly_dates = df_close.groupby(df_close.index.to_period('M')).apply(lambda x: x.index[-1])
    monthly_dates = pd.DatetimeIndex(monthly_dates)

    if verbose:
        label = "🛡️ 风控增强版" if enable_risk else "📊 基线"
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"  {df_close.index[0].date()} ~ {df_close.index[-1].date()}")
        print(f"  股票池: {len(stock_data)} 只, Top-N: {top_n}")
        print(f"{'='*55}")

    warmup_days = 252

    for i, date in enumerate(df_close.index):
        # 计算组合市值
        position_value = 0
        for sym, shares in positions.items():
            price = _get_price(sym, date, stock_data, df_close)
            if price is not None and price > 0:
                position_value += shares * price

        portfolio_value = cash + position_value
        portfolio_values.append({"date": date, "value": portfolio_value})

        # === 每日止损检查 ===
        if enable_risk and dc is not None:
            dc.update(portfolio_value)
            to_stop = []
            for sym, shares in list(positions.items()):
                if sym in entry_prices and sym not in [bond_etf, gold_etf, cash_etf]:
                    price = _get_price(sym, date, stock_data, df_close)
                    if price is not None and price > 0:
                        loss = (price - entry_prices[sym]) / entry_prices[sym]
                        if loss <= -risk_limits.stop_loss_pct:
                            to_stop.append((sym, price, loss))
            for sym, price, loss in to_stop:
                if sym in positions:
                    cash += positions[sym] * price * (1 - 0.00125)
                    stop_loss_events.append({
                        'date': date, 'code': sym,
                        'entry': round(entry_prices[sym], 2),
                        'exit': round(price, 2),
                        'loss': round(loss * 100, 1)
                    })
                    del positions[sym]
                    del entry_prices[sym]

        # === 月度调仓 ===
        if date in monthly_dates and i >= warmup_days:
            rebalance_dates.append(date)

            # 获取制度
            if regime_series is not None and date in regime_series.index:
                s_val = float(regime_series.loc[date])
                if s_val >= 0.70: regime = 'RISKON'
                elif s_val >= 0.50: regime = 'NEUTRAL'
                elif s_val >= 0.30: regime = 'RISKOFF'
                else: regime = 'CRISIS'
            else:
                regime = 'NEUTRAL'

            alloc = allocation[regime]

            # 回撤熔断检查
            drawdown_override = None
            if enable_risk and dc is not None:
                if dc.should_liquidate:
                    drawdown_override = 'LIQUIDATE'
                    drawdown_events.append({
                        'date': str(date.date()), 'drawdown': round(dc.current_drawdown * 100, 1),
                        'action': 'LIQUIDATE'
                    })
                elif dc.position_multiplier == 0.0:
                    drawdown_override = 'HALT'
                    drawdown_events.append({
                        'date': str(date.date()), 'drawdown': round(dc.current_drawdown * 100, 1),
                        'action': 'HALT'
                    })

            # 选股
            selected_stocks = []
            if alloc['equity'] > 0 and stock_data and drawdown_override != 'LIQUIDATE':
                if drawdown_override == 'HALT':
                    selected_stocks = [s for s in positions.keys() if s in stock_data]
                else:
                    valid_stocks = {}
                    for code, sdf in stock_data.items():
                        if date in sdf.index and len(sdf[sdf.index <= date]) >= 250:
                            valid_stocks[code] = sdf
                    if valid_stocks:
                        scores = compute_stock_factors(valid_stocks, date)
                        raw_selected = select_top_stocks(scores, max(top_n * 2, 30))
                        if enable_risk:
                            selected_stocks = _sector_filter(
                                raw_selected, top_n, alloc['equity'] / top_n,
                                risk_limits.max_sector_pct, risk_limits.max_position_pct,
                                sector_skip_events, date
                            )
                        else:
                            selected_stocks = raw_selected[:top_n]

            # 构建目标权重
            target_weights = {}
            if drawdown_override == 'LIQUIDATE':
                target_weights[bond_etf] = 0.65
                target_weights[gold_etf] = 0.15
                target_weights[cash_etf] = 0.20
            elif regime == 'CRISIS' or not selected_stocks:
                target_weights[bond_etf] = alloc['bond']
                target_weights[gold_etf] = alloc['gold']
                target_weights[cash_etf] = alloc['cash']
            else:
                n = len(selected_stocks)
                if n > 0:
                    psw = alloc['equity'] / n
                    if enable_risk and dc is not None and dc.position_multiplier == 0.5:
                        psw *= 0.5
                    for code in selected_stocks:
                        target_weights[code] = min(psw, risk_limits.max_position_pct) if enable_risk else psw
                if alloc['bond'] > 0:
                    target_weights[bond_etf] = alloc['bond']
                if alloc['gold'] > 0:
                    target_weights[gold_etf] = alloc['gold']
                if alloc['cash'] > 0:
                    target_weights[cash_etf] = alloc['cash']

            # 执行
            for sym in list(positions.keys()):
                if sym not in target_weights:
                    price = _get_price(sym, date, stock_data, df_close)
                    if price is not None and price > 0:
                        cash += positions[sym] * price * (1 - 0.00125)
                        del positions[sym]
                        entry_prices.pop(sym, None)

            for sym, weight in target_weights.items():
                if weight <= 0:
                    continue
                price = _get_price(sym, date, stock_data, df_close)
                if price is None or price <= 0:
                    continue
                target_value = portfolio_value * weight
                lot_size = 100
                target_shares = int(target_value / price / lot_size) * lot_size
                current_shares = positions.get(sym, 0)
                diff = target_shares - current_shares

                if diff > 0:
                    cost = diff * price * 1.00125
                    if cost <= cash:
                        cash -= cost
                        positions[sym] = target_shares
                        if sym not in entry_prices or current_shares == 0:
                            entry_prices[sym] = price
                elif diff < 0:
                    cash += abs(diff) * price * 0.99875
                    positions[sym] = target_shares
                    if target_shares == 0:
                        entry_prices.pop(sym, None)

    # === 绩效指标 ===
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

    result = {
        "metrics": {
            "annual_return": round(ann_ret * 100, 1),
            "max_drawdown": round(max_dd * 100, 1),
            "sharpe_ratio": round(sharpe, 2),
            "total_return": round(total_ret * 100, 1),
            "volatility": round(rets.std() * np.sqrt(252) * 100, 1),
        },
        "annual_returns": dict(annual_returns.round(1)),
    }

    if enable_risk:
        result["risk_stats"] = {
            "stop_loss_count": len(stop_loss_events),
            "drawdown_event_count": len(drawdown_events),
            "sector_filter_count": len(sector_skip_events),
            "drawdown_events": drawdown_events,
            "stop_loss_samples": stop_loss_events[:10],
        }

    return result


def _sector_filter(raw, top_n, per_w, max_sector, max_single, events, date):
    selected = []
    counts = {}
    for code in raw:
        if len(selected) >= top_n:
            break
        sec = classify_sector(code)
        c = counts.get(sec, 0)
        sw = (c + 1) * per_w
        if per_w > max_single:
            continue
        if sw > max_sector:
            events.append({'date': date, 'code': code, 'sector': sec})
            continue
        selected.append(code)
        counts[sec] = c + 1
    return selected


def _get_price(sym, date, stock_data, df_close):
    if sym in stock_data:
        sdf = stock_data[sym]
        if date in sdf.index:
            return float(sdf.loc[date, 'close'])
        prev = sdf[sdf.index < date]
        if len(prev) > 0:
            return float(prev.iloc[-1]['close'])
    elif sym in df_close.columns:
        if date in df_close.index:
            val = df_close.loc[date, sym]
            if not pd.isna(val):
                return float(val)
    return None
