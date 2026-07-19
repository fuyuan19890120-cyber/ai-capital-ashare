"""
个股级回测引擎

SMA250制度择时框架 + 多因子个股精选
RISKON/NEUTRAL/RISKOFF时用个股，CRISIS时用ETF防御

2026-07-18 修复(审计 S1/S4/S5/S6, 见 reports/v4_audit_fix_report.md):
  - execution='next_open': T日收盘出信号, T+1开盘成交(旧版按信号日收盘成交不可实现)
  - stamp_duty=True: 个股卖出印花税(2023-08-28前0.1%/后0.05%), ETF不收
  - 涨跌停约束: 开盘涨停不追买、开盘跌停不杀卖(挂起次日重试)
  - ffill_valuation=True: 停牌日持仓按最后可得收盘价估值(旧版计0, 净值虚降)
  旧行为全部保留开关, 用于复现旧数字和逐项归因。
"""
import pandas as pd
import numpy as np
from config import START_DATE
from src.stock_data import compute_stock_factors, select_top_stocks, filter_reversal_stocks

STAMP_DUTY_CUT_DATE = pd.Timestamp("2023-08-28")  # 印花税 0.1% -> 0.05%
CHINEXT_20PCT_DATE = pd.Timestamp("2020-08-24")   # 创业板涨跌幅 10% -> 20%


def _stamp_duty_rate(date):
    return 0.0005 if date >= STAMP_DUTY_CUT_DATE else 0.001


def _price_limit(code, date):
    """个股涨跌幅限制(不含ST细分, 近似)"""
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.20 if date >= CHINEXT_20PCT_DATE else 0.10
    return 0.10


def run_stock_backtest(
    df_close,          # ETF对齐价格（用于防御资产）
    regime_series,     # 制度分 Series
    stock_data,        # {code: DataFrame} 个股数据
    st_filter=None,    # ST股票集合
    top_n=10,          # 选股数量
    verbose=True,
    execution="next_open",   # 'next_open'=T+1开盘成交(修复) | 'same_close'=信号日收盘成交(旧版)
    stamp_duty=True,         # 个股卖出印花税(旧版=False)
    ffill_valuation=True,    # 停牌日按最后收盘价估值(旧版=False, 计0)
    df_open=None,            # ETF开盘价(execution='next_open'时使用, 缺列回退close)
    rebalance_dates=None,    # 自定义调仓日(DatetimeIndex); None=每月最后交易日
    select_fn=None,          # 自定义选股: select_fn(date)->[code,...]; None=旧四因子路径
    equity_scale=None,       # {调仓日: 系数} 条件式波动率目标, 权益仓位乘数(余量转现金ETF)
    downgrade_exec=None,     # {执行日: 目标权益占比} 回撤棘轮: 当日开盘将个股仓位比例降至目标, 停泊现金ETF
    forced_regime=None,      # {调仓日: 档位} 覆盖制度判断(广度SMA30锁定用, 如强制'RISKON')
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
    if df_open is not None:
        df_open = df_open.sort_index().reindex(df_close.index)

    bond_etf = 'sh511010'
    gold_etf = 'sh518880'
    cash_etf = 'sh511880'

    # Ensure stock data is sorted
    for code in stock_data:
        stock_data[code] = stock_data[code].sort_index()

    # 价格面板(一次构建): 估值用 ffill 收盘, 成交用当日 open(不填充, 缺=停牌)
    cal = df_close.index
    close_panel = pd.DataFrame({c: sdf['close'] for c, sdf in stock_data.items()}).reindex(cal)
    open_panel = pd.DataFrame({c: sdf['open'] for c, sdf in stock_data.items()}).reindex(cal)
    close_ffill = close_panel.ffill()
    prev_close_ffill = close_ffill.shift(1)  # 涨跌停判断的前收(停牌跨越取最后成交价)

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
    rebalance_log = []  # 实际发生的调仓日(勿与同名参数混淆)
    skipped_trades = {"buy_limit_up": 0, "buy_suspended": 0, "sell_deferred": 0}
    turnover_value = 0.0  # 累计成交额(双边), 供换手率审计

    pending_rebalance = None  # (target_weights, base_value) 待次日开盘执行
    pending_sells = set()     # 停牌/跌停未卖出, 逐日重试

    if rebalance_dates is None:
        monthly_dates = df_close.groupby(df_close.index.to_period('M')).apply(lambda x: x.index[-1])
        monthly_dates = pd.DatetimeIndex(monthly_dates)
    else:
        monthly_dates = pd.DatetimeIndex([d for d in rebalance_dates if d in df_close.index])
    equity_scale = equity_scale or {}
    downgrade_exec = downgrade_exec or {}
    forced_regime = forced_regime or {}

    if verbose:
        print(f"Stock Backtest: {df_close.index[0].date()} ~ {df_close.index[-1].date()}")
        print(f"Stocks available: {len(stock_data)}, Top-N: {top_n}")
        print(f"Rebalance months: {len(monthly_dates)}, execution={execution}, "
              f"stamp_duty={stamp_duty}, ffill_valuation={ffill_valuation}")

    warmup_days = 252

    def is_stock(sym):
        return sym in stock_data

    def exec_price(sym, date, use_open):
        """成交价: 个股取面板价(停牌=NaN); ETF取df_open/df_close"""
        if is_stock(sym):
            p = (open_panel if use_open else close_panel).at[date, sym]
            return float(p) if pd.notna(p) else None
        src = df_open if (use_open and df_open is not None and sym in getattr(df_open, 'columns', [])) else df_close
        if sym in src.columns:
            v = src.at[date, sym]
            if pd.notna(v):
                return float(v)
            if use_open and sym in df_close.columns:  # ETF缺open回退close
                v = df_close.at[date, sym]
                return float(v) if pd.notna(v) else None
        return None

    def do_sell(sym, shares, price, date):
        nonlocal cash, turnover_value
        proceeds = shares * price * (1 - 0.00125)  # 0.025% comm + 0.1% slip
        if stamp_duty and is_stock(sym):
            proceeds -= shares * price * _stamp_duty_rate(date)
        cash += proceeds
        turnover_value += shares * price

    def try_sell_all(sym, date, use_open):
        """全仓卖出; 停牌/开盘跌停则挂起。返回是否成交"""
        nonlocal positions
        price = exec_price(sym, date, use_open)
        if price is None or price <= 0:
            if is_stock(sym):
                pending_sells.add(sym)
                skipped_trades["sell_deferred"] += 1
                return False
            return False
        if use_open and is_stock(sym):
            prev = prev_close_ffill.at[date, sym]
            if pd.notna(prev) and prev > 0:
                lim = _price_limit(sym, date)
                if price / float(prev) - 1 <= -(lim - 0.002):  # 开盘贴跌停, 视为卖不出
                    pending_sells.add(sym)
                    skipped_trades["sell_deferred"] += 1
                    return False
        do_sell(sym, positions[sym], price, date)
        del positions[sym]
        pending_sells.discard(sym)
        return True

    def execute_rebalance(target_weights, base_value, date, use_open):
        """按目标权重调仓。use_open=True 时应用涨跌停/停牌约束"""
        nonlocal cash, positions, turnover_value
        # 重新入选的目标从挂起卖单中撤销, 避免次日被误卖
        pending_sells.difference_update({s for s, w in target_weights.items() if w > 0})
        # 先卖出非目标
        for sym in list(positions.keys()):
            if sym not in target_weights or target_weights[sym] == 0:
                try_sell_all(sym, date, use_open)

        # 买入/调整目标
        for sym, weight in target_weights.items():
            if weight <= 0:
                continue
            price = exec_price(sym, date, use_open)
            if price is None or price <= 0:
                if is_stock(sym):
                    skipped_trades["buy_suspended"] += 1
                continue

            target_value = base_value * weight
            lot_size = 100
            target_shares = int(target_value / price / lot_size) * lot_size
            current_shares = positions.get(sym, 0)
            diff = target_shares - current_shares

            if diff > 0:
                if use_open and is_stock(sym):
                    prev = prev_close_ffill.at[date, sym]
                    if pd.notna(prev) and prev > 0:
                        lim = _price_limit(sym, date)
                        if price / float(prev) - 1 >= (lim - 0.002):  # 开盘贴涨停, 不追买
                            skipped_trades["buy_limit_up"] += 1
                            continue
                cost = diff * price * 1.00125  # 0.025% comm + 0.1% slip
                if cost <= cash:
                    cash -= cost
                    turnover_value += diff * price
                    positions[sym] = target_shares
            elif diff < 0:
                if use_open and is_stock(sym):
                    prev = prev_close_ffill.at[date, sym]
                    if pd.notna(prev) and prev > 0:
                        lim = _price_limit(sym, date)
                        if price / float(prev) - 1 <= -(lim - 0.002):  # 开盘贴跌停, 减仓同样卖不出
                            skipped_trades["sell_deferred"] += 1
                            continue
                do_sell(sym, abs(diff), price, date)
                positions[sym] = target_shares

    for i, date in enumerate(df_close.index):
        # === T+1 开盘: 执行挂起的调仓与未完成卖出 ===
        if execution == "next_open":
            if pending_rebalance is not None:
                tw, base_value = pending_rebalance
                execute_rebalance(tw, base_value, date, use_open=True)
                pending_rebalance = None
            elif pending_sells:
                for sym in list(pending_sells):
                    if sym in positions:
                        try_sell_all(sym, date, use_open=True)
                    else:
                        pending_sells.discard(sym)

        # === 回撤棘轮: 触发日次日开盘将个股仓位降至目标占比, 停泊现金ETF ===
        if date in downgrade_exec and pending_rebalance is None:
            target_frac = downgrade_exec[date]
            eq_val, port_val = 0.0, cash
            for sym, shares in positions.items():
                if is_stock(sym):
                    p = prev_close_ffill.at[date, sym]  # 用前收测算, 避免偷看当日收盘(复审 C5)
                else:
                    s = df_close[sym].loc[:date].dropna() if sym in df_close.columns else None
                    if s is not None and len(s) and s.index[-1] == date:
                        s = s.iloc[:-1]
                    p = s.iloc[-1] if s is not None and len(s) else None
                if p is not None and pd.notna(p):
                    v = shares * float(p)
                    port_val += v
                    if is_stock(sym):
                        eq_val += v
            if port_val > 0 and eq_val / port_val > target_frac + 0.02:
                sell_ratio = 1.0 - (target_frac * port_val) / eq_val
                freed = 0.0
                for sym in [s for s in positions if is_stock(s)]:
                    px = exec_price(sym, date, use_open=True)
                    if px is None or px <= 0:
                        continue  # 停牌卖不出, 保守: 不强平
                    prev = prev_close_ffill.at[date, sym]
                    if pd.notna(prev) and prev > 0:
                        lim = _price_limit(sym, date)
                        if px / float(prev) - 1 <= -(lim - 0.002):
                            continue  # 开盘贴跌停
                    lot = int(positions[sym] * sell_ratio / 100) * 100
                    if lot >= 100:
                        cash_before = cash
                        do_sell(sym, lot, px, date)
                        freed += cash - cash_before
                        positions[sym] -= lot
                        if positions[sym] == 0:
                            del positions[sym]
                # 释放资金停泊现金ETF
                pk = exec_price(cash_etf, date, use_open=True)
                if pk and pk > 0 and freed > 0:
                    lots = int(freed / pk / 100) * 100
                    if lots > 0:
                        cost = lots * pk * 1.00125
                        if cost <= cash:
                            cash -= cost
                            turnover_value += lots * pk
                            positions[cash_etf] = positions.get(cash_etf, 0) + lots

        today_prices = df_close.loc[date]

        # === 收盘估值 ===
        position_value = 0
        for sym, shares in positions.items():
            if sym in stock_data:
                if ffill_valuation:
                    p = close_ffill.at[date, sym]  # 停牌按最后收盘价估值
                    if pd.notna(p):
                        position_value += shares * float(p)
                else:  # 旧版行为: 停牌日计0
                    p = close_panel.at[date, sym]
                    if pd.notna(p):
                        position_value += shares * float(p)
            elif sym in today_prices and not pd.isna(today_prices[sym]):
                position_value += shares * today_prices[sym]

        portfolio_value = cash + position_value
        portfolio_values.append({"date": date, "value": portfolio_value})

        # === 月末收盘: 出信号 ===
        if date in monthly_dates and i >= warmup_days:
            rebalance_log.append(date)

            # Get regime
            if regime_series is not None and date in regime_series.index:
                s_val = float(regime_series.loc[date])
                if s_val >= 0.70: regime = 'RISKON'
                elif s_val >= 0.50: regime = 'NEUTRAL'
                elif s_val >= 0.30: regime = 'RISKOFF'
                else: regime = 'CRISIS'
            else:
                regime = 'NEUTRAL'
            if date in forced_regime:  # 广度SMA30锁定等覆盖机制
                regime = forced_regime[date]

            if verbose and len(rebalance_log) <= 3:
                print(f"  {date.date()}: regime={regime}")

            alloc = allocation[regime]
            # 条件式波动率目标: 高波动五分位时缩权益, 余量转现金ETF
            es = float(equity_scale.get(date, 1.0))
            eq_w = alloc['equity'] * min(es, 1.0)
            extra_cash = alloc['equity'] - eq_w

            # === Stock Selection (during RISKON/NEUTRAL/RISKOFF) ===
            selected_stocks = []
            if eq_w > 0 and stock_data:
                if select_fn is not None:
                    try:
                        selected_stocks = select_fn(date, regime)
                    except TypeError:
                        selected_stocks = select_fn(date)  # legacy select_fn(date)
                else:
                    # Filter ST stocks
                    valid_stocks = {}
                    for code, sdf in stock_data.items():
                        if st_filter and code in st_filter:
                            continue
                        if date in sdf.index and len(sdf[sdf.index <= date]) >= 250:
                            valid_stocks[code] = sdf
                    # V4.2 反转过滤: 剔除近20日涨幅前25%的候选(A股反转效应)
                    if valid_stocks:
                        valid_stocks = filter_reversal_stocks(valid_stocks, date)
                    if valid_stocks:
                        scores = compute_stock_factors(valid_stocks, date)
                        selected_stocks = select_top_stocks(scores, top_n)

            # === Build Target Weights ===
            target_weights = {}
            if regime == 'CRISIS' or not selected_stocks:
                target_weights[bond_etf] = alloc['bond']
                target_weights[gold_etf] = alloc['gold']
                target_weights[cash_etf] = alloc['cash'] + (alloc['equity'] if regime != 'CRISIS' else 0)
            else:
                per_stock_weight = eq_w / len(selected_stocks)
                for code in selected_stocks:
                    target_weights[code] = per_stock_weight
                if alloc['bond'] > 0:
                    target_weights[bond_etf] = alloc['bond']
                if alloc['gold'] > 0:
                    target_weights[gold_etf] = alloc['gold']
                if alloc['cash'] + extra_cash > 0:
                    target_weights[cash_etf] = alloc['cash'] + extra_cash

            # === Execute ===
            if execution == "next_open":
                pending_rebalance = (target_weights, portfolio_value)  # 次日开盘执行
            else:  # 'same_close': 旧版行为, 信号日收盘价成交(不可实现, 仅供对照)
                execute_rebalance(target_weights, portfolio_value, date, use_open=False)

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
    avg_value = df_values["value"].mean()
    annual_turnover = turnover_value / avg_value / max(n_years, 1e-9)

    return {
        "values": df_values,
        "metrics": {
            "annual_return": ann_ret * 100,
            "max_drawdown": max_dd * 100,
            "sharpe_ratio": sharpe,
            "total_return": total_ret * 100,
            "volatility": rets.std() * np.sqrt(252) * 100,
            "annual_turnover_x": annual_turnover,
            "skipped_trades": dict(skipped_trades),
            "annual_returns": annual_returns,
        },
        "rebalance_dates": rebalance_log,
    }
