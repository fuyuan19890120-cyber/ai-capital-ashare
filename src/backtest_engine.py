"""
轻量回测引擎 v2 — 纯制度择时（跳过动量选股）

隔离测试证明：A 股动量选股是负贡献，制度择时本身有效。
策略逻辑：
  RISKON  → 100% 权益（沪深300 + 中证500 + 创业板，等权）
  NEUTRAL → 55% 权益 + 35% 债券 + 10% 黄金
  RISKOFF → 15% 权益 + 55% 债券 + 15% 黄金 + 15% 货币
  CRISIS  → 0% 权益 + 80% 债券 + 10% 黄金 + 10% 货币
"""
import pandas as pd
import numpy as np
from config import START_DATE, VOL_TARGET, MAX_SINGLE_POSITION, VOL_LOOKBACK


def run_regime_backtest(df_close, regime_series, verbose=True, momentum_weight_equity=False):
    """
    纯制度择时回测

    参数:
        df_close: 对齐后的收盘价 DataFrame
        regime_series: 制度分数 Series, index=date
        verbose: 是否打印进度

    返回:
        dict: 含 values, metrics, regime_history
    """
    # 确保日期排序
    df_close = df_close.sort_index()
    start_mask = df_close.index >= START_DATE
    df_close = df_close[start_mask]

    # 定义固定资产映射
    equity_etfs = ['sh510300', 'sh510500', 'sz159915']  # 沪深300, 中证500, 创业板
    bond_etf = 'sh511010'    # 国债ETF
    gold_etf = 'sh518880'    # 黄金ETF
    cash_etf = 'sh511880'    # 货币ETF

    # 制度 → 资产权重
    allocation = {
        'RISKON':  {e: 1.0/3 for e in equity_etfs},  # 等权权益
        'NEUTRAL': {**{e: 0.55/3 for e in equity_etfs}, bond_etf: 0.30, gold_etf: 0.10, cash_etf: 0.05},
        'RISKOFF': {**{e: 0.15/3 for e in equity_etfs}, bond_etf: 0.45, gold_etf: 0.15, cash_etf: 0.15},
        'CRISIS':  {bond_etf: 0.70, gold_etf: 0.15, cash_etf: 0.15},
    }

    initial_capital = 1_000_000
    current_positions = {}
    cash = initial_capital
    portfolio_values = []
    rebalance_dates = []

    # 获取每月最后一个交易日
    monthly_dates = df_close.groupby(df_close.index.to_period('M')).apply(lambda x: x.index[-1])
    monthly_dates = pd.DatetimeIndex(monthly_dates)

    if verbose:
        print(f"Regime Backtest: {df_close.index[0].date()} ~ {df_close.index[-1].date()}")
        print(f"Trading days: {len(df_close)}, Rebalance months: {len(monthly_dates)}")

    warmup_days = 252

    for i, date in enumerate(df_close.index):
        today_prices = df_close.loc[date]

        # 计算当日组合市值
        position_value = 0
        for sym, shares in current_positions.items():
            if sym in today_prices and not pd.isna(today_prices[sym]):
                position_value += shares * today_prices[sym]
        portfolio_value = cash + position_value
        portfolio_values.append({"date": date, "value": portfolio_value})

        # 月度调仓
        if date in monthly_dates and i >= warmup_days:
            rebalance_dates.append(date)

            # 获取当前制度
            if regime_series is not None and date in regime_series.index:
                score = regime_series.loc[date]
                if isinstance(score, pd.Series):
                    score = score.iloc[0]
                regime = _classify_regime(float(score))
            else:
                regime = 'NEUTRAL'

            if verbose and len(rebalance_dates) <= 3:
                print(f"  {date.date()}: regime={regime} (score={score:.2f})")

            target_weights = allocation[regime].copy()

            # ===== 权益内部权重分配 =====
            equity_assets = [e for e in equity_etfs if e in target_weights and target_weights[e] > 0]
            if equity_assets and len(equity_assets) > 1:
                total_equity_weight = sum(target_weights[e] for e in equity_assets)

                if momentum_weight_equity:
                    # 成长股动量加权：3个月动量越强，配比越高
                    hist = df_close[equity_assets].loc[:date].iloc[-63:]
                    rets = (hist.iloc[-1] / hist.iloc[0] - 1).clip(lower=0)
                    if rets.sum() > 0:
                        eq_w = rets / rets.sum()
                    else:
                        eq_w = pd.Series(1.0/len(equity_assets), index=equity_assets)
                else:
                    # 等权
                    eq_w = pd.Series(1.0/len(equity_assets), index=equity_assets)

                for e in equity_assets:
                    target_weights[e] = total_equity_weight * eq_w.get(e, 1.0/len(equity_assets))

            # ===== 波动率目标：仅在非 RISKON 时缩放权益敞口 =====
            # RISKON 时牛市波动率高是正常的，不限制
            if regime != 'RISKON':
                all_assets = [a for a in target_weights if target_weights[a] > 0 and a in df_close.columns]
                if len(all_assets) >= 2:
                    hist_all = df_close[all_assets].pct_change().dropna().iloc[-63:]
                    if len(hist_all) > 20:
                        target_w = pd.Series({a: target_weights.get(a, 0) for a in all_assets})
                        cov = hist_all.cov()
                        port_vol = np.sqrt(target_w @ cov @ target_w) * np.sqrt(252)

                        target_vol = VOL_TARGET
                        if port_vol > target_vol:
                            scale = target_vol / port_vol
                            for e in equity_etfs:
                                if e in target_weights:
                                    target_weights[e] *= scale
                            freed = 1.0 - sum(target_weights.values())
                            if cash_etf in target_weights:
                                target_weights[cash_etf] += freed
                            else:
                                target_weights[cash_etf] = freed

            # ===== 单一资产上限 =====
            max_single = MAX_SINGLE_POSITION
            for sym in list(target_weights.keys()):
                if target_weights[sym] > max_single:
                    excess = target_weights[sym] - max_single
                    target_weights[sym] = max_single
                    if cash_etf in target_weights:
                        target_weights[cash_etf] += excess
                    else:
                        target_weights[cash_etf] = excess

            # ===== 执行调仓 =====
            for sym in list(current_positions.keys()):
                if sym not in target_weights or target_weights.get(sym, 0) == 0:
                    if sym in today_prices and not pd.isna(today_prices[sym]):
                        shares = current_positions[sym]
                        price = today_prices[sym]
                        cash += shares * price * (1 - 0.00125)  # 0.025% comm + 0.1% slip
                        del current_positions[sym]

            # 调整到目标权重
            for sym, weight in target_weights.items():
                if sym not in today_prices or pd.isna(today_prices[sym]):
                    continue
                target_value = portfolio_value * weight
                price = today_prices[sym]
                target_shares = int(target_value / price / 100) * 100
                current_shares = current_positions.get(sym, 0)
                diff = target_shares - current_shares
                if diff > 0:
                    cost = diff * price * 1.00125
                    if cost <= cash:
                        cash -= cost
                        current_positions[sym] = target_shares
                elif diff < 0:
                    cash += abs(diff) * price * 0.99875
                    current_positions[sym] = target_shares

    # 计算指标
    df_values = pd.DataFrame(portfolio_values).set_index("date")
    df_values["return"] = df_values["value"].pct_change()

    metrics = _compute_metrics(df_values, initial_capital)

    return {
        "values": df_values,
        "metrics": metrics,
        "rebalance_dates": rebalance_dates,
    }


def _classify_regime(score):
    if score >= 0.70:
        return "RISKON"
    elif score >= 0.50:
        return "NEUTRAL"
    elif score >= 0.30:
        return "RISKOFF"
    else:
        return "CRISIS"


def _compute_metrics(df_values, initial_capital):
    returns = df_values["return"].dropna()
    if returns.empty:
        return {}

    total_return = df_values["value"].iloc[-1] / initial_capital - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    excess = returns - 0.02 / 252
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0

    annual_returns = returns.groupby(returns.index.year).apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    return {
        "total_return": total_return * 100,
        "annual_return": annual_return * 100,
        "max_drawdown": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "volatility": returns.std() * np.sqrt(252) * 100,
        "annual_returns": annual_returns,
    }
