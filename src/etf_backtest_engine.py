"""
ETF 专用回测引擎 (R4)

核心改进 vs stock_backtest.py:
  - 每日信号总线: 支持月中 SURGE 触发、KDJ 中途出场等非月度信号
  - 部分调仓: 只交易权重有变化的 ETF, 其他持仓不动
  - 信号优先级: 风控出场 > SURGE > KDJ出场 > 月度常规调仓

架构:
  每日循环 → 执行挂起调仓(T+1开盘) → 检查每日信号 → 盯市估值
  月末额外: regime检测 → 因子选股 → 生成目标权重

参考: backtest_combo_v2.py 的「方向2: KDJ中途下车」已验证每日KDJ检查+定向卖出的可行性
"""
import pandas as pd
import numpy as np

ETF_COST = 0.0006  # 单边: 万0.5佣金 + 0.05%滑点
LOT_SIZE = 100     # A股ETF 100股整数倍


def run_etf_backtest(
    df_close,              # DataFrame: index=date, columns=ETF codes, values=收盘价
    df_open=None,          # DataFrame: 开盘价 (None则用收盘价替代)
    regime_series=None,    # Series: index=date, values=regime分数 [0,1]
    monthly_select_fn=None,  # 月末选股: fn(date, regime_val, context) -> [(code, weight), ...]
    daily_signals=None,    # dict: {name: fn(date, positions, context) -> {code: weight|'exit'}}
    surge_config=None,     # SURGE配置 dict 或 None
    start_date='2016-01-01',
    end_date='2026-07-31',
    initial_capital=1_000_000,
    cost_rate=ETF_COST,
    verbose=True,
    context=None,          # 用户自定义上下文, 传递给所有信号函数
):
    """
    ETF 回测主引擎

    参数:
        df_close: ETF 收盘价宽表
        df_open: ETF 开盘价宽表 (缺则用收盘价, 但会有偷看偏差)
        regime_series: 制度分序列
        monthly_select_fn: 月末选股函数, 返回 [(code, weight), ...]
        daily_signals: 每日信号函数字典 {name: fn}, 每个返回 {code: action}
        surge_config: SURGE 实时监控配置
        context: 用户自定义上下文 (如ETF池分类、因子计算器等)
    """
    # ── 数据准备 ──
    df_close = df_close.sort_index()
    mask = (df_close.index >= start_date) & (df_close.index <= end_date)
    df_close = df_close[mask].copy()

    if df_open is not None:
        df_open = df_open.sort_index().reindex(df_close.index)
    else:
        df_open = df_close.copy()

    all_dates = df_close.index
    n_days = len(all_dates)

    if n_days < 252:
        return {"values": pd.DataFrame(), "metrics": {}, "error": "Insufficient data (<252 days)"}

    # ── 调仓日期: 每月最后交易日 ──
    monthly_groups = {}
    for d in all_dates:
        monthly_groups.setdefault((d.year, d.month), []).append(d)
    rebalance_dates = set(max(dates) for dates in monthly_groups.values())
    warmup_cutoff = all_dates[min(252, len(all_dates) - 1)]

    # ── 状态变量 ──
    cash = float(initial_capital)
    positions = {}           # {code: shares}
    pending_rebalance = None # (target_weights_dict, base_value)
    pending_exits = set()    # 待卖出的code

    # SURGE 状态
    surge_state = {"active": False, "days_remaining": 0, "peak_value": None}
    if surge_config is None:
        surge_config = {"enabled": False}

    # ── 日志 ──
    portfolio_values = []
    rebalance_log = []
    trade_log = []
    signal_log = []
    surge_events = []

    # ── 内部函数 ──
    def get_price(code, date, use_open=False):
        src = df_open if use_open else df_close
        if code in src.columns:
            val = src.at[date, code]
            if pd.notna(val) and val > 0:
                return float(val)
        return None

    def mark_to_market(date):
        total = cash
        for code, shares in positions.items():
            px = get_price(code, date, use_open=False)
            if px is not None:
                total += shares * px
        return total

    def execute_sell(code, shares, price, date):
        nonlocal cash
        if shares <= 0:
            return
        proceeds = shares * price * (1 - cost_rate)
        cash += proceeds
        trade_log.append({"date": date, "code": code, "action": "SELL",
                          "shares": int(shares), "price": round(price, 4)})

    def execute_buy(code, shares, price, date):
        nonlocal cash
        if shares <= 0:
            return 0
        cost = shares * price * (1 + cost_rate)
        if cost <= cash:
            cash -= cost
            trade_log.append({"date": date, "code": code, "action": "BUY",
                              "shares": int(shares), "price": round(price, 4)})
            return shares
        else:
            affordable = int(cash / (price * (1 + cost_rate)) / LOT_SIZE) * LOT_SIZE
            if affordable >= LOT_SIZE:
                cost = affordable * price * (1 + cost_rate)
                cash -= cost
                trade_log.append({"date": date, "code": code, "action": "BUY",
                                  "shares": int(affordable), "price": round(price, 4)})
                return affordable
        return 0

    def apply_target_weights(target_weights, base_value, date, use_open=True):
        """部分调仓: 只交易权重有变化的ETF"""
        nonlocal cash, positions

        # 1. 卖出不在目标中的
        for code in list(positions.keys()):
            if code not in target_weights or target_weights[code] <= 0:
                px = get_price(code, date, use_open)
                if px is not None:
                    execute_sell(code, positions[code], px, date)
                    del positions[code]
                    pending_exits.discard(code)

        # 2. 买入/调整目标中的
        for code, weight in target_weights.items():
            if weight <= 0:
                continue
            px = get_price(code, date, use_open)
            if px is None:
                continue

            target_value = base_value * weight
            target_shares = int(target_value / px / LOT_SIZE) * LOT_SIZE
            current_shares = positions.get(code, 0)

            if target_shares > current_shares:
                diff = target_shares - current_shares
                bought = execute_buy(code, diff, px, date)
                positions[code] = current_shares + bought
            elif target_shares < current_shares:
                diff = current_shares - target_shares
                execute_sell(code, diff, px, date)
                if target_shares > 0:
                    positions[code] = target_shares
                else:
                    del positions[code]

    # ── 准备 context ──
    if context is None:
        context = {}
    context["df_close"] = df_close
    context["df_open"] = df_open
    context["all_dates"] = all_dates

    # ── 每日循环 ──
    if verbose:
        print(f"ETF Backtest: {all_dates[0].date()} ~ {all_dates[-1].date()}")
        print(f"  ETFs: {len(df_close.columns)},  Cost: {cost_rate:.4f}")
        print(f"  SURGE: {surge_config.get('enabled', False)}")
        print(f"  Daily signals: {list(daily_signals.keys()) if daily_signals else 'none'}")

    for i, date in enumerate(all_dates):
        # ═══ T+1 执行挂起调仓 ═══
        if pending_rebalance is not None:
            tw, base_value = pending_rebalance
            apply_target_weights(tw, base_value, date, use_open=True)
            pending_rebalance = None
        elif pending_exits:
            for code in list(pending_exits):
                if code in positions:
                    px = get_price(code, date, use_open=True)
                    if px is not None:
                        execute_sell(code, positions[code], px, date)
                        del positions[code]
                        pending_exits.discard(code)

        # ═══ 每日信号检查 ═══
        if i >= 252 and date >= warmup_cutoff:
            day_context = {
                **context,
                "date": date,
                "positions": dict(positions),
                "cash": cash,
                "portfolio_value": mark_to_market(date),
                "regime_val": float(regime_series.loc[date]) if (regime_series is not None and date in regime_series.index) else 0.5,
            }

            # A. SURGE 触发检查 (支持 monthly/daily 两种模式)
            surge_mode = surge_config.get("mode", "daily")
            surge_enabled = surge_config.get("enabled", False)
            check_surge_trigger = (
                surge_enabled and daily_signals and "surge" in daily_signals
                and not surge_state.get("active")
                and (surge_mode == "daily" or (surge_mode == "monthly" and date in rebalance_dates))
            )
            if check_surge_trigger:
                surge_actions = daily_signals["surge"](date, positions, day_context)
                if surge_actions:
                    signal_log.append({"date": date, "signal": "SURGE", "actions": surge_actions})
                    portfolio_val = mark_to_market(date)
                    pending_rebalance = (surge_actions, portfolio_val)
                    surge_state = {"active": True, "days_remaining": surge_config.get("lock_days", 14),
                                   "peak_value": portfolio_val, "entry_date": date}
                    surge_events.append({
                        "start_date": date, "end_date": None,
                        "start_value": portfolio_val, "end_value": None, "pnl_pct": None
                    })

            # SURGE 锁仓管理 (每日: 倒计时 + 回撤熔断)
            if surge_enabled and surge_state.get("active"):
                surge_state["days_remaining"] -= 1
                pv = mark_to_market(date)

                if surge_state.get("peak_value") is None or pv > surge_state["peak_value"]:
                    surge_state["peak_value"] = pv

                # 回撤熔断
                peak = surge_state["peak_value"]
                if peak > 0 and (pv - peak) / peak < -surge_config.get("dd_melt", 0.08):
                    surge_state["active"] = False
                    surge_state["days_remaining"] = 0
                    if surge_events:
                        surge_events[-1]["end_date"] = date
                        surge_events[-1]["end_value"] = pv
                        surge_events[-1]["pnl_pct"] = (pv - surge_events[-1]["start_value"]) / surge_events[-1]["start_value"] * 100

                # 锁仓期满
                if surge_state["days_remaining"] <= 0:
                    surge_state["active"] = False
                    if surge_events and surge_events[-1].get("end_date") is None:
                        surge_events[-1]["end_date"] = date
                        surge_events[-1]["end_value"] = pv
                        surge_events[-1]["pnl_pct"] = (pv - surge_events[-1]["start_value"]) / surge_events[-1]["start_value"] * 100

            # B. 其他每日信号 (KDJ 出场等) — SURGE锁仓期间屏蔽
            if daily_signals and not (surge_enabled and surge_state.get("active")):
                for sig_name, sig_fn in daily_signals.items():
                    if sig_name == "surge":
                        continue
                    actions = sig_fn(date, positions, day_context)
                    if actions:
                        signal_log.append({"date": date, "signal": sig_name, "actions": actions})
                        # 所有出场信号统一挂起, T+1开盘执行 (避免前视偏差)
                        for code, action in actions.items():
                            if action in (0, 'exit', 0.0) and code in positions:
                                pending_exits.add(code)

        # ═══ 收盘估值 ═══
        portfolio_value = mark_to_market(date)
        portfolio_values.append({"date": date, "value": portfolio_value})

        # ═══ 月末调仓 ═══
        if date in rebalance_dates and i >= 252 and date >= warmup_cutoff:
            rebalance_log.append(date)

            if regime_series is not None and date in regime_series.index:
                r_val = float(regime_series.loc[date])
                if np.isnan(r_val):
                    r_val = 0.5
            else:
                r_val = 0.5

            # SURGE 锁仓期内跳过常规月末调仓
            if surge_state.get("active") and surge_config.get("enabled"):
                if verbose and len(rebalance_log) <= 5:
                    print(f"  {date.date()}: SURGE lock, skip monthly (remaining {surge_state['days_remaining']}d)")
                continue

            month_context = {**context, "date": date, "regime_val": r_val,
                             "positions": dict(positions), "cash": cash,
                             "portfolio_value": portfolio_value}
            if monthly_select_fn is not None:
                selected = monthly_select_fn(date, r_val, month_context)
            else:
                selected = []

            target_weights = {}
            for item in selected:
                if isinstance(item, tuple):
                    code, weight = item
                else:
                    code, weight = item, 0
                target_weights[code] = weight

            total_w = sum(target_weights.values())
            if total_w == 0 and len(target_weights) > 0:
                eq_w = 1.0 / len(target_weights)
                target_weights = {c: eq_w for c in target_weights}

            if verbose and len(rebalance_log) <= 5:
                top_codes = list(target_weights.keys())[:3]
                print(f"  {date.date()}: regime={r_val:.2f}, n={len(target_weights)}, top3={top_codes}")

            pending_rebalance = (target_weights, portfolio_value)

    # ── 最终估值 ──
    final_value = mark_to_market(all_dates[-1])
    portfolio_values[-1]["value"] = final_value

    # ═══ 指标 ═══
    df_values = pd.DataFrame(portfolio_values).set_index("date")
    df_values["return"] = df_values["value"].pct_change()
    rets = df_values["return"].dropna()

    if len(rets) < 20:
        return {"values": df_values, "metrics": {}, "error": "Insufficient returns data"}

    total_ret = df_values["value"].iloc[-1] / initial_capital - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.expanding().max() - 1).min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    annual_returns = {}
    for yr, grp in rets.groupby(rets.index.year):
        annual_returns[yr] = (1 + grp).prod() - 1

    total_turnover = sum(abs(t["shares"] * t["price"]) for t in trade_log)
    avg_nav = df_values["value"].mean()
    annual_turnover = total_turnover / avg_nav / max(n_years, 1e-9)

    metrics = {
        "total_return": total_ret * 100,
        "annual_return": ann_ret * 100,
        "annual_volatility": ann_vol * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd * 100,
        "calmar_ratio": calmar,
        "n_trades": len(trade_log),
        "annual_turnover": annual_turnover,
        "annual_returns": {yr: v * 100 for yr, v in annual_returns.items()},
        "n_rebalances": len(rebalance_log),
        "n_surge_events": len(surge_events),
    }

    return {
        "values": df_values,
        "metrics": metrics,
        "trade_log": pd.DataFrame(trade_log) if trade_log else pd.DataFrame(),
        "signal_log": pd.DataFrame(signal_log) if signal_log else pd.DataFrame(),
        "surge_events": surge_events,
        "rebalance_dates": rebalance_log,
    }
