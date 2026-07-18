# -*- coding: utf-8 -*-
"""
可转债日内研究 - bar 级回测引擎 (v1)

设计原则(吸取隔夜策略研究的教训):
  1. 无前视: 信号在 bar t 收盘计算, 成交在 bar t+1 开盘价
  2. 全部平仓离场: v1 只做纯日内 T+0(收盘强平, 不留隔夜敞口), 隔夜是另一个问题
  3. 成本诚实: 每笔计双边成本, 结论默认三档成本对照
  4. 容量意识: 每笔按入场 bar 成交额记录 participation, 供容量估算

约定输入 bars 长表: [symbol, datetime, open, high, low, close, volume, amount]
信号函数 signal_fn(bars) -> DataFrame[symbol, datetime, strength]
  datetime = 信号产生的 bar(引擎自动移到下一根 bar 开盘成交)
"""
import numpy as np
import pandas as pd

from .costs import CostModel


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["date"] = df["datetime"].dt.date
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    # 每券每日的 bar 序号与"下一根bar"信息
    g = df.groupby("symbol", group_keys=False)
    df["next_open"] = g["open"].shift(-1)
    df["next_dt"] = g["datetime"].shift(-1)
    # 当日最后一根 bar 的收盘价(日内强平价)
    day_close = df.groupby(["symbol", "date"])["close"].transform("last")
    df["day_close"] = day_close
    last_dt = df.groupby(["symbol", "date"])["datetime"].transform("max")
    df["is_last_bar"] = df["datetime"] == last_dt
    return df


def run_backtest(bars: pd.DataFrame, signal_fn, cost: CostModel,
                 max_positions: int = 5) -> dict:
    """
    信号 bar 收盘 → 下一根 bar 开盘买入 → 当日最后一根 bar 收盘卖出。
    仓位: 当日信号按 strength 降序取前 max_positions 只, 等权满仓分摊。
    返回 {trades, daily, stats}
    """
    df = _prepare(bars)
    sig = signal_fn(bars).dropna(subset=["strength"])
    if sig.empty:
        return {"trades": pd.DataFrame(), "daily": pd.DataFrame(), "stats": {}}

    m = sig.merge(df, on=["symbol", "datetime"], how="inner")
    # 信号 bar 已是当日最后一根 → 无法日内完成买卖, 丢弃; 跨日的 next bar 同样丢弃
    m = m[~m["is_last_bar"]]
    m = m[pd.to_datetime(m["next_dt"]).dt.date == m["date"]]
    m = m.dropna(subset=["next_open", "day_close"])
    if m.empty:
        return {"trades": pd.DataFrame(), "daily": pd.DataFrame(), "stats": {}}

    # 每日按 strength 取前 N
    m = m.sort_values(["date", "strength"], ascending=[True, False])
    m["rank"] = m.groupby("date").cumcount()
    taken = m[m["rank"] < max_positions].copy()

    taken["entry_price"] = taken["next_open"]
    taken["exit_price"] = taken["day_close"]
    taken["gross_ret"] = taken["exit_price"] / taken["entry_price"] - 1.0
    taken["net_ret"] = taken["gross_ret"] - cost.round_trip
    taken["entry_amount"] = taken["amount"]  # 信号 bar 成交额, 容量参考

    trades = taken[["date", "symbol", "datetime", "strength", "entry_price",
                    "exit_price", "gross_ret", "net_ret", "entry_amount"]].reset_index(drop=True)

    # 组合日收益: 当日等权(未触发日收益为0, 资金闲置)
    daily = trades.groupby("date")["net_ret"].mean().rename("ret").to_frame()
    all_days = pd.Series(sorted(df["date"].unique()), name="date")
    daily = daily.reindex(all_days).fillna(0.0)
    daily["nav"] = (1 + daily["ret"]).cumprod()

    stats = _stats(trades, daily)
    return {"trades": trades, "daily": daily, "stats": stats}


def _stats(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    n_days = len(daily)
    years = max(n_days / 244.0, 1e-9)
    nav_end = daily["nav"].iloc[-1]
    ann = nav_end ** (1 / years) - 1 if nav_end > 0 else -1.0
    dd = (daily["nav"] / daily["nav"].cummax() - 1).min()
    active = daily[daily["ret"] != 0]
    sharpe = np.nan
    if daily["ret"].std() > 0:
        sharpe = daily["ret"].mean() / daily["ret"].std() * np.sqrt(244)
    return {
        "总交易笔数": len(trades),
        "覆盖交易日": n_days,
        "触发日占比": round(len(active) / n_days, 3) if n_days else 0,
        "每笔均值(费后)": round(trades["net_ret"].mean(), 5) if len(trades) else np.nan,
        "每笔胜率(费后)": round((trades["net_ret"] > 0).mean(), 3) if len(trades) else np.nan,
        "累计净值": round(nav_end, 4),
        "年化收益(费后)": round(ann, 4),
        "最大回撤": round(dd, 4),
        "夏普(日频)": round(sharpe, 2) if sharpe == sharpe else None,
    }


def cost_sensitivity(bars: pd.DataFrame, signal_fn, scenarios: dict,
                     max_positions: int = 5) -> pd.DataFrame:
    """同一信号在多档成本下的关键指标对照"""
    rows = []
    for name, cm in scenarios.items():
        r = run_backtest(bars, signal_fn, cm, max_positions=max_positions)
        row = {"成本档": name, **r["stats"]}
        rows.append(row)
    return pd.DataFrame(rows)
