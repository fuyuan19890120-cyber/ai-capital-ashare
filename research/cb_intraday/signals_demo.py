# -*- coding: utf-8 -*-
"""
可转债日内研究 - 演示策略(验证全链路, 不代表可用 alpha)

三个最朴素的 60 分钟基线信号, 用途是:
  1. 验证 采集→存储→引擎→成本敏感性 全链路跑通
  2. 给出"朴素信号在三档成本下的真实水平"作为后续研究的基准线
60 分钟 bar 结构: 10:30 / 11:30 / 14:00 / 15:00 四根
"""
import pandas as pd

MIN_FIRST_HOUR_AMOUNT = 20_000_000  # 首小时成交额过滤: 2000万, 保证可交易性


def _first_bar(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["date"] = df["datetime"].dt.date
    first_dt = df.groupby(["symbol", "date"])["datetime"].transform("min")
    return df[df["datetime"] == first_dt].copy()


def momentum_first_hour(bars: pd.DataFrame) -> pd.DataFrame:
    """首小时动量: 首根bar涨幅(相对开盘)最大且成交额达标 → 追涨, 收盘卖"""
    fb = _first_bar(bars)
    fb = fb[fb["amount"] >= MIN_FIRST_HOUR_AMOUNT]
    fb["strength"] = fb["close"] / fb["open"] - 1.0
    fb = fb[fb["strength"] > 0.005]  # 首小时涨幅>0.5%才算"有动量"
    return fb[["symbol", "datetime", "strength"]]


def reversal_first_hour(bars: pd.DataFrame) -> pd.DataFrame:
    """首小时反转: 首根bar跌幅最深且成交额达标 → 抄底, 收盘卖"""
    fb = _first_bar(bars)
    fb = fb[fb["amount"] >= MIN_FIRST_HOUR_AMOUNT]
    ret = fb["close"] / fb["open"] - 1.0
    fb = fb[ret < -0.005]
    fb["strength"] = -ret  # 跌得越深 strength 越大
    return fb[["symbol", "datetime", "strength"]]


def afternoon_momentum(bars: pd.DataFrame) -> pd.DataFrame:
    """尾盘动量确认: 至14:00累计涨幅最大 → 14:00入场, 持最后1小时"""
    df = bars.copy()
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    bar14 = df[df["time"] == "14:00"].copy()
    day_open = df.groupby(["symbol", "date"])["open"].transform("first")
    df["day_open"] = day_open
    bar14 = bar14.merge(
        df[["symbol", "date", "day_open"]].drop_duplicates(),
        on=["symbol", "date"], how="left")
    # 当日至14:00累计成交额过滤
    cum_amt = (df[df["time"].isin(["10:30", "11:30", "14:00"])]
               .groupby(["symbol", "date"])["amount"].sum().rename("cum_amount").reset_index())
    bar14 = bar14.merge(cum_amt, on=["symbol", "date"], how="left")
    bar14 = bar14[bar14["cum_amount"] >= MIN_FIRST_HOUR_AMOUNT * 2]
    bar14["strength"] = bar14["close"] / bar14["day_open"] - 1.0
    bar14 = bar14[bar14["strength"] > 0.01]
    return bar14[["symbol", "datetime", "strength"]]


DEMO_STRATEGIES = {
    "首小时动量追涨": momentum_first_hour,
    "首小时反转抄底": reversal_first_hour,
    "尾盘动量(持最后1小时)": afternoon_momentum,
}
