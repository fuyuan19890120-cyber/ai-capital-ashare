#!/usr/bin/env python3
"""
日频 SURGE 监控(2026-07-19 上线)

背景: 广度SMA30锁定进场经回测验证(+4.2pp, 复审+扰动敏感性通过), 但实盘管线
此前只在月末检查一次、无锁状态 —— 17 次历史触发只能碰上 2 次。本脚本补上缺口:

每个交易日收盘后运行:
  1. 拉取沪深300指数 + 3只权益ETF(510300/510500/159915)最新日线
  2. 按实盘口径(src/signal_generator._check_breadth_lock 同款公式)计算触发
  3. 锁状态持久化到 signals/surge_state.json:
       {active, trigger_date, expire_date(触发日起第21个交易日), checked_date, detail}
  4. 触发/到期/状态变化时打印显著提示(供 cron 邮件/Actions 日志)

月末调仓管线(run_monthly / signal_generator)读取该文件: active=True 时强制 RISKON。
触发日次日应手动/自动执行进场(与回测口径一致: T收盘信号, T+1开盘成交)。

用法: venv/bin/python surge_monitor.py           # 收盘后跑(建议 15:35 后)
      venv/bin/python surge_monitor.py --dry     # 只打印不写状态
部署: crontab -e 添加
      35 15 * * 1-5 cd ~/ai-capital-ashare && venv/bin/python surge_monitor.py >> signals/surge_monitor.log 2>&1
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(BASE, "signals", "surge_state.json")

# 预注册参数(与回测/实盘口径一致, 勿因单次回测调整; 0.80 严确认版是候选、非默认)
LOOKBACK = 15        # 广度回看交易日
LOCK_DAYS = 21       # 锁定交易日
S30_MIN = 0.70       # SMA30 制度分阈值
BASE_MIN = 0.15      # SMA250 制度分底线
EQ_ETFS = {"sh510300": "510300", "sh510500": "510500", "sz159915": "159915"}


def fetch_daily():
    """指数与ETF日线(近1.5年足够算 SMA250)。ETF: 东财qfq优先(Actions可用), 本机被风控时回退新浪(不复权,
    ETF分红日SMA有~1%级失真, 可接受)。"""
    import akshare as ak
    idx = ak.stock_zh_index_daily(symbol="sh000300")
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"].sort_index()
    etfs = {}
    for sym, code in EQ_ETFS.items():
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                     start_date="20240101", end_date="20500101", adjust="qfq")
            etfs[sym] = df.set_index(pd.to_datetime(df["日期"]))["收盘"].sort_index()
        except Exception:  # noqa: BLE001 - 东财被风控, 回退新浪
            df = ak.fund_etf_hist_sina(symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            etfs[sym] = df.set_index("date")["close"].sort_index()
    return idx, etfs


def regime_score(close, fast_window):
    """0.6*tanh趋势 + 0.4*金叉; fast_window=250 为基础分, =30 为快线分(金叉项 sma50>sma_fast)"""
    sma_fast = close.rolling(fast_window).mean()
    sma50 = close.rolling(50).mean()
    dev = (close.iloc[-1] - sma_fast.iloc[-1]) / sma_fast.iloc[-1]
    trend = 0.5 + 0.5 * np.tanh(dev * 10)
    if fast_window == 250:
        golden = 1.0 if sma50.iloc[-1] > sma_fast.iloc[-1] else 0.0
    else:
        golden = 1.0 if sma50.iloc[-1] > sma_fast.iloc[-1] else 0.0  # 实盘口径: sma50 vs sma30
    return 0.6 * trend + 0.4 * golden


def breadth_at(etfs, offset):
    """offset=0 当日, offset=15 为15个交易日前; 3只ETF站上各自SMA50的比例"""
    up = 0
    for sym, close in etfs.items():
        sma50 = close.rolling(50).mean()
        i = -1 - offset
        if len(close) >= 50 + offset and close.iloc[i] > sma50.iloc[i]:
            up += 1
    return up / len(etfs)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"active": False, "trigger_date": None, "expire_date": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    idx, etfs = fetch_daily()
    today = str(idx.index[-1].date())
    cal = idx.index  # 用指数日历近似交易日历

    base = regime_score(idx, 250)
    s30 = regime_score(idx, 30)
    b_now = breadth_at(etfs, 0)
    b_prev = breadth_at(etfs, LOOKBACK)
    trig = (base >= BASE_MIN) and (s30 >= S30_MIN) and (b_now > 2 / 3) and (b_prev < 1 / 3)

    state = load_state()
    # 到期检查(按指数日历数交易日)
    if state.get("active") and state.get("expire_date") and today >= state["expire_date"]:
        state = {"active": False, "trigger_date": None, "expire_date": None}
        print(f"[{today}] 🔓 SURGE 锁定到期, 恢复 SMA250 制度判断")

    if trig and not state.get("active"):
        pos = cal.get_indexer([idx.index[-1]])[0]
        expire = cal[min(pos + LOCK_DAYS - 1, len(cal) - 1)]
        # 若未来日历不足(必然), 到期日按自然日估算: 21交易日≈29自然日
        expire_date = str((idx.index[-1] + pd.Timedelta(days=29)).date()) if pos + LOCK_DAYS - 1 >= len(cal) else str(expire.date())
        state = {"active": True, "trigger_date": today, "expire_date": expire_date}
        print(f"[{today}] 🚨🚨 SURGE 触发! 广度 {b_prev:.0%}→{b_now:.0%}, SMA30分 {s30:.3f}, "
              f"SMA250分 {base:.3f} —— 次日开盘按 RISKON 进场, 锁定至 {expire_date}")
    elif state.get("active"):
        print(f"[{today}] 🔒 SURGE 锁定中(触发 {state['trigger_date']}, 至 {state['expire_date']}), "
              f"制度强制 RISKON | 广度 {b_now:.0%} s30 {s30:.3f} base {base:.3f}")
    else:
        print(f"[{today}] ── 未触发 | 广度 {b_now:.0%}(15日前 {b_prev:.0%}) "
              f"s30 {s30:.3f} base {base:.3f}")

    state["checked_date"] = today
    state["detail"] = {"breadth_now": round(b_now, 3), "breadth_prev": round(b_prev, 3),
                       "s30": round(s30, 3), "base": round(base, 3)}
    if not args.dry:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
