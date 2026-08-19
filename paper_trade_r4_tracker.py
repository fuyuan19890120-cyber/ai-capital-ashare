#!/usr/bin/env python3
"""
ETF-R4 纸面实盘追踪 · 手动模拟 · 从6月底起
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from datetime import datetime

import backtest_etf_r4 as bt
bt.DB_PATH = os.path.expanduser("~/ai-capital-ashare/data/etf_tushare.db")  # tushare 前复权正确
from backtest_etf_r4 import load_qmt_data, build_panels, compute_regime_2factor, make_monthly_select

START = "2026-06-30"
CAPITAL = 1_000_000; COST = 0.001  # 0.1%单边(对齐回测)
TRACKER = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper_tracker.json")

df_raw = load_qmt_data()
close, open_, high, low, vol = build_panels(df_raw)
regime = compute_regime_2factor(close, tanh_mult=10)  # ×10(对齐策略)
monthly_sel = make_monthly_select(close, vol, high, low, use_kdj_defense=True)

# 调仓日期: 只包含"当月已结束"的月末(排除数据截止月, 避免把数据最后一天误当月末调仓)
all_days = close.index
md = {}
for d in all_days: md.setdefault((d.year,d.month),[]).append(d)
month_ends = sorted([v[-1] for v in md.values()])
data_end = close.index[-1]
rebal_dates = [me for me in month_ends if me >= pd.Timestamp(START) and me < data_end]
if not rebal_dates: rebal_dates = [close.index[-1]]

# 每日循环: 月末调仓 + 每日估值
cash = CAPITAL; positions = {}; records = []; daily_nav = []
rebal_set = set(rebal_dates)

for d in [x for x in close.index if x >= pd.Timestamp(START)]:
    # 月末调仓
    if d in rebal_set:
        rv = float(regime.loc[d]) if d in regime.index else 0.5
        # 估值上期持仓
        if positions:
            tv = cash
            for c, s in positions.items():
                px = float(close.at[d, c]) if pd.notna(close.at[d, c]) else 0
                tv += s * px
            prev_val = records[-1]["start_value"] if records else CAPITAL
            ret = (tv/prev_val - 1) * 100
            records[-1]["return_pct"] = round(ret, 1)
            records[-1]["end_value"] = round(tv, 2)
            records[-1]["end_date"] = str(d.date())
            cash = tv; positions = {}
        # 选股 + 买入
        picks = monthly_sel(d, rv, {})
        if picks:
            total_eq = cash * 0.95
            pos_desc = []
            for code, weight in picks:
                px = float(close.at[d, code]) if pd.notna(close.at[d, code]) else 0
                if px <= 0: continue
                shares = int(total_eq * weight / px / 100) * 100
                cost = shares * px * (1 + COST)
                if cost <= cash:
                    cash -= cost; positions[code] = shares
                    pos_desc.append(f"{code} {weight*100:.0f}%")
            records.append({
                "date": str(d.date()),
                "start_value": round(cash + sum(positions.get(c,0)*float(close.at[d,c]) for c in positions if pd.notna(close.at[d,c])), 2),
                "positions": pos_desc,
                "end_value": None, "end_date": None, "return_pct": None,
            })
    # 每日估值
    nav = cash
    for c, s in positions.items():
        px = float(close.at[d, c]) if pd.notna(close.at[d, c]) else 0
        nav += s * px
    daily_nav.append({"d": str(d.date()), "v": round(nav/1e4, 2),
                      "r": round(float(regime.loc[d]) if d in regime.index else 0.5, 3)})

# 最后一期至今收益
if records and positions:
    final_nav = daily_nav[-1]["v"] * 1e4
    prev = records[-1]["start_value"]
    ret = (final_nav/prev - 1) * 100 if prev > 0 else 0
    records[-1]["return_pct"] = round(ret, 1)
    records[-1]["end_value"] = round(final_nav, 2)
    records[-1]["end_date"] = str(close.index[-1].date())
else:
    final_nav = CAPITAL

cum_ret = (final_nav/CAPITAL - 1) * 100

tracker = {
    "strategy": "ETF-R4 70/30",
    "start_date": START,
    "initial_capital": CAPITAL,
    "current_nav": round(final_nav, 2),
    "total_return": round(cum_ret, 1),
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "records": records,
    "daily_nav": daily_nav,
}

json.dump(tracker, open(TRACKER, "w"), ensure_ascii=False, indent=1)

print(f"ETF-R4 纸面实盘: {START} ~ {records[-1]['end_date'] if records else START}")
for r in records:
    ret = r.get("return_pct")
    ret_str = f" → {ret:+.1f}%" if ret is not None else " → 持有中"
    print(f"  {r['date']}: {', '.join(r['positions'])}{ret_str}")
print(f"\n当前净值: {final_nav:,.0f}  累计收益: {cum_ret:+.1f}%")
