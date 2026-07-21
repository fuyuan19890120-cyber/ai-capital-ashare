#!/usr/bin/env python3
"""
ETF-R1 最终定稿: V4 投票行业 ETF + SURGE 集中优化
- 普通 RISKON: 2行业ETF(各25%)+创业板50(30%)+科创50(20%)
- SURGE 期: 纯2行业ETF(各50%, 100%集中)
- 信号栈: V4.1 SMA250四档 + SURGE三宽基口径(日频/锁21交易日)
- 行业映射: Baostock 证监会分类 + 关键词(覆盖率72%, 2026-07-13快照)
用法: venv/bin/python etf_r1_final.py
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from etf_r1_backtest import load_panels, build_surge, SECTOR_MAP, BROAD
from run_final_backtest import load_index, compute_regime, load_stocks
from run_v5_backtest import month_end_dates
from src.stock_backtest import run_stock_backtest
from src.stock_data import compute_stock_factors, select_top_stocks

dfc, dfo = load_panels(); idx = load_index(); rs = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
extra, forced, d2t = build_surge(idx["close"], dfc, rs, cal, me)
rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
lc = dfc.notna().cumsum()
lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

ind_map = {}
mf = pd.read_csv("/Users/fuyuan/ai-capital-ashare/data/industry_map_bs.csv", dtype=str)
mf = mf[mf["industryClassification"] == "证监会行业分类"]; mf["bare"] = mf["code"].str[3:]
for _, r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"] != "": ind_map[r["bare"]] = str(r["industry"])
sd = load_stocks(None)

cache = {}
def sel(date):
    if date in cache: return cache[date]
    valid, votes = {}, {}
    for c, s in sd.items():
        if date in s.index and len(s[s.index <= date]) >= 250: valid[c] = s
    if not valid:
        result = [s for s in BROAD if lo(s, date)]; cache[date] = result; return result
    scores = compute_stock_factors(valid, date); top30 = select_top_stocks(scores, 30)
    for code in top30[:30]:
        ind = ind_map.get(code, "")
        for etf, kws in SECTOR_MAP.items():
            if any(k in ind for k in kws): votes[etf] = votes.get(etf, 0) + 1; break
    ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    top = [e for e, _ in ranked[:2] if lo(e, date)]
    gb = [s for s in ["sz159915", "sh588000"] if lo(s, date)]
    while len(top) < 2 and gb: top.append(gb.pop(0))
    if len(top) < 2: top = [s for s in BROAD if lo(s, date)][:2]

    is_surge = d2t.get(date) is not None
    if is_surge:
        result = top[:2]  # SURGE: 纯2行业
    else:
        result = top[:2]  # 普通RISKON: 2行业+2成长
        for g in ["sz159915", "sh588000"]:
            if lo(g, date) and g not in result: result.append(g)
        while len(result) < 4:
            for e, _ in ranked:
                if e not in result and lo(e, date): result.append(e); break
            else: break
    cache[date] = result[:4]; return result[:4]

r = run_stock_backtest(dfc, rs, {}, top_n=4, verbose=False, execution="next_open",
                       stamp_duty=True, ffill_valuation=True, df_open=dfo,
                       rebalance_dates=rd, select_fn=sel, forced_regime=forced)
v = r["values"]["value"]; rt = v.pct_change().dropna()
yrs = len(rt) / 244; ann = (v.iloc[-1] / 1e6) ** (1 / yrs) - 1
dd = (v / v.cummax() - 1).min(); sh = rt.mean() / rt.std() * np.sqrt(244)
sub = {}
for a, b in [("2016", "2018"), ("2019", "2021"), ("2022", "2024"), ("2025", "2026")]:
    s = rt.loc[a:b]; sub[a + "-" + b[2:]] = round(((1 + s).prod() ** (244 / len(s)) - 1) * 100, 1)
yearly = {int(k): round(v, 1) for k, v in
          (rt.groupby(rt.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)).items()}

print(f"\n{'='*55}")
print("ETF 投票版 最终定稿(V4信号 + SURGE优化)")
print(f"{'='*55}")
print(f"年化 {ann*100:.2f}%  回撤 {dd*100:.1f}%  夏普 {sh:.2f}  换手 {r['metrics']['annual_turnover_x']:.1f}x")
print(f"分时段: {sub}")
print(f"逐年: {yearly}")
print(f"\n普通RISKON: 2行业ETF(各25%)+创业板50(30%)+科创50(20%)")
print(f"SURGE期间: 纯2行业ETF(各50%, 集中优化)")
print(f"对照 E0纯宽基 10.4%/-21.0%/0.69 | 个股版 V4.1可信 11.8%/-32.1%/0.65")

out = os.path.join(os.path.dirname(__file__), "backtests", "etf_r1_final.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"ann": round(ann*100,2), "mdd": round(dd*100,1), "sharpe": round(sh,2),
               "turnover": round(r["metrics"]["annual_turnover_x"],1), "sub": sub, "yearly": yearly},
              f, ensure_ascii=False, indent=2)
print(f"[saved] {out}")
