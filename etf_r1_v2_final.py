#!/usr/bin/env python3
"""
ETF-R1 终稿: 两档 0.70 + SURGE 28d(主策略) + SURGE 独立 42d(卫星策略)
复现: venv/bin/python etf_r1_v2_final.py
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

dfc, dfo = load_panels(); idx = load_index(); rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
lc = dfc.notna().cumsum(); lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20
ind_map = {}
mf = pd.read_csv("/Users/fuyuan/ai-capital-ashare/data/industry_map_bs.csv", dtype=str)
mf = mf[mf["industryClassification"] == "证监会行业分类"]; mf["bare"] = mf["code"].str[3:]
for _, r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"] != "": ind_map[r["bare"]] = str(r["industry"])
sd = load_stocks(None)

def make_sel():
    """V4 投票: RISKON/SURGE 2行业, SURGE 纯2行业, RISKON 2行业+2成长"""
    cache = {}
    def sel(date, d2t):
        if (date, id(d2t)) in cache: return cache[(date, id(d2t))]
        valid, votes = {}, {}
        for c, s in sd.items():
            if date in s.index and len(s[s.index <= date]) >= 250: valid[c] = s
        if not valid:
            r = [s for s in BROAD if lo(s, date)]; cache[(date, id(d2t))] = r; return r
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
            r = top[:2]
        else:
            r = top[:2]
            for g in ["sz159915", "sh588000"]:
                if lo(g, date) and g not in r: r.append(g)
            while len(r) < 4:
                for e, _ in ranked:
                    if e not in r and lo(e, date): r.append(e); break
                else: break
        cache[(date, id(d2t))] = r[:4]; return r[:4]
    return sel


def run_v2():
    """主策略: 两档 0.70 + SURGE 28d"""
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=28)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
    rs_mod = rs_raw.copy()
    for d in rs_mod.index: rs_mod.loc[d] = 0.85 if rs_mod.loc[d] >= 0.70 else 0.20
    sf = make_sel()
    rr = run_stock_backtest(
        dfc, rs_mod, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)
    return rr


def run_s42():
    """卫星策略: SURGE 独立 42d, 非 SURGE 全债"""
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=42)
    # 复审修复: build_surge 在触发日已处 RISKON 时不加 extra, 但 S42 的 rs_empty 全期 CRISIS,
    # 触发日本身就是切换日。将所有触发日显式加入调仓日列表。
    trig_dates = [d for d, t in d2t.items() if d == t]
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra) | set(trig_dates)))
    rs_empty = pd.Series(0.20, index=rs_raw.index)
    forced_all = {d: "RISKON" for d in d2t if d2t[d] is not None}
    sf = make_sel()
    rr = run_stock_backtest(
        dfc, rs_empty, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced_all)
    return rr


def stats(rr):
    v = rr["values"]["value"]; rt = v.pct_change().dropna()
    yrs = len(rt) / 244; ann = (v.iloc[-1] / 1e6) ** (1 / yrs) - 1
    dd = (v / v.cummax() - 1).min(); sh = rt.mean() / rt.std() * np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]
    yearly = {int(k): round(v, 1) for k, v in
              (rt.groupby(rt.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)).items()}
    return {"ann": round(ann * 100, 2), "mdd": round(dd * 100, 1), "sharpe": round(sh, 2),
            "turnover": round(t, 1), "yearly": yearly}


if __name__ == "__main__":
    print("V2 主策略(两档0.70+SURGE28d):", flush=True)
    r_v2 = run_v2()
    s_v2 = stats(r_v2)
    print(json.dumps(s_v2, ensure_ascii=False, indent=2))

    print("\nS42 卫星策略(SURGE独立42d):", flush=True)
    r_s42 = run_s42()
    s_s42 = stats(r_s42)
    print(json.dumps(s_s42, ensure_ascii=False, indent=2))

    out = os.path.join(os.path.dirname(__file__), "backtests", "etf_v2_s42_final.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"V2": s_v2, "S42": s_s42}, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")
