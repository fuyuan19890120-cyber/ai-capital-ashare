# -*- coding: utf-8 -*-
"""
可转债双低策略 V3 — Tushare 混合架构增强版
新增强:
  1. issue_size 过滤 (≥0.3亿, 排除微型转债; 用发行规模代替remain_size——后者退市归零会引入存活偏差)
  2. delist_date 精准存活偏差控制
  3. cb_basic 正股代码一致性已100%验证

数据源:
  - AKShare CSV: 日线 OHLCV + 转股价值(历史转股价修正后)
  - Tushare cb_basic: 基础信息 + issue_size + delist_date + remain_size(仅实盘参考)
  - _ratings.csv: 信用评级(存量)
  - stock_fund/: 正股基本面(存量)
"""
import pandas as pd, numpy as np
from pathlib import Path
import sqlite3, os

CB = Path.home() / "ai-capital-ashare" / "data" / "cb"
SF = CB / "stock_fund"
DB = Path.home() / "ai-capital-ashare" / "data" / "cb_tushare.db"
COST_RT = 0.001
IS_START, IS_CUT = pd.Timestamp("2018-01-01"), pd.Timestamp("2022-01-01")

# ============================================================
# 0. Load Tushare enhanced metadata
# ============================================================
conn = sqlite3.connect(DB)
basic = pd.read_sql("SELECT * FROM cb_basic", conn)
basic['code6'] = basic['ts_code'].str.replace('.SH','').str.replace('.SZ','')
basic['issue_size_yi'] = pd.to_numeric(basic['issue_size'], errors='coerce') / 100_000_000  # 元→亿
basic['remain_size_yi'] = pd.to_numeric(basic['remain_size'], errors='coerce') / 100_000_000
conn.close()

# Build lookup dicts
bond_meta = {}
for _, r in basic.iterrows():
    bond_meta[r['code6']] = {
        'issue_size_yi': r['issue_size_yi'],
        'remain_size_yi': r['remain_size_yi'],
        'delist_date': str(r['delist_date']) if pd.notna(r['delist_date']) else None,
        'list_date': str(r['list_date']),
        'conv_price': r['conv_price'],
        'stk_code': r['stk_code'],
    }

print(f"Tushare metadata loaded: {len(bond_meta)} CBs")

# ============================================================
# 1. R1: Stock fundamentals (same as V2)
# ============================================================
mine_by_year = {}
covered = set()
for f in SF.glob("*.csv"):
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if "日期" not in df.columns: continue
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    ann = df[df["日期"].dt.month == 12].sort_values("日期")
    if not len(ann): continue
    covered.add(f.stem)
    prof_col = next((c for c in df.columns if "净利润" in c), None)
    debt_col = next((c for c in df.columns if "资产负债率" in c), None)
    prof = pd.to_numeric(ann[prof_col], errors="coerce") if prof_col else pd.Series(np.nan, index=ann.index)
    debt = pd.to_numeric(ann[debt_col], errors="coerce") if debt_col else pd.Series(np.nan, index=ann.index)
    loss2 = (prof < 0) & (prof.shift(1) < 0)
    flag = loss2 | (debt > 90)
    for dt_, fl in zip(ann["日期"], flag):
        if fl:
            mine_by_year.setdefault(dt_.year, set()).add(f.stem)
print(f"R1 正股覆盖 {len(covered)} 只")

# ============================================================
# 2. R2: Ratings (existing _ratings.csv)
# ============================================================
lst = pd.read_csv(CB / "_list.csv", dtype=str)
lst["bond"] = lst["债券代码"].str.zfill(6); lst["stk"] = lst["正股代码"].str.zfill(6)
bond2stk = dict(zip(lst.bond, lst.stk))

rat = pd.read_csv(CB / "_ratings.csv", dtype={"债券代码": str})
rat["code"] = rat["债券代码"].str.zfill(6)
GOOD = {"AAA","AA+","AA","AA-","AA+sti","AAsti","AA-sti"}
rating_ok = set(rat[rat["信用评级"].astype(str).str.strip().isin(GOOD)].code)

# ============================================================
# 3. New: issue_size filter (使用发行规模, 不受退市归零影响)
# ============================================================
MIN_ISSUE_YI = 0.3  # 最小发行规模(亿)
issue_ok = set()
for code, meta in bond_meta.items():
    sz = meta['issue_size_yi']
    if pd.notna(sz) and sz >= MIN_ISSUE_YI:
        issue_ok.add(code)

# Stats
issue_ok_listed = sum(1 for c, m in bond_meta.items()
                      if m['delist_date'] is None and m['issue_size_yi'] is not None
                      and m['issue_size_yi'] >= MIN_ISSUE_YI)
total_with_issue = sum(1 for m in bond_meta.values() if m['issue_size_yi'] is not None)
print(f"R3 发行规模≥{MIN_ISSUE_YI}亿: {len(issue_ok)} "
      f"(存续 {issue_ok_listed}/{sum(1 for m in bond_meta.values() if m['delist_date'] is None)} "
      f"数据 {total_with_issue}/{len(bond_meta)})")

def mined_stocks_at(rd):
    y = rd.year - 1 if (rd.month, rd.day) >= (5, 1) else rd.year - 2
    return mine_by_year.get(y, set())

# ============================================================
# 4. Load price panel (same as V2 — AKShare CSV for 转股价值)
# ============================================================
frames = []
for f in sorted(CB.glob("[0-9]*.csv")):
    df = pd.read_csv(f)
    df.columns = ["date","close","bond_value","conv_value","bond_prem","conv_prem"][:len(df.columns)]
    df["date"] = pd.to_datetime(df["date"]); df["code"] = f.stem
    frames.append(df[["date","code","close","conv_prem"]])
px = pd.concat(frames, ignore_index=True).sort_values(["code","date"])
px = px[px.close.between(15,999) & px.conv_prem.between(-40,500)]
px["listed_days"] = px.groupby("code").cumcount()
px = px[px.listed_days >= 5]
px["dl"] = px.close + px.conv_prem
wide = px.pivot_table(index="date", columns="code", values="close")
dlw = px.pivot_table(index="date", columns="code", values="dl")
rets = wide.pct_change()  # forward-fill needed: first return after listing must not be NaN→0
dates = wide.index
print(f"Price panel: {len(wide.columns)} CBs, {len(dates)} trading days")

# ============================================================
# 5. Backtest engine
# ============================================================
def run(n=20, min_price=100, use_r1=False, r1_missing_excl=False, use_r2=False, use_r3=False):
    rb = [pd.Timestamp(d) for d in pd.Series(dates, index=dates).groupby(dates.to_period("W")).max().values
          if pd.Timestamp(d) >= IS_START - pd.Timedelta(days=15)]
    dr, ph = {}, []
    r3_excluded_count = 0
    for i, rd in enumerate(rb):
        if rd not in dlw.index: continue
        row = dlw.loc[rd].dropna()
        if min_price: row = row[wide.loc[rd].reindex(row.index) >= min_price]
        if use_r2: row = row[row.index.isin(rating_ok)]
        if use_r3:
            before = len(row)
            row = row[row.index.isin(issue_ok)]
            r3_excluded_count += (before - len(row))
        if use_r1:
            mined = mined_stocks_at(rd)
            def keep(b):
                s = bond2stk.get(b)
                if s is None or s not in covered:
                    return not r1_missing_excl
                return s not in mined
            row = row[[keep(b) for b in row.index]]
        if len(row) < n*2: continue
        hold = list(row.nsmallest(n).index)
        turn = 1.0 if not ph else len(set(hold)-set(ph))/n
        end = rb[i+1] if i+1 < len(rb) else dates[-1]
        for j, nd in enumerate(dates[(dates > rd) & (dates <= end)]):
            r = rets.loc[nd, hold].mean()
            if pd.isna(r): r = 0.0
            if j == 0: r -= turn * COST_RT
            dr[nd] = r
        ph = hold
    s = pd.Series(dr).sort_index(); return s[s.index >= IS_START], r3_excluded_count

def stats(s, label):
    out = f"{label:<42}"
    for nm, seg in [("内", s[s.index < IS_CUT]), ("外", s[s.index >= IS_CUT])]:
        yrs = (seg.index[-1]-seg.index[0]).days/365.25
        eq = (1+seg).cumprod()
        out += f" | {nm}: 年化{(eq.iloc[-1]**(1/yrs)-1)*100:+6.1f}% 回撤{(eq/eq.cummax()-1).min()*100:6.1f}%"
    y2223 = s[(s.index >= "2022-01-01") & (s.index < "2024-01-01")]
    out += f" | 22-23:{((1+y2223).prod()-1)*100:+6.1f}%"
    print(out)
    return (1+s).cumprod()

# ============================================================
# 6. Run comparison
# ============================================================
print(f"\n{'='*100}")
print("V2 基线 (原版AKShare, 参考)")
print(f"{'='*100}")
stats(run()[0], "基线 N=20+价≥100")
stats(run(use_r2=True)[0], "+R2评级≥AA-")
stats(run(use_r1=True, use_r2=True)[0], "+R1+R2 (V2最终推荐)")

print(f"\n{'='*100}")
print("V3 增强 (Tushare issue_size, 发行规模代理历史规模)")
print(f"{'='*100}")
stats(run(use_r3=True)[0], "+R3 发行规模≥0.3亿")
stats(run(use_r2=True, use_r3=True)[0], "+R2评级+R3规模")
stats(run(use_r1=True, use_r2=True, use_r3=True)[0], "+R1+R2+R3 (V3最终推荐)")

# ============================================================
# 7. V2 vs V3 detailed comparison
# ============================================================
print(f"\n{'='*100}")
print("V2 vs V3 详细对比")
print(f"{'='*100}")
s_v2, _ = run(use_r1=True, use_r2=True, use_r3=False)
s_v3, excl_count = run(use_r1=True, use_r2=True, use_r3=True)
print(f"V3 R3 过滤 {excl_count} 次(周×券)")

for label, ser in [("V2 R1+R2", s_v2), ("V3 R1+R2+R3(issue_size)", s_v3)]:
    eq = (1+ser).cumprod()
    dd = (eq/eq.cummax()-1)
    yrs = (ser.index[-1]-ser.index[0]).days/365.25
    sharpe = ser.mean()/ser.std()*np.sqrt(52) if ser.std() > 0 else 0
    print(f"\n{label}:")
    print(f"  全期年化: {((eq.iloc[-1])**(1/yrs)-1)*100:+.2f}%")
    print(f"  年化波动: {ser.std()*np.sqrt(52)*100:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  最大回撤: {dd.min()*100:.2f}%")
    print(f"  周胜率: {(ser>0).mean()*100:.1f}%")
    print(f"  月胜率: {ser.resample('ME').apply(lambda x: (1+x).prod()-1).pipe(lambda x: (x>0).mean())*100:.1f}%")
    y2223 = ser[(ser.index >= "2022-01-01") & (ser.index < "2024-01-01")]
    print(f"  22-23累计: {((1+y2223).prod()-1)*100:+.2f}%")

# ============================================================
# 8. Issue size filter impact sanity check
# ============================================================
print(f"\n{'='*100}")
print("R3 issue_size 过滤效果")
print(f"{'='*100}")
# How many CBs pass issue_size in different periods?
for period, start, end in [("2018", "2018-01-01", "2019-01-01"),
                            ("2020", "2020-01-01", "2021-01-01"),
                            ("2022", "2022-01-01", "2023-01-01"),
                            ("2024", "2024-01-01", "2025-01-01")]:
    period_codes = set(px[(px['date'] >= start) & (px['date'] < end)]['code'].unique())
    passing = period_codes & issue_ok
    print(f"  {period}: {len(passing)}/{len(period_codes)} pass issue_size≥{MIN_ISSUE_YI}亿 "
          f"({len(passing)/len(period_codes)*100:.0f}%)")

print("\nDone!")
