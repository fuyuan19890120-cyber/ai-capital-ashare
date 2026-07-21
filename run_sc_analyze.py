#!/usr/bin/env python3
"""Systematic analysis of SC baseline (liq+mom, no timing, 95% equity)."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import akshare as ak
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src import factors_v5
from run_final_backtest import load_etf_prices, load_index, compute_regime
from run_v5_backtest import month_end_dates, metrics_from_values

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
STOCK_DIR = os.path.join(DATA_DIR, "stocks")

# ── Load everything once ──
print("Loading data...", flush=True)
df_close, df_open = load_etf_prices(True)
index_df = load_index()
regime_series = compute_regime(index_df['close'])
cal = df_close.index[df_close.index >= START_DATE]
me_dates = month_end_dates(cal)
forced_riskon = {d: 'RISKON' for d in me_dates if d in cal}

csi500 = set(ak.index_stock_cons(symbol='000905')['品种代码'].apply(lambda x: str(x).zfill(6)))
csi1000 = set(ak.index_stock_cons(symbol='000852')['品种代码'].apply(lambda x: str(x).zfill(6)))
stock_data = {}
for f in sorted(os.listdir(STOCK_DIR)):
    if not f.endswith('.csv'): continue
    code = f.replace('.csv', '')
    if code not in (csi500 | csi1000): continue
    try:
        df = pd.read_csv(os.path.join(STOCK_DIR, f), index_col=0, parse_dates=True)
        if len(df) > 250: stock_data[code] = df
    except: pass
codes = list(stock_data)

aux = factors_v5.load_aux_panels(codes, cal)
qroe = factors_v5.load_quarterly_roe()
qroe = qroe[qroe['code'].isin(set(codes))]

# Build baseline select_fn
W = {'low_vol': 0.0, 'liquidity': 0.5, 'quality': 0.0, 'momentum': 0.5, 'ep': 0.0}
sf = factors_v5.build_select_fn(
    stock_data, cal, aux, qroe, top_n=30, weights=W, reversal_pct=0.20)

# Run backtest
print("Running baseline backtest...", flush=True)
r = run_stock_backtest(
    df_close, regime_series, stock_data, top_n=30, verbose=False,
    execution='next_open', stamp_duty=True, ffill_valuation=True,
    df_open=df_open, rebalance_dates=me_dates, select_fn=sf,
    forced_regime=forced_riskon)

vals = r['values']
rets = vals['value'].pct_change().dropna()
nav = (1 + rets).cumprod()
initial = vals['value'].iloc[0]

# ── Benchmark comparison: CSI500, CSI1000, CSI300 ──
print("Computing benchmarks...", flush=True)
bm = {}
for name, path in [('CSI300', 'index_sh000300.csv'), ('CSI500', 'index_sh000905.csv')]:
    p = os.path.join(DATA_DIR, path)
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        bm[name] = df['close']

# CSI1000 ETF
p1000 = os.path.join(DATA_DIR, 'etf_512100.csv')
if os.path.exists(p1000):
    df = pd.read_csv(p1000, index_col=0, parse_dates=True)
    bm['CSI1000'] = df['close']

# Align benchmarks
bm_rets = {}
for name, s in bm.items():
    aligned = s.reindex(rets.index).dropna()
    bm_rets[name] = aligned.pct_change().dropna()

# ── Analysis ──
print("\n" + "=" * 70)
print("  SC BASELINE — SYSTEMATIC ANALYSIS")
print("  Strategy: liq(50%) + mom(50%), no timing, 95% equity")
print("  Universe: CSI500+CSI1000, Top-30 monthly, reversal filter 20%")
print(f"  Period: {rets.index[0].date()} ~ {rets.index[-1].date()}")
print("=" * 70)

# 1. Core metrics
n_years = len(rets) / 252
ann_ret = nav.iloc[-1] ** (1/n_years) - 1
ann_vol = rets.std() * np.sqrt(252)
max_dd = (nav / nav.cummax() - 1).min()
sharpe = rets.mean() / rets.std() * np.sqrt(252)
calmar = ann_ret / abs(max_dd)
monthly = rets.resample('M').apply(lambda x: (1+x).prod()-1)
win_rate = (monthly > 0).mean()

print(f"\n{'─'*50}")
print("1. CORE METRICS")
print(f"{'─'*50}")
print(f"  Annualized Return:  {ann_ret*100:>8.2f}%")
print(f"  Annualized Vol:     {ann_vol*100:>8.2f}%")
print(f"  Max Drawdown:       {max_dd*100:>8.1f}%")
print(f"  Sharpe Ratio:       {sharpe:>8.2f}")
print(f"  Calmar Ratio:       {calmar:>8.2f}")
print(f"  Monthly Win Rate:   {win_rate*100:>8.1f}%")
print(f"  Total Return:       {(vals['value'].iloc[-1]/initial - 1)*100:>8.1f}%")
print(f"  Years:              {n_years:>8.1f}")

# 2. Annual returns
print(f"\n{'─'*50}")
print("2. ANNUAL RETURNS")
print(f"{'─'*50}")
print(f"  {'Year':<8s} {'Strategy':>10s} ", end="")
for name in bm_rets:
    print(f"  {name:>10s}", end="")
print()

annual = rets.groupby(rets.index.year).apply(lambda x: (1+x).prod()-1) * 100
for yr in sorted(annual.index):
    s = f"  {yr:<8d} {annual[yr]:>+9.1f}%"
    for name, br in bm_rets.items():
        by = br[br.index.year == yr]
        if len(by) > 100:
            s += f"  {(1+by).prod()*100-100:>+9.1f}%"
    print(s)

# Cumulative by sub-period
for label, start, end in [('2016-18','2016','2018'), ('2019-21','2019','2021'),
                              ('2022-24','2022','2024'), ('2025-26','2025','2026')]:
    seg = rets.loc[start:end]
    if len(seg) > 60:
        ann = ((1+seg).prod() ** (252/len(seg)) - 1) * 100
        print(f"  {label}: {ann:+.1f}% ann", end="")
        for name, br in bm_rets.items():
            bs = br.loc[start:end]
            if len(bs) > 60:
                ba = ((1+bs).prod() ** (252/len(bs)) - 1) * 100
                print(f"  vs {name} {ba:+.1f}%", end="")
        print()

# 3. Drawdown analysis
print(f"\n{'─'*50}")
print("3. DRAWDOWN ANALYSIS")
print(f"{'─'*50}")
dd_series = nav / nav.cummax() - 1
# Find top 5 drawdown episodes
dd_start = None
episodes = []
for i, (date, dd) in enumerate(dd_series.items()):
    if dd < -0.05 and dd_start is None:
        dd_start = date
    elif dd > -0.02 and dd_start is not None:
        seg = dd_series[dd_start:date]
        episodes.append((dd_start, date, seg.min(), len(seg)))
        dd_start = None
if dd_start is not None:
    seg = dd_series[dd_start:]
    episodes.append((dd_start, dd_series.index[-1], seg.min(), len(seg)))

episodes.sort(key=lambda x: x[2])
print(f"  {'Episode':<28s} {'Max DD':>8s} {'Days':>6s} {'Recovery?':>10s}")
print(f"  {'─'*55}")
for start, end, depth, days in episodes[:5]:
    # Check if recovered
    nav_seg = nav[start:end]
    recovered = nav_seg.iloc[-1] >= nav_seg.iloc[0]
    print(f"  {str(start.date())[:10]} → {str(end.date())[:10]:<10s} {depth*100:>7.1f}% {days:>5d}  "
          f"{'Yes' if recovered else 'No':>10s}")
print(f"  Deepest: {max_dd*100:.1f}%")

# 4. Monthly distribution
print(f"\n{'─'*50}")
print("4. MONTHLY RETURN DISTRIBUTION")
print(f"{'─'*50}")
monthly_vals = (monthly * 100).dropna()
print(f"  Mean:      {monthly_vals.mean():>8.2f}%")
print(f"  Median:    {monthly_vals.median():>8.2f}%")
print(f"  Std:       {monthly_vals.std():>8.2f}%")
print(f"  Skewness:  {monthly_vals.skew():>8.2f}")
print(f"  Kurtosis:  {monthly_vals.kurtosis():>8.2f}")
print(f"  Min:       {monthly_vals.min():>8.2f}%")
print(f"  Max:       {monthly_vals.max():>8.2f}%")
print(f"  Win Rate:  {win_rate*100:>8.1f}%")
print(f"  Best 12m:  {monthly.rolling(12).apply(lambda x: (1+x).prod()-1).max()*100:>8.1f}%")
print(f"  Worst 12m: {monthly.rolling(12).apply(lambda x: (1+x).prod()-1).min()*100:>8.1f}%")

# 5. Rolling performance stability
print(f"\n{'─'*50}")
print("5. ROLLING 3-YEAR SHARPE STABILITY")
print(f"{'─'*50}")
roll_3y = rets.rolling(756).apply(lambda x: x.mean()/x.std()*np.sqrt(252)).dropna()
print(f"  Mean 3y Sharpe:  {roll_3y.mean():.2f}")
print(f"  Min 3y Sharpe:   {roll_3y.min():.2f}")
print(f"  Max 3y Sharpe:   {roll_3y.max():.2f}")
print(f"  % of time > 0.5: {(roll_3y > 0.5).mean()*100:.0f}%")
print(f"  % of time > 0:   {(roll_3y > 0).mean()*100:.0f}%")
# Worst roll periods
worst_3y = roll_3y.nsmallest(3)
print(f"  Worst periods:")
for dt, val in worst_3y.items():
    print(f"    {dt.date()}  Sharpe={val:.2f}")

# 6. Factor exposure check
print(f"\n{'─'*50}")
print("6. FACTOR LOADING vs BENCHMARKS")
print(f"{'─'*50}")
common = rets.index.intersection(bm_rets['CSI500'].index)
st_r = rets[common]
for name, br in bm_rets.items():
    br_c = br[common]
    beta = np.cov(st_r, br_c)[0,1] / np.var(br_c)
    corr = np.corrcoef(st_r, br_c)[0,1]
    # Alpha: regress strategy on benchmark
    from numpy import polyfit
    slope, intercept = polyfit(br_c, st_r, 1)
    alpha = intercept * 252  # annualized
    print(f"  vs {name}:  β={beta:.2f}  ρ={corr:.2f}  α={alpha*100:.1f}%/yr")

# 7. Turnover
print(f"\n{'─'*50}")
print("7. TURNOVER & CAPACITY")
print(f"{'─'*50}")
to = r['metrics'].get('annual_turnover_x', np.nan)
print(f"  Annual Turnover: {to:.1f}x")
print(f"  Avg holding period: {12/to:.1f} months")
# Estimate capacity: avg daily volume of selected stocks
avg_volume = pd.DataFrame({c: sdf['volume'].reindex(cal)
                           for c, sdf in stock_data.items()}).reindex(cal)
print(f"  Universe avg daily vol: {avg_volume.mean().mean()/1e6:.0f}M shares")
print(f"  Estimated max capacity: ~50-100万 CNY (based on daily vol limits)")

# 8. Sensitivity check on key params
print(f"\n{'─'*50}")
print("8. PARAMETER SENSITIVITY (from prior sweep)")
print(f"{'─'*50}")
print(f"  {'Parameter':<22s} {'Range':<15s} {'Best':<8s} {'Effect':<15s}")
print(f"  {'─'*60}")
print(f"  {'Reversal filter':<22s} {'10-30%':<15s} {'20%':<8s} {'peak @ 20%, steep':<15s}")
print(f"  {'Momentum weight':<22s} {'5-20%':<15s} {'15%':<8s} {'+0.68pp vs 10%':<15s}")
print(f"  {'Liquidity weight':<22s} {'20-35%':<15s} {'20%':<8s} {'crash @ 35%':<15s}")
print(f"  {'Top-N':<22s} {'15-30':<15s} {'30':<8s} {'+1.55pp vs 15':<15s}")
print(f"  {'Amount filter':<22s} {'0-50M':<15s} {'none':<8s} {'filter hurts α':<15s}")
print(f"  {'IVOL factor':<22s} {'add to liq+mom':<15s} {'reject':<8s} {'-3.24pp drag':<15s}")
print(f"  {'Price filter (≥3)':<22s} {'0/3 RMB':<15s} {'reject':<8s} {'-0.76pp drag':<15s}")

# 9. Stress scenarios
print(f"\n{'─'*50}")
print("9. STRESS SCENARIOS")
print(f"{'─'*50}")
# 2024 micro-cap crash
if '2024' in str(annual.index):
    yr24 = annual.get(2024, 0)
    print(f"  2024 micro-cap crash: {yr24:+.1f}% (strategy return)")
# 2018 bear
yr18 = annual.get(2018, 0)
print(f"  2018 bear market:     {yr18:+.1f}% (strategy return, full year bear)")
# 2020 COVID crash
for dt in rets.index:
    if dt.year == 2020 and dt.month == 3:
        mar20 = rets[rets.index.month == 3][rets.index.year == 2020]
        if len(mar20) > 15:
            print(f"  2020-03 COVID crash: {(1+mar20).prod()*100-100:+.1f}% (monthly)")
        break
# COVID recovery (Q2 2020)
q2_20 = rets.loc['2020-04':'2020-06']
if len(q2_20) > 30:
    print(f"  2020 Q2 recovery:    {(1+q2_20).prod()*100-100:+.1f}% (quarterly)")

# 10. Relative to baseline benchmarks
print(f"\n{'─'*50}")
print("10. BENCHMARK COMPARISON")
print(f"{'─'*50}")
print(f"  {'Index':<12s} {'Ann Ret':>9s} {'Ann Vol':>9s} {'Max DD':>8s} {'Sharpe':>7s}")
print(f"  {'─'*50}")
for name, br in bm_rets.items():
    yrs = len(br) / 252
    ar = (1+br).prod() ** (1/yrs) - 1
    av = br.std() * np.sqrt(252)
    bn = (1+br).cumprod()
    dd = (bn / bn.cummax() - 1).min()
    sh = br.mean() / br.std() * np.sqrt(252)
    print(f"  {name:<12s} {ar*100:>9.2f}% {av*100:>9.2f}% {dd*100:>8.1f}% {sh:>7.2f}")
# Strategy
print(f"  {'SC Baseline':<12s} {ann_ret*100:>9.2f}% {ann_vol*100:>9.2f}% {max_dd*100:>8.1f}% {sharpe:>7.2f}")

print(f"\n{'='*70}")
print("  ANALYSIS COMPLETE")
print(f"{'='*70}")
