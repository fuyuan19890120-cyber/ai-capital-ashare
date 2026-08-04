#!/usr/bin/env python3
"""Test drawdown control methods for SC strategy — liq+mom baseline, no timing."""
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
from run_v5_backtest import month_end_dates, metrics_from_values, build_ratchet

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
STOCK_DIR = os.path.join(DATA_DIR, "stocks")
TOP_N = 30
REV_PCT = 0.20

# ── Data ──
print("[1/2] Loading data...", flush=True)
df_close, df_open = load_etf_prices(True)
index_df = load_index()
regime_series = compute_regime(index_df['close'])
cal = df_close.index[df_close.index >= START_DATE]
me_dates = month_end_dates(cal)

# Force all dates to RISKON (no regime timing)
forced_riskon = {d: 'RISKON' for d in me_dates if d in cal}

# Universe: CSI500 + CSI1000
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
print(f"  Universe: {len(codes)} stocks", flush=True)

aux = factors_v5.load_aux_panels(codes, cal)
qroe = factors_v5.load_quarterly_roe()
qroe = qroe[qroe['code'].isin(set(codes))]

# Base select_fn: liq 50% + mom 50%
W_BASE = {'low_vol': 0.0, 'liquidity': 0.5, 'quality': 0.0, 'momentum': 0.5, 'ep': 0.0}

def run(label, top_n=TOP_N, equity_scale=None, weights=None, downgrade=None):
    w = weights or W_BASE
    sf = factors_v5.build_select_fn(
        stock_data, cal, aux, qroe, top_n=top_n, weights=w, reversal_pct=REV_PCT)
    r = run_stock_backtest(
        df_close, regime_series, stock_data, top_n=top_n, verbose=False,
        execution='next_open', stamp_duty=True, ffill_valuation=True,
        df_open=df_open, rebalance_dates=me_dates, select_fn=sf,
        forced_regime=forced_riskon,
        equity_scale=equity_scale, downgrade_exec=downgrade)
    m = metrics_from_values(r['values'])
    calmar = round(-m['ann'] / m['mdd'], 2) if m['mdd'] < 0 else 0
    print(f"  {label:<32s} ann={m['ann']:>7.2f}%  mdd={m['mdd']:>7.1f}%  "
          f"sharpe={m['sharpe']:>5.2f}  calmar={calmar:>6.2f}", flush=True)
    return m

# ── Tests ──
print("\n[2/2] Running tests...\n", flush=True)

results = {}

# 1. Baseline
results['baseline (no timing, 95% eq)'] = run('baseline (no timing, 95% eq)')

# 2. 85% equity buffer
es_85 = {d: 0.85/0.95 for d in me_dates if d in cal}
results['85% equity (15% cash buffer)'] = run('85% equity (15% cash buffer)', equity_scale=es_85)

# 3. 80% equity buffer
es_80 = {d: 0.80/0.95 for d in me_dates if d in cal}
results['80% equity (20% cash buffer)'] = run('80% equity (20% cash buffer)', equity_scale=es_80)

# 4. Top-40 diversification
results['top-40 picks'] = run('top-40 picks', top_n=40)

# 5. Top-50 diversification
results['top-50 picks'] = run('top-50 picks', top_n=50)

# 6. liq70+mom30 (low-vol tilt)
w_lowvol = {'low_vol': 0.0, 'liquidity': 0.7, 'quality': 0.0, 'momentum': 0.3, 'ep': 0.0}
results['liq70+mom30 (lowvol tilt)'] = run('liq70+mom30 (lowvol tilt)', weights=w_lowvol)

# 7. liq60+mom40
w_mid = {'low_vol': 0.0, 'liquidity': 0.6, 'quality': 0.0, 'momentum': 0.4, 'ep': 0.0}
results['liq60+mom40'] = run('liq60+mom40', weights=w_mid)

# 8. Drawdown ratchet
dg = build_ratchet(index_df['close'], regime_series, cal, me_dates)
results['drawdown ratchet'] = run('drawdown ratchet', downgrade=dg)

# ── Summary ──
print(f"\n{'Strategy':<34s} {'Ann%':>8s} {'MDD%':>8s} {'Sharpe':>7s} {'Calmar':>7s}")
print("-" * 70)
for name, m in sorted(results.items(), key=lambda x: x[1]['ann'], reverse=True):
    calmar = round(-m['ann'] / m['mdd'], 2) if m['mdd'] < 0 else 0
    print(f"{name:<34s} {m['ann']:>8.2f} {m['mdd']:>8.1f} {m['sharpe']:>7.2f} {calmar:>7.2f}")

# Also rank by Calmar (return/drawdown)
print(f"\n── Ranked by Calmar Ratio ──")
print(f"{'Strategy':<34s} {'Ann%':>8s} {'MDD%':>8s} {'Calmar':>7s}")
print("-" * 60)
for name, m in sorted(results.items(), key=lambda x: -x['ann']/x['mdd'] if x['mdd']<0 else 0):
    calmar = round(-m['ann'] / m['mdd'], 2) if m['mdd'] < 0 else 0
    print(f"{name:<34s} {m['ann']:>8.2f} {m['mdd']:>8.1f} {calmar:>7.2f}")
