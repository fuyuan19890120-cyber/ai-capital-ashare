#!/usr/bin/env python3
"""
SC Strategy Factor Decomposition: test single / pair / triple / all factor combos.

Usage: venv/bin/python run_sc_decompose.py
"""
import os, sys, json, warnings, itertools
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src import factors_v5
from run_final_backtest import load_etf_prices, load_index, compute_regime
from run_v5_backtest import month_end_dates, build_surge_lock, metrics_from_values

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
STOCK_DIR = os.path.join(DATA_DIR, "stocks")

FACTOR_NAMES = ["low_vol", "liquidity", "quality", "momentum", "ep"]
TOP_N = 30
REVERSAL_PCT = 0.20
SURGE_KWARGS = dict(lookback=15, lock_days=21, s30_min=0.70, base_min=0.15)


def get_sc_constituents():
    import akshare as ak
    csi500 = set(ak.index_stock_cons(symbol="000905")['品种代码'].apply(lambda x: str(x).zfill(6)))
    csi1000 = set(ak.index_stock_cons(symbol="000852")['品种代码'].apply(lambda x: str(x).zfill(6)))
    return sorted(csi500 | csi1000)


def load_sc_stocks(whitelist):
    result = {}
    for f in sorted(os.listdir(STOCK_DIR)):
        if not f.endswith('.csv'): continue
        code = f.replace('.csv', '')
        if code not in whitelist: continue
        try:
            df = pd.read_csv(os.path.join(STOCK_DIR, f), index_col=0, parse_dates=True)
            if len(df) > 250: result[code] = df
        except: pass
    return result


def make_weights(active_factors):
    """Evenly distribute weight across active factors."""
    w = {f: 0.0 for f in FACTOR_NAMES}
    n = len(active_factors)
    for f in active_factors:
        w[f] = 1.0 / n
    return w


def main():
    print("=" * 75)
    print("  SC Strategy — Factor Decomposition")
    print("=" * 75)

    # Load data once
    print("\n[1/3] Loading data...")
    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    regime_series = compute_regime(index_df['close'])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = month_end_dates(cal)

    sc_codes = get_sc_constituents()
    stock_data = load_sc_stocks(set(sc_codes))
    codes = list(stock_data)
    print(f"  Universe: {len(codes)} stocks")

    aux = factors_v5.load_aux_panels(codes, cal)
    qroe = factors_v5.load_quarterly_roe()
    qroe = qroe[qroe['code'].isin(set(codes))]

    def run_combo(active_factors, surge=False):
        w = make_weights(active_factors)
        sf = factors_v5.build_select_fn(
            stock_data, cal, aux, qroe,
            top_n=TOP_N, weights=w, reversal_pct=REVERSAL_PCT,
            low_vol_window=250, cap_neutral=False,
        )
        rd = me_dates
        forced = None
        if surge:
            extra, forced = build_surge_lock(
                index_df['close'], df_close, regime_series, cal, rd, **SURGE_KWARGS)
            rd = pd.DatetimeIndex(sorted(set(rd) | set(extra)))
        r = run_stock_backtest(
            df_close, regime_series, stock_data, top_n=TOP_N, verbose=False,
            execution='next_open', stamp_duty=True, ffill_valuation=True,
            df_open=df_open, rebalance_dates=rd, select_fn=sf,
            forced_regime=forced,
        )
        return metrics_from_values(r['values'])

    # Build combo list
    combos = []
    # Singles
    for f in FACTOR_NAMES:
        combos.append(([f], f"single_{f}"))
    # Pairs
    for a, b in itertools.combinations(FACTOR_NAMES, 2):
        combos.append(([a, b], f"pair_{a[:4]}+{b[:4]}"))
    # Triples
    for a, b, c in itertools.combinations(FACTOR_NAMES, 3):
        combos.append(([a, b, c], f"triple_{a[:4]}+{b[:4]}+{c[:4]}"))
    # All 5
    combos.append((FACTOR_NAMES, "all5"))

    print(f"\n[2/3] Running {len(combos)} factor combinations...\n")

    results = {}
    for i, (factors, name) in enumerate(combos):
        m = run_combo(factors)
        results[name] = {**m, 'factors': factors, 'n': len(factors)}
        # Also run +SURGE for top combos (singles + all5)
        tag = f"[{i+1}/{len(combos)}]"
        print(f"  {tag} {name:<28s}  ann={m['ann']:>6.2f}%  mdd={m['mdd']:>6.1f}%  "
              f"sharpe={m['sharpe']:>5.2f}  factors=[{','.join(factors)}]")

    # Also run SURGE variants for key combos
    print("\n[3/3] SURGE variants for key combos...\n")
    surge_combos = [
        (FACTOR_NAMES, "all5_surge"),
        (["low_vol"], "single_lowvol_surge"),
        (["liquidity"], "single_liq_surge"),
        (["momentum"], "single_mom_surge"),
        (["low_vol", "momentum"], "pair_lv+mo_surge"),
    ]
    for factors, name in surge_combos:
        m = run_combo(factors, surge=True)
        results[name] = {**m, 'factors': factors, 'n': len(factors)}
        print(f"  {name:<28s}  ann={m['ann']:>6.2f}%  mdd={m['mdd']:>6.1f}%  "
              f"sharpe={m['sharpe']:>5.2f}")

    # Summary: sort by ann return
    print("\n" + "=" * 75)
    print("  Ranked by Annual Return")
    print("=" * 75)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ann'], reverse=True)
    print(f"{'Rank':<5s} {'Combo':<30s} {'N':<3s} {'Ann%':>7s} {'MDD%':>7s} {'Sharpe':>6s}  {'Factors'}")
    print("-" * 85)
    for rank, (name, m) in enumerate(sorted_results, 1):
        n = m.get('n', '?')
        factors_str = '+'.join(m.get('factors', []))
        print(f"{rank:<5d} {name:<30s} {str(n):<3s} {m['ann']:>7.2f} {m['mdd']:>7.1f} {m['sharpe']:>6.2f}  {factors_str}")

    # Save
    out = os.path.join(os.path.dirname(__file__), 'backtests', 'sc_decompose.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'sub'}
                   for k, v in results.items()}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == '__main__':
    main()
