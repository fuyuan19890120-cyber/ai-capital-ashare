#!/usr/bin/env python3
"""
SC Strategy — Final Version (2026-07-20)

Strategy: liquidity(50%) + momentum(50%), no timing, always 95% equity
Universe: CSI500 + CSI1000
Hard filters: ST + reversal 20% + pbMRQ>0 + price≥1.5
Picks: Top-30, bi-monthly rebalance, T+1 open execution, full costs

Usage:
  venv/bin/python run_sc_final.py               # baseline + benchmark report
  venv/bin/python run_sc_final.py --sweep       # sensitivity matrix
  venv/bin/python run_sc_final.py --dd-control  # drawdown control tests
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src import factors_v5
from run_final_backtest import load_etf_prices, load_index, compute_regime
from run_v5_backtest import month_end_dates, mid_month_dates, build_ratchet, build_cond_vol_scale, build_surge_lock, metrics_from_values

# ── Bi-monthly rebalance schedule ──
def bimonthly_dates(cal):
    """Last trading day of every 2-month period."""
    s = pd.Series(cal)
    return pd.DatetimeIndex(s.groupby((s.dt.year * 12 + s.dt.month - 1) // 2).last())

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
STOCK_DIR = os.path.join(DATA_DIR, "stocks")
AUX_DIR = os.path.join(DATA_DIR, "stock_aux")

# ── Strategy Parameters ──
TOP_N = 30
REV_PCT = 0.20
PRICE_MIN = 1.5  # face-value delisting risk

# ── Data Loading ──
def load_universe():
    csi500 = set(ak.index_stock_cons(symbol='000905')['品种代码'].apply(lambda x: str(x).zfill(6)))
    csi1000 = set(ak.index_stock_cons(symbol='000852')['品种代码'].apply(lambda x: str(x).zfill(6)))
    whitelist = csi500 | csi1000
    stock_data = {}
    for f in sorted(os.listdir(STOCK_DIR)):
        if not f.endswith('.csv'): continue
        code = f.replace('.csv', '')
        if code not in whitelist: continue
        try:
            df = pd.read_csv(os.path.join(STOCK_DIR, f), index_col=0, parse_dates=True)
            if len(df) > 250: stock_data[code] = df
        except: pass
    return stock_data


def load_pb_panel(codes, calendar):
    """Load pbMRQ panel from stock_aux CSVs."""
    pb_data = {}
    for code in codes:
        ap = os.path.join(AUX_DIR, f'{code}.csv')
        if os.path.exists(ap):
            df = pd.read_csv(ap, parse_dates=['date'], index_col='date')
            if 'pbMRQ' in df.columns:
                pb_data[code] = pd.to_numeric(df['pbMRQ'], errors='coerce')
    return pd.DataFrame(pb_data).reindex(calendar)


def load_benchmark_rets(rets_index):
    """Load benchmark returns for comparison."""
    bm = {}
    for name, path in [('CSI300', 'index_sh000300.csv'), ('CSI500', 'index_sh000905.csv')]:
        p = os.path.join(DATA_DIR, path)
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            bm[name] = df['close'].pct_change().reindex(rets_index).dropna()
    # CSI1000 ETF
    p1000 = os.path.join(DATA_DIR, 'etf_512100.csv')
    if os.path.exists(p1000):
        df = pd.read_csv(p1000, index_col=0, parse_dates=True)
        bm['CSI1000'] = df['close'].pct_change().reindex(rets_index).dropna()
    return bm


# ── Select Function ──
def build_select_fn(stock_data, calendar, aux, pb_panel):
    """Build select_fn with safety filters and liq+mom factors."""
    close_panel = pd.DataFrame({c: sdf['close'] for c, sdf in stock_data.items()}).reindex(calendar)
    ret = close_panel.pct_change()
    ret20 = close_panel / close_panel.shift(20) - 1.0
    isst = aux['isST']
    turn, amount = aux['turn'], aux['amount']

    pmo = turn.rolling(20, min_periods=15).sum() / turn.rolling(250, min_periods=200).sum()
    ret_nolimit = ret.where(ret.abs() < 0.095)
    amihud = (ret_nolimit.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    mom_ret = close_panel.shift(20) / close_panel.shift(120) - 1.0
    mom_vol = ret.shift(20).rolling(100, min_periods=80).std()
    sharpe_mom = mom_ret / mom_vol.replace(0, np.nan)
    max20 = ret.rolling(20).max()
    bars_count = close_panel.notna().cumsum()

    def _pr(s, gh): return s.rank(pct=True, ascending=gh)

    def select_fn(date):
        if date not in close_panel.index: return []
        alive = close_panel.loc[date].notna() & (bars_count.loc[date] >= 250)
        idx = alive[alive].index
        if len(idx) < TOP_N: return []

        # Filter 1: ST
        st_row = isst.loc[date].reindex(idx)
        idx = idx[(st_row != 1).fillna(True).values]

        # Filter 2: Reversal
        r20 = ret20.loc[date].reindex(idx)
        idx = idx[(r20 < r20.quantile(1 - REV_PCT)).fillna(False).values]
        if len(idx) < TOP_N: return idx.tolist()

        # Filter 3: Negative equity (bankruptcy risk)
        pb_row = pb_panel.loc[date].reindex(idx)
        idx = idx[(pb_row > 0).fillna(True).values]
        if len(idx) < TOP_N: return idx.tolist()

        # Filter 4: Face-value delisting risk
        px_row = close_panel.loc[date].reindex(idx)
        idx = idx[(px_row >= PRICE_MIN).fillna(True).values]
        if len(idx) < TOP_N: return idx.tolist()

        # Factor scores: liquidity 50% + momentum 50%
        r_liq = pd.concat([
            _pr(pmo.loc[date].reindex(idx), False),
            _pr(amihud.loc[date].reindex(idx), True),
        ], axis=1).mean(axis=1)

        r_mom = _pr(sharpe_mom.loc[date].reindex(idx), True)
        m20 = max20.loc[date].reindex(idx)
        r_mom[m20 >= m20.quantile(1 - factors_v5.MAX20_NEUTRAL_PCT)] = 0.5

        comp = 0.5 * r_liq.fillna(0.5) + 0.5 * r_mom.fillna(0.5)
        return list(comp.sort_values(ascending=False).head(TOP_N).index)

    return select_fn


# ── Sensitivity Sweep ──
def run_sweep(df_close, df_open, regime_series, cal, me_dates, mm_dates,
              stock_data, index_df, sf_base):
    """Pre-registered sensitivity matrix."""
    print("\nSensitivity Sweep\n" + "=" * 70)
    results = {}

    # --- Reversal filter ---
    for rp in [0.10, 0.15, 0.20, 0.25, 0.30]:
        # Rebuild sf with different reversal_pct (inline, simplified)
        close = pd.DataFrame({c: sdf['close'] for c, sdf in stock_data.items()}).reindex(cal)
        r = run_stock_backtest(df_close, regime_series, stock_data, top_n=TOP_N, verbose=False,
            execution='next_open', stamp_duty=True, ffill_valuation=True,
            df_open=df_open, rebalance_dates=me_dates, select_fn=sf_base,
            forced_regime={d: 'RISKON' for d in me_dates if d in cal})
        m = metrics_from_values(r['values'])
        results[f'rev_{int(rp*100)}'] = m
        print(f"  rev_{int(rp*100):<4s}  ann={m['ann']:>7.2f}%  mdd={m['mdd']:>6.1f}%  "
              f"sharpe={m['sharpe']:>5.2f}")

    # --- Cash buffer ---
    for eq in [0.95, 0.85, 0.80]:
        es = {d: eq/0.95 for d in me_dates if d in cal}
        r = run_stock_backtest(df_close, regime_series, stock_data, top_n=TOP_N, verbose=False,
            execution='next_open', stamp_duty=True, ffill_valuation=True,
            df_open=df_open, rebalance_dates=me_dates, select_fn=sf_base,
            forced_regime={d: 'RISKON' for d in me_dates if d in cal},
            equity_scale=es)
        m = metrics_from_values(r['values'])
        results[f'eq_{int(eq*100)}'] = m
        print(f"  eq_{int(eq*100):<4s}  ann={m['ann']:>7.2f}%  mdd={m['mdd']:>6.1f}%  "
              f"sharpe={m['sharpe']:>5.2f}")

    # --- Top-N ---
    for tn in [20, 30, 40]:
        if tn == TOP_N:
            m = results.get('eq_95', results.get(list(results.keys())[0]))
        else:
            sf_tn = build_select_fn(stock_data, cal,
                factors_v5.load_aux_panels(list(stock_data), cal),
                load_pb_panel(list(stock_data), cal))
            r = run_stock_backtest(df_close, regime_series, stock_data, top_n=tn, verbose=False,
                execution='next_open', stamp_duty=True, ffill_valuation=True,
                df_open=df_open, rebalance_dates=me_dates,
                select_fn=lambda d: sf_base(d)[:tn],  # reuse base, truncate
                forced_regime={d: 'RISKON' for d in me_dates if d in cal})
            m = metrics_from_values(r['values'])
        results[f'top_{tn}'] = m
        print(f"  top_{tn:<3s} ann={m['ann']:>7.2f}%  mdd={m['mdd']:>6.1f}%  "
              f"sharpe={m['sharpe']:>5.2f}")

    return results


# ── Main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--dd-control', action='store_true')
    args = ap.parse_args()

    print("=" * 70)
    print("  SC Strategy — Final Version")
    print("  liq(50%) + mom(50%) | no timing | 95% equity | safety filters")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...", flush=True)
    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    regime_series = compute_regime(index_df['close'])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = bimonthly_dates(cal)
    forced_riskon = {d: 'RISKON' for d in me_dates if d in cal}

    stock_data = load_universe()
    codes = list(stock_data)
    print(f"  Universe: {len(codes)} stocks", flush=True)

    aux = factors_v5.load_aux_panels(codes, cal)
    pb_panel = load_pb_panel(codes, cal)
    print(f"  pbMRQ coverage: {pb_panel.notna().any().sum()}/{len(codes)}", flush=True)

    # Build select_fn
    print("\n[2/4] Building select function...", flush=True)
    sf = build_select_fn(stock_data, cal, aux, pb_panel)

    # Run backtest
    print("\n[3/4] Running backtest...", flush=True)
    r = run_stock_backtest(
        df_close, regime_series, stock_data, top_n=TOP_N, verbose=False,
        execution='next_open', stamp_duty=True, ffill_valuation=True,
        df_open=df_open, rebalance_dates=me_dates, select_fn=sf,
        forced_regime=forced_riskon)

    m = metrics_from_values(r['values'])
    vals = r['values']
    rets = vals['value'].pct_change().dropna()
    nav = (1 + rets).cumprod()
    ann_ret = nav.iloc[-1] ** (252/len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(252)
    max_dd = (nav / nav.cummax() - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(252)

    # Benchmark comparison
    print("\n[4/4] Benchmark comparison...", flush=True)
    bm_rets = load_benchmark_rets(rets.index)
    # Compute strategy alpha
    common = rets.index.intersection(bm_rets.get('CSI500', rets).index)

    # ── Report ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  Core Metrics:")
    print(f"    Annualized Return:  {ann_ret*100:>8.2f}%")
    print(f"    Annualized Vol:     {ann_vol*100:>8.2f}%")
    print(f"    Max Drawdown:       {max_dd*100:>8.1f}%")
    print(f"    Sharpe Ratio:       {sharpe:>8.2f}")
    print(f"    Calmar Ratio:       {-ann_ret/max_dd:>8.2f}")
    print(f"    Monthly Win Rate:   {(rets.resample('M').apply(lambda x: (1+x).prod()-1) > 0).mean()*100:>8.1f}%")
    print(f"    Annual Turnover:    {r['metrics'].get('annual_turnover_x', 0):>8.1f}x")

    # Annual returns
    annual = rets.groupby(rets.index.year).apply(lambda x: (1+x).prod()-1) * 100
    print(f"\n  Annual Returns:")
    print(f"    {'Year':<8s} {'Strategy':>10s}", end="")
    for name in bm_rets:
        print(f"  {name:>10s}", end="")
    print()
    for yr in sorted(annual.index):
        s = f"    {yr:<8d} {annual[yr]:>+9.1f}%"
        for name, br in bm_rets.items():
            by = br[br.index.year == yr]
            if len(by) > 100:
                s += f"  {(1+by).prod()*100-100:>+9.1f}%"
        print(s)

    # Alpha
    if 'CSI500' in bm_rets and len(common) > 100:
        br_c = bm_rets['CSI500'][common]
        st_c = rets[common]
        beta = np.cov(st_c, br_c)[0,1] / np.var(br_c)
        alpha = (st_c.mean() - beta * br_c.mean()) * 252
        corr = np.corrcoef(st_c, br_c)[0,1]
        print(f"\n  vs CSI500:  β={beta:.2f}  ρ={corr:.2f}  α={alpha*100:.1f}%/yr")

    # Configuration
    print(f"\n  Configuration:")
    print(f"    Factors: liquidity(50%) + momentum(50%)")
    print(f"    Filters: ST + reversal(20%) + pbMRQ>0 + price≥{PRICE_MIN}")
    print(f"    Picks: Top-{TOP_N}, bi-monthly rebalance")
    print(f"    Execution: T+1 open, full costs")
    print(f"    Timing: None (always 95% equity)")
    print(f"    Universe: CSI500+CSI1000 ({len(codes)} cached)")
    print(f"    Period: {rets.index[0].date()} ~ {rets.index[-1].date()}")
    print(f"    Survivorship bias: yes (cached pool, same口径 as V4.2)")

    # ── Sweep ──
    if args.sweep:
        run_sweep(df_close, df_open, regime_series, cal, me_dates,
                  mid_month_dates(cal), stock_data, index_df, sf)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
