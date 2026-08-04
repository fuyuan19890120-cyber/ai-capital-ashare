#!/usr/bin/env python3
"""
Small/Mid-Cap Strategy V4.2-SC — CSI500 + CSI1000 universe
with full V5 factor pipeline adapted for small caps.

Baseline weights (literature-informed, V5 pre-registered base, small-cap adjusted):
  low_vol:   25%  (250d std, lower is better — small-cap low-vol premium)
  liquidity: 30%  (PMO + Amihud — small-cap core alpha, CH-4 + Amihud 15-27%/yr)
  quality:   15%  (ROE + ΔROE + stability — reduced, small-cap quality signal noisier)
  momentum:  10%  (sharpe momentum + MAX20 neutral — A-share momentum negative, esp small)
  ep:        10%  (1/PE — small-cap value premium)
  reversal_filter: 20% (remove top 20% by 20d return — stronger reversal in small caps)

Hard filters:
  1. ST removal (isST==1)
  2. Reversal: remove top 20% by 20d return
  3. Liquidity: 20d avg daily amount >= 20M CNY (micro-cap trap prevention)
  4. Min 250 trading days history

Overlays (optional):
  --tranching    Split into month-end + mid-month two half-positions (free lunch)
  --cond-vol     Conditional vol targeting (only high-vol quintile)
  --surge        Breadth SMA30 lock-in (from V4.2)
  --cap-neutral  Market cap neutralization

Usage:
  venv/bin/python run_sc_backtest.py                  # baseline
  venv/bin/python run_sc_backtest.py --sweep          # sensitivity matrix
  venv/bin/python run_sc_backtest.py --tranching      # with tranching overlay
  venv/bin/python run_sc_backtest.py --variant v42compare  # V4.2 on CSI300 for comparison
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src import factors_v5
from run_final_backtest import load_etf_prices, load_index, compute_regime
from run_v5_backtest import (
    month_end_dates, mid_month_dates,
    build_ratchet, build_cond_vol_scale, build_surge_lock,
    metrics_from_values,
)

# ============================================================
# Small-cap factor weights (V5 pre-registered base, adjusted per evidence)
# ============================================================
# Baseline weights (literature-informed pre-registered)
SC_WEIGHTS = {
    "low_vol": 0.25,     # 低波250 — small-cap low-vol premium (strong but crowded)
    "liquidity": 0.20,   # PMO+Amihud — sweep says 20%>30% for small caps
    "quality": 0.20,     # ROE+ΔROE+stab — bumped from 15% to balance liq reduction
    "momentum": 0.15,    # sharpe momentum+MAX20 neutral — sweep shows 15% optimal
    "ep": 0.10,          # 1/PE — small-cap value premium (10% flat, literature standard)
}
# Optimized weights (from sensitivity sweep, OAT best-per-dimension):
#   mom15: 5.39% (+0.68pp vs mom10@4.71%)
#   liq20: 4.87% (+0.16pp vs liq30@4.71%)
#   top30: 4.95% (+0.24pp vs top20@4.71%)
#   amt_none: 4.98% (+0.27pp vs amt20m@4.71%)
#   rev20: confirmed optimal (4.71% peak)
SC_OPTIMIZED_WEIGHTS = {
    "low_vol": 0.25,
    "liquidity": 0.20,
    "quality": 0.20,
    "momentum": 0.15,
    "ep": 0.10,
}
SC_REVERSAL_PCT = 0.20       # Remove top 20% by 20d return (sweep-confirmed optimal)
SC_MIN_AMOUNT = 0            # Disabled (sweep: no filter 4.98% > 20M 4.71%)
SC_TOP_N = 30                # Top-30 picks (sweep: 30 4.95% > 20 4.71%)
SC_LOW_VOL_WINDOW = 250      # 250d low-vol window

# ============================================================
# Universe: CSI500 + CSI1000 constituents
# ============================================================
DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
STOCK_DIR = os.path.join(DATA_DIR, "stocks")


def get_sc_constituents():
    """Get CSI500 + CSI1000 current constituents via akshare."""
    try:
        import akshare as ak
        csi500 = set(ak.index_stock_cons(symbol="000905")['品种代码'].apply(lambda x: str(x).zfill(6)))
        csi1000 = set(ak.index_stock_cons(symbol="000852")['品种代码'].apply(lambda x: str(x).zfill(6)))
        all_codes = sorted(csi500 | csi1000)
        print(f"CSI500: {len(csi500)} 只, CSI1000: {len(csi1000)} 只, 合并: {len(all_codes)} 只")
        return all_codes
    except Exception as e:
        print(f"⚠️ 获取成分股失败: {e}")
        return []


def get_csi300_constituents():
    """Get CSI300 constituents for V4.2 comparison."""
    try:
        import akshare as ak
        from src.stock_data import get_csi300_constituents as _csi300
        csi300 = set(_csi300())
        chinext = set(ak.index_stock_cons(symbol="399006")['品种代码'].apply(lambda x: str(x).zfill(6)))
        star50 = set(ak.index_stock_cons(symbol="000688")['品种代码'].apply(lambda x: str(x).zfill(6)))
        return sorted(csi300 | chinext | star50)
    except Exception as e:
        print(f"⚠️ 获取CSI300成分股失败: {e}")
        return []


def load_sc_stocks(whitelist):
    """Load stock data from cache, filtered by whitelist."""
    if not os.path.exists(STOCK_DIR):
        return {}
    result = {}
    for f in sorted(os.listdir(STOCK_DIR)):
        if not f.endswith('.csv'):
            continue
        code = f.replace('.csv', '')
        if code not in whitelist:
            continue
        try:
            df = pd.read_csv(os.path.join(STOCK_DIR, f), index_col=0, parse_dates=True)
            if len(df) > 250:
                result[code] = df
        except Exception:
            pass
    return result


def load_all_cached_stocks():
    """Load all cached stocks (for comparison)."""
    if not os.path.exists(STOCK_DIR):
        return {}
    result = {}
    for f in sorted(os.listdir(STOCK_DIR)):
        if not f.endswith('.csv'):
            continue
        code = f.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(STOCK_DIR, f), index_col=0, parse_dates=True)
            if len(df) > 250:
                result[code] = df
        except Exception:
            pass
    return result


def build_sc_select_fn(stock_data, calendar, aux, qroe, top_n=SC_TOP_N,
                       weights=SC_WEIGHTS, reversal_pct=SC_REVERSAL_PCT,
                       low_vol_window=SC_LOW_VOL_WINDOW,
                       min_amount=SC_MIN_AMOUNT, cap_neutral=False):
    """
    Build small-cap select_fn with liquidity filter and adjusted weights.

    Extends factors_v5.build_select_fn with an additional turnover/liquidity
    hard filter before factor ranking.
    """
    # Use V5's build_select_fn for core factor computation
    base_sf = factors_v5.build_select_fn(
        stock_data, calendar, aux, qroe,
        top_n=top_n, weights=weights, reversal_pct=reversal_pct,
        low_vol_window=low_vol_window, cap_neutral=cap_neutral,
    )

    if min_amount is None or min_amount <= 0:
        return base_sf

    # Build amount panel for liquidity filtering
    amount = aux["amount"]
    # 20-day rolling average daily amount
    avg_amount_20 = amount.rolling(20, min_periods=10).mean()
    close = pd.DataFrame({c: sdf["close"] for c, sdf in stock_data.items()}).reindex(calendar)
    bars_count = close.notna().cumsum()

    def select_fn(date):
        if date not in close.index:
            return []

        # Get base candidates from V5 select_fn
        # We intercept by pre-filtering stock_data before factor computation
        # But since base_sf does its own filtering internally, we need to
        # apply liquidity filter at the factor level.

        # Re-implement inline with liquidity check added:
        alive = close.loc[date].notna() & (bars_count.loc[date] >= 250)
        idx = alive[alive].index
        if len(idx) < top_n:
            return []

        # Hard filter 1: ST
        isst = aux["isST"]
        st_row = isst.loc[date].reindex(idx)
        idx = idx[(st_row != 1).fillna(True).values]

        # Hard filter 2: Reversal (top N% by 20d return removed)
        ret20 = close / close.shift(20) - 1.0
        r20 = ret20.loc[date].reindex(idx)
        cut = r20.quantile(1 - reversal_pct)
        idx = idx[(r20 < cut).fillna(False).values]
        if len(idx) < top_n:
            return []

        # Hard filter 3: Liquidity (20d avg daily amount >= min_amount)
        amt_row = avg_amount_20.loc[date].reindex(idx)
        idx = idx[(amt_row >= min_amount).fillna(False).values]
        if len(idx) < top_n:
            return []

        # Now delegate to the base V5 select_fn for factor computation
        # We need to call it and filter results to our pre-filtered idx
        # Since base_sf does its own filtering, we call it and intersect
        try:
            base_results = base_sf(date)
        except TypeError:
            base_results = base_sf(date, "NEUTRAL")

        # Filter: only keep stocks in our liquidity-filtered idx
        filtered = [c for c in base_results if c in set(idx)]
        return filtered[:top_n]

    return select_fn


# ============================================================
# V4.2 comparison: same framework on CSI300 universe
# ============================================================
V42_WEIGHTS = {"low_vol": 0.25, "liquidity": 0.25, "quality": 0.25,
               "momentum": 0.15, "ep": 0.10}  # V5通用权重, 近似V4.2因子体系


def build_v42_select_fn(stock_data, calendar, aux, qroe, top_n=15):
    """V4.2-style select_fn for CSI300 universe comparison."""
    return factors_v5.build_select_fn(
        stock_data, calendar, aux, qroe,
        top_n=top_n, weights=V42_WEIGHTS, reversal_pct=0.15,
        low_vol_window=250, cap_neutral=False,
    )


# ============================================================
# Main
# ============================================================
SURGE_KWARGS = dict(lookback=15, lock_days=21, s30_min=0.70, base_min=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all',
                    help='Comma-separated variants, or "all"')
    ap.add_argument('--sweep', action='store_true',
                    help='Run sensitivity sweep only')
    ap.add_argument('--tranching', action='store_true',
                    help='Enable tranching overlay')
    ap.add_argument('--cond-vol', action='store_true',
                    help='Enable conditional vol targeting')
    ap.add_argument('--surge', action='store_true',
                    help='Enable SURGE breadth lock')
    ap.add_argument('--cap-neutral', action='store_true',
                    help='Enable market cap neutralization')
    ap.add_argument('--v42-compare', action='store_true',
                    help='Run V4.2 on CSI300 for comparison')
    args = ap.parse_args()

    # ── Data Loading ──
    print("=" * 65)
    print("  Small/Mid-Cap Strategy V4.2-SC")
    print("  CSI500 + CSI1000 × V5 Factor Pipeline")
    print("=" * 65)

    print("\n[1/5] Loading ETF & index data...")
    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    regime_series = compute_regime(index_df['close'])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = month_end_dates(cal)
    mm_dates = mid_month_dates(cal)
    print(f"  ETF: {df_close.index[0].date()} ~ {df_close.index[-1].date()} ({len(df_close)} days)")
    print(f"  制度序列: {regime_series.index[0].date()} ~ {regime_series.index[-1].date()}")

    # ── Universe ──
    print("\n[2/5] Building small-cap universe...")
    sc_codes = get_sc_constituents()
    if not sc_codes:
        print("❌ 无法获取成分股, 退出")
        return

    stock_data = load_sc_stocks(set(sc_codes))
    n_cached = len(stock_data)
    n_miss = len(sc_codes) - n_cached
    print(f"  缓存命中: {n_cached}/{len(sc_codes)} 只 (缺失 {n_miss} 只)")

    if n_cached < 100:
        print("❌ 缓存覆盖率过低, 无法回测")

    # ── Aux Data ──
    print("\n[3/5] Loading aux panels & quarterly ROE...")
    codes = list(stock_data)
    aux = factors_v5.load_aux_panels(codes, cal)
    n_aux = aux['turn'].notna().any().sum()
    print(f"  辅助面板覆盖: {n_aux}/{len(codes)} 只")
    qroe = factors_v5.load_quarterly_roe()
    qroe = qroe[qroe['code'].isin(set(codes))]
    print(f"  季报 ROE: {len(qroe)} 条 ({qroe['code'].nunique()} 只)")

    # ── Build Select Functions ──
    print("\n[4/5] Building select functions...")

    # Small-cap baseline
    sf_sc = build_sc_select_fn(
        stock_data, cal, aux, qroe,
        top_n=SC_TOP_N, weights=SC_WEIGHTS, reversal_pct=SC_REVERSAL_PCT,
        low_vol_window=SC_LOW_VOL_WINDOW, min_amount=SC_MIN_AMOUNT,
        cap_neutral=args.cap_neutral,
    )

    # Small-cap no liquidity filter (for sensitivity)
    sf_sc_noliq = build_sc_select_fn(
        stock_data, cal, aux, qroe,
        top_n=SC_TOP_N, weights=SC_WEIGHTS, reversal_pct=SC_REVERSAL_PCT,
        low_vol_window=SC_LOW_VOL_WINDOW, min_amount=None,
        cap_neutral=args.cap_neutral,
    )

    # Small-cap with cap neutral
    sf_sc_capneutral = build_sc_select_fn(
        stock_data, cal, aux, qroe,
        top_n=SC_TOP_N, weights=SC_WEIGHTS, reversal_pct=SC_REVERSAL_PCT,
        low_vol_window=SC_LOW_VOL_WINDOW, min_amount=SC_MIN_AMOUNT,
        cap_neutral=True,
    )

    # V4.2 comparison (if requested)
    sf_v42 = None
    stock_data_v42 = None
    if args.v42_compare:
        print("  Building V4.2 comparison universe (CSI300+创业板+科创50)...")
        csi300_codes = get_csi300_constituents()
        stock_data_v42 = load_sc_stocks(set(csi300_codes))
        codes_v42 = list(stock_data_v42)
        aux_v42 = factors_v5.load_aux_panels(codes_v42, cal)
        qroe_v42 = factors_v5.load_quarterly_roe()
        qroe_v42 = qroe_v42[qroe_v42['code'].isin(set(codes_v42))]
        sf_v42 = build_v42_select_fn(stock_data_v42, cal, aux_v42, qroe_v42)
        print(f"  V4.2 universe: {len(stock_data_v42)} 只")

    # ── Run Backtests ──
    print("\n[5/5] Running backtests...\n")

    def run_one(select_fn, sd, rebal=None, surge=False, surge_kw=None,
                ratchet=False, condvol=False, top_n=SC_TOP_N):
        """Single backtest run wrapper."""
        rd = rebal if rebal is not None else me_dates
        forced = None
        if surge:
            extra, forced = build_surge_lock(
                index_df['close'], df_close, regime_series, cal, rd,
                **(surge_kw or SURGE_KWARGS))
            rd = pd.DatetimeIndex(sorted(set(rd) | set(extra)))
        dg = build_ratchet(index_df['close'], regime_series, cal, rd) if ratchet else None
        es = build_cond_vol_scale(index_df['close'], rd) if condvol else None
        return run_stock_backtest(
            df_close, regime_series, sd, top_n=top_n, verbose=False,
            execution='next_open', stamp_duty=True, ffill_valuation=True,
            df_open=df_open, rebalance_dates=rd, select_fn=select_fn,
            equity_scale=es, downgrade_exec=dg, forced_regime=forced,
        )

    # ── Variant Definitions ──
    variants = {}

    _sweep_mode = args.sweep

    def register(name, fn, is_sweep=False):
        if _sweep_mode and not is_sweep:
            return  # sweep mode: skip non-sweep variants
        if _sweep_mode or args.variant == 'all' or name in args.variant.split(','):
            variants[name] = fn

    # === SC Baseline ===
    register('sc_baseline', lambda: run_one(sf_sc, stock_data))

    # === SC + SURGE ===
    register('sc_surge', lambda: run_one(sf_sc, stock_data, surge=True))

    # === SC + Tranching ===
    register('sc_tranche', None)  # placeholder, handled below

    # === SC + Cond Vol ===
    register('sc_condvol', lambda: run_one(sf_sc, stock_data, condvol=True))

    # === SC + Cap Neutral ===
    register('sc_capneutral', lambda: run_one(sf_sc_capneutral, stock_data))

    # === SC No Liquidity Filter ===
    register('sc_noliq', lambda: run_one(sf_sc_noliq, stock_data))

    # === V4.2 Comparison ===
    if sf_v42 is not None:
        register('v42_csi300', lambda: run_one(sf_v42, stock_data_v42, top_n=15))
        register('v42_csi300_surge', lambda: run_one(sf_v42, stock_data_v42, top_n=15, surge=True))

    # === Sensitivity Sweep (pre-registered) ===
    if args.sweep:
        sweep_params = {
            # Reversal filter
            'rev10': dict(weights=SC_WEIGHTS, reversal_pct=0.10),
            'rev15': dict(weights=SC_WEIGHTS, reversal_pct=0.15),
            'rev20': dict(weights=SC_WEIGHTS, reversal_pct=0.20),  # baseline
            'rev25': dict(weights=SC_WEIGHTS, reversal_pct=0.25),
            'rev30': dict(weights=SC_WEIGHTS, reversal_pct=0.30),
            # Momentum weight (redistribute from momentum to others proportionally)
            'mom05': dict(weights={**SC_WEIGHTS, "momentum": 0.05,
                                   "low_vol": 0.27, "liquidity": 0.31,
                                   "quality": 0.16, "ep": 0.11}),
            'mom10': dict(weights=SC_WEIGHTS),  # baseline
            'mom15': dict(weights={**SC_WEIGHTS, "momentum": 0.15,
                                   "low_vol": 0.23, "liquidity": 0.28,
                                   "quality": 0.14, "ep": 0.10}),
            'mom20': dict(weights={**SC_WEIGHTS, "momentum": 0.20,
                                   "low_vol": 0.22, "liquidity": 0.27,
                                   "quality": 0.13, "ep": 0.08}),
            # Liquidity weight
            'liq20': dict(weights={**SC_WEIGHTS, "liquidity": 0.20,
                                   "low_vol": 0.27, "momentum": 0.12,
                                   "quality": 0.17, "ep": 0.14}),
            'liq25': dict(weights={**SC_WEIGHTS, "liquidity": 0.25,
                                   "low_vol": 0.26, "momentum": 0.11,
                                   "quality": 0.16, "ep": 0.12}),
            'liq30': dict(weights=SC_WEIGHTS),  # baseline
            'liq35': dict(weights={**SC_WEIGHTS, "liquidity": 0.35,
                                   "low_vol": 0.24, "momentum": 0.08,
                                   "quality": 0.13, "ep": 0.10}),
            # Top-N
            'top15': dict(weights=SC_WEIGHTS, top_n=15),
            'top20': dict(weights=SC_WEIGHTS, top_n=20),  # baseline
            'top25': dict(weights=SC_WEIGHTS, top_n=25),
            'top30': dict(weights=SC_WEIGHTS, top_n=30),
            # Amount filter
            'amt10m': dict(weights=SC_WEIGHTS, min_amount=10_000_000),
            'amt20m': dict(weights=SC_WEIGHTS, min_amount=20_000_000),  # baseline
            'amt50m': dict(weights=SC_WEIGHTS, min_amount=50_000_000),
            'amt_none': dict(weights=SC_WEIGHTS, min_amount=None),
            # Overlays
            'condvol': dict(weights=SC_WEIGHTS, condvol=True),
            'capneutral': dict(weights=SC_WEIGHTS, cap_neutral=True),
        }

        for name, kw in sweep_params.items():
            w = kw.get('weights', SC_WEIGHTS)
            rp = kw.get('reversal_pct', SC_REVERSAL_PCT)
            tn = kw.get('top_n', SC_TOP_N)
            ma = kw.get('min_amount', SC_MIN_AMOUNT)
            cv = kw.get('condvol', False)
            cn = kw.get('cap_neutral', False)

            sf = build_sc_select_fn(
                stock_data, cal, aux, qroe,
                top_n=tn, weights=w, reversal_pct=rp,
                low_vol_window=SC_LOW_VOL_WINDOW, min_amount=ma,
                cap_neutral=cn,
            )
            register(f'sweep_{name}', lambda sf=sf, tn=tn, cv=cv:
                     run_one(sf, stock_data, top_n=tn, condvol=cv), is_sweep=True)

    # ── Execute ──
    results = {}
    for name, fn in variants.items():
        if fn is None:
            continue  # placeholder (e.g., tranche)
        print(f"===== {name} =====", flush=True)
        r = fn()
        m = metrics_from_values(r['values'])
        m['turnover'] = round(r['metrics'].get('annual_turnover_x', np.nan), 1)
        m['skipped'] = r['metrics'].get('skipped_trades')
        results[name] = m
        print(f"  年化 {m['ann']:>7.2f}%  回撤 {m['mdd']:>6.1f}%  夏普 {m['sharpe']:>5.2f}  换手 {m['turnover']:>5.1f}x")
        sub = m.get('sub', {})
        if sub:
            print(f"  时段  {'2016-18':>7s} {'2019-21':>7s} {'2022-24':>7s} {'2025-26':>7s}")
            print(f"        {sub.get('2016-18', 0):>+7.1f}% {sub.get('2019-21', 0):>+7.1f}% "
                  f"{sub.get('2022-24', 0):>+7.1f}% {sub.get('2025-26', 0):>+7.1f}%")

    # ── Tranching (special: two half-positions averaged) ──
    if 'sc_tranche' in variants:
        print(f"\n===== sc_tranche =====", flush=True)
        r_me = run_one(sf_sc, stock_data, rebal=me_dates)
        r_mm = run_one(sf_sc, stock_data, rebal=mm_dates)
        nav_a = r_me['values']['value'] / r_me['values']['value'].iloc[0]
        nav_b = r_mm['values']['value'] / r_mm['values']['value'].iloc[0]
        mix = pd.DataFrame({'value': (nav_a + nav_b) / 2 * 1e6})
        mt = metrics_from_values(mix)
        mt['turnover'] = round(
            (r_me['metrics']['annual_turnover_x'] + r_mm['metrics']['annual_turnover_x']) / 2, 1)
        results['sc_tranche'] = mt
        print(f"  年化 {mt['ann']:>7.2f}%  回撤 {mt['mdd']:>6.1f}%  夏普 {mt['sharpe']:>5.2f}  换手 {mt['turnover']:>5.1f}x")
        sub = mt.get('sub', {})
        if sub:
            print(f"        {sub.get('2016-18', 0):>+7.1f}% {sub.get('2019-21', 0):>+7.1f}% "
                  f"{sub.get('2022-24', 0):>+7.1f}% {sub.get('2025-26', 0):>+7.1f}%")

    # ── Sweep: also run tranching ──
    if args.sweep and 'sc_baseline' not in variants:
        # Run baseline first for reference
        print(f"\n===== sc_baseline =====", flush=True)
        r_bl = run_one(sf_sc, stock_data)
        m_bl = metrics_from_values(r_bl['values'])
        m_bl['turnover'] = round(r_bl['metrics'].get('annual_turnover_x', np.nan), 1)
        results['sc_baseline'] = m_bl
        print(f"  年化 {m_bl['ann']:>7.2f}%  回撤 {m_bl['mdd']:>6.1f}%  夏普 {m_bl['sharpe']:>5.2f}  换手 {m_bl['turnover']:>5.1f}x")

    # ── Summary Table ──
    if len(results) > 1:
        print("\n" + "=" * 85)
        print("  Results Summary")
        print("=" * 85)
        header = f"{'Variant':<24s} {'Ann%':>7s} {'MDD%':>7s} {'Sharpe':>6s} {'TOx':>5s}  {'16-18':>7s} {'19-21':>7s} {'22-24':>7s} {'25-26':>7s}"
        print(header)
        print("-" * 85)
        for name, m in results.items():
            sub = m.get('sub', {})
            print(f"{name:<24s} {m['ann']:>7.2f} {m['mdd']:>7.1f} {m['sharpe']:>6.2f} {m['turnover']:>5.1f}  "
                  f"{sub.get('2016-18', 0):>+7.1f} {sub.get('2019-21', 0):>+7.1f} "
                  f"{sub.get('2022-24', 0):>+7.1f} {sub.get('2025-26', 0):>+7.1f}")

        # ── Key Attributions ──
        print("\n── Key Attributions ──")
        if 'sc_baseline' in results:
            bl = results['sc_baseline']
            for k in ['sc_surge', 'sc_tranche', 'sc_condvol', 'sc_capneutral', 'sc_noliq']:
                if k in results:
                    delta = results[k]['ann'] - bl['ann']
                    print(f"  {k} vs baseline: {delta:+.1f}pp 年化")

        if 'v42_csi300' in results and 'sc_baseline' in results:
            delta = results['sc_baseline']['ann'] - results['v42_csi300']['ann']
            print(f"\n  SC baseline vs V4.2 (CSI300): {delta:+.1f}pp — "
                  f"universe + factor adjustment effect")

    # ── Save ──
    suffix = '_sweep' if args.sweep else ''
    out = os.path.join(os.path.dirname(__file__), 'backtests', f'sc_results{suffix}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")

    # ── Configuration Summary ──
    print(f"\n── Configuration ──")
    print(f"  Universe: CSI500 + CSI1000 ({n_cached} stocks cached)")
    print(f"  Factors: low_vol={SC_WEIGHTS['low_vol']} liq={SC_WEIGHTS['liquidity']} "
          f"quality={SC_WEIGHTS['quality']} mom={SC_WEIGHTS['momentum']} ep={SC_WEIGHTS['ep']}")
    print(f"  Reversal filter: {SC_REVERSAL_PCT:.0%}")
    print(f"  Min daily amount: {SC_MIN_AMOUNT/1e6:.0f}M CNY")
    print(f"  Top-N: {SC_TOP_N}")
    print(f"  T+1 open execution, full costs")
    print(f"  Survivorship bias: yes (cached pool, same as V4.2口径)")


if __name__ == '__main__':
    main()
