#!/usr/bin/env python3
"""
IVOL (特质波动率) + 股价硬过滤 — 对比测试

测试以下组合 vs 当前最优 (liq+mom, 无择时):
  1. liq + mom + ivol (三因子, 替换low_vol)
  2. liq + mom + price_filter (两因子 + 低价股剔除)
  3. liq + mom + ivol + price_filter (三因子 + 低价股剔除)
  4. ivol only (单因子基准)
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src import factors_v5
from run_final_backtest import load_etf_prices, load_index, compute_regime
from run_v5_backtest import month_end_dates, metrics_from_values

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
STOCK_DIR = os.path.join(DATA_DIR, "stocks")
TOP_N = 30
REVERSAL_PCT = 0.20
PRICE_MIN = 3.0   # 剔除 3 元以下股票


def get_sc_constituents():
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


def compute_ivol_panel(stock_data, calendar, index_close):
    """
    计算特质波动率面板:
      IVOL = std(stock_ret - beta * mkt_ret) over 250d window
      用 CSI500 指数作为市场代理, beta 按同窗滚动估计。

    返回: DataFrame (calendar × codes), IVOL 值, 越低越好
    """
    close = pd.DataFrame({c: sdf["close"] for c, sdf in stock_data.items()}).reindex(calendar)
    ret = close.pct_change()
    mkt_ret = index_close.pct_change().reindex(calendar)

    # 滚动 beta: cov(stock, mkt) / var(mkt) over 250d
    # pandas rolling cov returns MultiIndex columns, 逐列算
    ivol = pd.DataFrame(index=calendar, columns=close.columns, dtype=float)
    win = 250
    min_periods = 200

    for code in close.columns:
        r = ret[code].dropna()
        m = mkt_ret.reindex(r.index).dropna()
        common = r.index.intersection(m.index)
        if len(common) < min_periods:
            continue
        r = r[common]
        m = m[common]

        # Rolling beta
        cov_roll = r.rolling(win, min_periods=min_periods).cov(m)
        var_roll = m.rolling(win, min_periods=min_periods).var()
        beta = cov_roll / var_roll.replace(0, np.nan)

        # Systematic return = beta * mkt
        sys_ret = beta * m

        # Idiosyncratic return = actual - systematic
        idio_ret = r - sys_ret

        # IVOL = rolling std of idiosyncratic returns
        ivol_code = idio_ret.rolling(win, min_periods=min_periods).std()
        ivol[code] = ivol_code

    return ivol


def build_select_fn_with_ivol(stock_data, calendar, aux, qroe, ivol_panel,
                               active_factors, top_n=TOP_N, reversal_pct=REVERSAL_PCT,
                               price_min=0.0):
    """
    Build select_fn using IVOL (replacing low_vol) + optional price filter.

    active_factors: list of factor names. Supported: 'ivol','liquidity','quality','momentum','ep'
    """
    n = len(active_factors)
    w = {f: (1.0 / n if f in active_factors else 0.0)
         for f in ['low_vol', 'liquidity', 'quality', 'momentum', 'ep']}

    # Build base V5 select_fn (include all factors, weights control which matter)
    # We hijack the low_vol slot with IVOL later
    base_sf = factors_v5.build_select_fn(
        stock_data, calendar, aux, qroe,
        top_n=top_n, weights=w, reversal_pct=reversal_pct,
        low_vol_window=250, cap_neutral=False,
    )

    # Get the pre-computed panels from V5 (we need close for price filter)
    close = pd.DataFrame({c: sdf["close"] for c, sdf in stock_data.items()}).reindex(calendar)
    ret20 = close / close.shift(20) - 1.0
    ret = close.pct_change()
    turn, amount, pe, isst = aux["turn"], aux["amount"], aux["peTTM"], aux["isST"]
    # PMO + Amihud
    pmo = turn.rolling(20, min_periods=15).sum() / turn.rolling(250, min_periods=200).sum()
    ret_nolimit = ret.where(ret.abs() < 0.095)
    amihud = (ret_nolimit.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    # Sharpe momentum
    mom_ret = close.shift(20) / close.shift(120) - 1.0
    mom_vol = ret.shift(20).rolling(100, min_periods=80).std()
    sharpe_mom = mom_ret / mom_vol.replace(0, np.nan)
    max20 = ret.rolling(20).max()
    # EP
    ep = 1.0 / pe.where(pe > 0)
    # Quality
    bars_count = close.notna().cumsum()

    def _pct_rank(s, good_high):
        return s.rank(pct=True, ascending=good_high)

    def select_fn(date):
        if date not in close.index:
            return []

        alive = close.loc[date].notna() & (bars_count.loc[date] >= 250)
        idx = alive[alive].index
        if len(idx) < top_n:
            return []

        # Hard filter 1: ST
        st_row = isst.loc[date].reindex(idx)
        idx = idx[(st_row != 1).fillna(True).values]

        # Hard filter 2: Reversal
        r20 = ret20.loc[date].reindex(idx)
        cut = r20.quantile(1 - reversal_pct)
        idx = idx[(r20 < cut).fillna(False).values]
        if len(idx) < top_n:
            return []

        # Hard filter 3: Price floor (NEW)
        if price_min > 0:
            px = close.loc[date].reindex(idx)
            idx = idx[(px >= price_min).fillna(False).values]
            if len(idx) < top_n:
                return []

        # Factor scores
        r_liq = pd.concat([
            _pct_rank(pmo.loc[date].reindex(idx), good_high=False),
            _pct_rank(amihud.loc[date].reindex(idx), good_high=True),
        ], axis=1).mean(axis=1)

        # IVOL: lower is better (replaces low_vol)
        r_ivol = _pct_rank(ivol_panel.loc[date].reindex(idx), good_high=False)

        # Quality
        qs = factors_v5.quality_scores_at(qroe, date)
        qdf = pd.DataFrame.from_dict(qs, orient="index", columns=["roe","droe","stab"]).reindex(idx)
        r_q = pd.concat([
            _pct_rank(qdf["roe"], good_high=True),
            _pct_rank(qdf["droe"], good_high=True),
            _pct_rank(qdf["stab"], good_high=False),
        ], axis=1).mean(axis=1)

        # Momentum
        r_mom = _pct_rank(sharpe_mom.loc[date].reindex(idx), good_high=True)
        m20 = max20.loc[date].reindex(idx)
        lottery = m20 >= m20.quantile(1 - factors_v5.MAX20_NEUTRAL_PCT)
        r_mom[lottery.fillna(False)] = 0.5

        # EP
        r_ep = _pct_rank(ep.loc[date].reindex(idx), good_high=True).fillna(0.0)

        # Composite with active factor weights
        comp = pd.Series(0.0, index=idx)
        wt_sum = 0.0
        for f in active_factors:
            wt = 1.0 / len(active_factors)
            if f == 'ivol':
                comp += wt * r_ivol.fillna(0.5)
            elif f == 'liquidity':
                comp += wt * r_liq.fillna(0.5)
            elif f == 'quality':
                comp += wt * r_q.fillna(0.5)
            elif f == 'momentum':
                comp += wt * r_mom.fillna(0.5)
            elif f == 'ep':
                comp += wt * r_ep
            wt_sum += wt

        comp = comp / wt_sum if wt_sum > 0 else comp
        return list(comp.sort_values(ascending=False).head(top_n).index)

    return select_fn


def main():
    print("=" * 70)
    print("  IVOL Factor + Price Filter Test")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    # Also load CSI500 index for IVOL calculation
    csi500_path = os.path.join(DATA_DIR, "index_sh000905.csv")
    if os.path.exists(csi500_path):
        csi500_idx = pd.read_csv(csi500_path, index_col=0, parse_dates=True)
        print(f"  CSI500 index: {csi500_idx.index[0].date()} ~ {csi500_idx.index[-1].date()}")
    else:
        csi500_idx = index_df  # fallback to CSI300
        print("  WARNING: CSI500 index not found, using CSI300 as market proxy")

    regime_series = compute_regime(index_df['close'])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = month_end_dates(cal)

    # Universe
    sc_codes = get_sc_constituents()
    stock_data = load_sc_stocks(set(sc_codes))
    codes = list(stock_data)
    print(f"  Universe: {len(codes)} stocks")

    # Force RISKON — no regime timing
    forced_riskon = {d: 'RISKON' for d in me_dates if d in cal}

    # Aux data
    aux = factors_v5.load_aux_panels(codes, cal)
    qroe = factors_v5.load_quarterly_roe()
    qroe = qroe[qroe['code'].isin(set(codes))]

    # Compute IVOL panel
    print("\n[2/4] Computing IVOL panel...")
    ivol_panel = compute_ivol_panel(stock_data, cal, csi500_idx['close'])
    n_ivol = ivol_panel.notna().any().sum()
    print(f"  IVOL coverage: {n_ivol}/{len(codes)} stocks")

    # Current best baseline: liq + mom, no timing
    def run_test(active_factors, label, price_min=0.0):
        sf = build_select_fn_with_ivol(
            stock_data, cal, aux, qroe, ivol_panel,
            active_factors=active_factors, price_min=price_min,
        )
        r = run_stock_backtest(
            df_close, regime_series, stock_data, top_n=TOP_N, verbose=False,
            execution='next_open', stamp_duty=True, ffill_valuation=True,
            df_open=df_open, rebalance_dates=me_dates, select_fn=sf,
            forced_regime=forced_riskon,
        )
        return metrics_from_values(r['values'])

    print("\n[3/4] Running tests...\n")

    tests = [
        (['liquidity', 'momentum'], "liq+mom (baseline)", 0.0),
        (['ivol', 'liquidity', 'momentum'], "ivol+liq+mom", 0.0),
        (['liquidity', 'momentum'], "liq+mom + price≥3", PRICE_MIN),
        (['ivol', 'liquidity', 'momentum'], "ivol+liq+mom + price≥3", PRICE_MIN),
        (['ivol'], "ivol only", 0.0),
        (['ivol', 'liquidity'], "ivol+liq", 0.0),
        (['ivol', 'momentum'], "ivol+mom", 0.0),
    ]

    results = {}
    for factors, label, pmin in tests:
        m = run_test(factors, label, price_min=pmin)
        results[label] = {**m, 'factors': factors, 'price_min': pmin, 'n': len(factors)}
        pm_label = f"  股价≥{pmin}" if pmin > 0 else ""
        print(f"  {label:<30s}  ann={m['ann']:>7.2f}%  mdd={m['mdd']:>7.1f}%  "
              f"sharpe={m['sharpe']:>5.2f}  n={len(factors)}{pm_label}")

    # Summary
    print("\n" + "=" * 70)
    print("  Results Summary (all no-timing, always 95% equity)")
    print("=" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ann'], reverse=True)
    print(f"{'Rank':<5s} {'Combo':<32s} {'N':<3s} {'Ann%':>8s} {'MDD%':>8s} {'Sharpe':>7s}  Notes")
    print("-" * 80)
    for rank, (name, m) in enumerate(sorted_results, 1):
        n = m['n']
        pm = f"price≥{m['price_min']}" if m['price_min'] > 0 else ""
        factors_str = '+'.join(m['factors'])
        print(f"{rank:<5d} {name:<32s} {n:<3d} {m['ann']:>8.2f} {m['mdd']:>8.1f} {m['sharpe']:>7.2f}  "
              f"{pm} [{factors_str}]")

    # Save
    out = os.path.join(os.path.dirname(__file__), 'backtests', 'sc_ivol.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'sub'}
                   for k, v in results.items()}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == '__main__':
    main()
