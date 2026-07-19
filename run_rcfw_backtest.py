#!/usr/bin/env python3
"""
RC-FW v1 制度条件因子权重 — 全试验矩阵

预注册(2026-07-19):
  主口径: rcfw_45 (RISKON动量45%), rcfw_35 (RISKON动量35%)
  覆盖层: +SURGE, +错峰双腿(tranching)
  对照: rcfw_fixedweight(固定权重), v4fixed(V4.1), v4fixed_surge

用法: venv/bin/python run_rcfw_backtest.py [--variant all]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src import factors_v5, factors_rcfw
from run_final_backtest import load_etf_prices, load_index, load_stocks, compute_regime
from run_v5_backtest import (
    month_end_dates, mid_month_dates,
    build_ratchet, build_cond_vol_scale, build_surge_lock,
    metrics_from_values,
)

# ── SURGE 默认参数 ──
SURGE_KWARGS = dict(lookback=15, lock_days=21, s30_min=0.70, base_min=0.15)


def run_one(select_fn, df_close, df_open, regime_series, stock_data,
            rebal=None, surge=False, surge_kw=None, top_n=15):
    """单次回测封装。返回 result dict(含 values 和 metrics)。"""
    rd = rebal if rebal is not None else month_end_dates(
        df_close.index[df_close.index >= START_DATE])
    forced = None
    if surge:
        extra, forced = build_surge_lock(
            load_index()['close'], df_close, regime_series,
            df_close.index[df_close.index >= START_DATE], rd,
            **(surge_kw or SURGE_KWARGS))
        rd = pd.DatetimeIndex(sorted(set(rd) | set(extra)))
    return run_stock_backtest(
        df_close, regime_series, stock_data, top_n=top_n, verbose=False,
        execution='next_open', stamp_duty=True, ffill_valuation=True,
        df_open=df_open, rebalance_dates=rd, select_fn=select_fn,
        forced_regime=forced,
    )


def tranche_combine(r1, r2):
    """错峰双腿: 两个半仓组合的 NAV 均值(与 V5 复审 C6 一致)"""
    nav_a = r1['values']['value'] / r1['values']['value'].iloc[0]
    nav_b = r2['values']['value'] / r2['values']['value'].iloc[0]
    mix = pd.DataFrame({'value': (nav_a + nav_b) / 2 * 1e6})
    m = metrics_from_values(mix)
    m['turnover'] = round((r1['metrics']['annual_turnover_x']
                           + r2['metrics']['annual_turnover_x']) / 2, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all')
    args = ap.parse_args()

    # ── 数据加载 ──
    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    stock_data_all = load_stocks(None)  # 全缓存池
    regime_series = compute_regime(index_df['close'])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = month_end_dates(cal)
    mm_dates = mid_month_dates(cal)

    codes = list(stock_data_all)
    aux = factors_v5.load_aux_panels(codes, cal)
    qroe = factors_v5.load_quarterly_roe()
    qroe = qroe[qroe['code'].isin(set(codes))]

    print(f"个股 {len(stock_data_all)} 只 | 辅助面板 {aux['turn'].notna().any().sum()} 只")
    print(f"季报 ROE {qroe['code'].nunique()} 只 | 日历 {cal[0].date()}~{cal[-1].date()}")
    print(f"月度调仓 {len(me_dates)} 次 | 半月调仓 {len(mm_dates)} 次")

    # ── 构建 select_fn ──
    sf_45 = factors_rcfw.build_select_fn(
        stock_data_all, cal, aux, qroe, momentum_45=True)
    sf_35 = factors_rcfw.build_select_fn(
        stock_data_all, cal, aux, qroe, momentum_45=False)
    sf_fixed = factors_rcfw.build_select_fn(
        stock_data_all, cal, aux, qroe, momentum_45=True)  # 后面手动用固定权重

    # ── 试验矩阵 ──
    variants = {}

    def register(name, fn):
        if args.variant == 'all' or name in args.variant.split(','):
            variants[name] = fn

    # === 主口径 ===
    register('rcfw_45', lambda: run_one(sf_45, df_close, df_open, regime_series, stock_data_all))
    register('rcfw_35', lambda: run_one(sf_35, df_close, df_open, regime_series, stock_data_all))

    # === +SURGE ===
    register('rcfw_45_surge', lambda: run_one(
        sf_45, df_close, df_open, regime_series, stock_data_all, surge=True))
    register('rcfw_35_surge', lambda: run_one(
        sf_35, df_close, df_open, regime_series, stock_data_all, surge=True))

    # === +错峰双腿 (tranching) ===
    # 将在循环中特殊处理
    register('rcfw_45_tranche', None)   # placeholder
    register('rcfw_35_tranche', None)   # placeholder

    # === 对照: 固定权重 ===
    # 使用 V5 的因子计算(fixed weights)作为对照
    sf_v5 = factors_v5.build_select_fn(stock_data_all, cal, aux, qroe)
    register('rcfw_fixedweight', lambda: run_one(
        sf_v5, df_close, df_open, regime_series, stock_data_all))

    # === V4.1 基线 ===
    # 复用 V5 回测脚本的基线: 不传 select_fn, 用旧因子路径
    register('v4fixed', lambda: run_one(
        None, df_close, df_open, regime_series, stock_data_all))
    register('v4fixed_surge', lambda: run_one(
        None, df_close, df_open, regime_series, stock_data_all, surge=True))

    # ── 运行 ──
    results = {}
    for name, fn in variants.items():
        if fn is None:
            continue  # tranche 占位, 后面处理
        print(f"\n===== {name} =====", flush=True)
        r = fn()
        m = metrics_from_values(r['values'])
        m['turnover'] = round(r['metrics'].get('annual_turnover_x', np.nan), 1)
        m['skipped'] = r['metrics'].get('skipped_trades')
        results[name] = m
        print(f"  年化 {m['ann']}%  回撤 {m['mdd']}%  夏普 {m['sharpe']}  换手 {m['turnover']}x")
        print(f"  分时段 {m['sub']}")

    # ── 错峰双腿(特殊处理: 需要跑两组) ──
    for tag, sf in [('rcfw_45_tranche', sf_45), ('rcfw_35_tranche', sf_35)]:
        if tag not in variants:
            continue
        print(f"\n===== {tag} =====", flush=True)
        r_me = run_one(sf, df_close, df_open, regime_series, stock_data_all, rebal=me_dates)
        r_mm = run_one(sf, df_close, df_open, regime_series, stock_data_all, rebal=mm_dates)
        m = tranche_combine(r_me, r_mm)
        results[tag] = m
        print(f"  年化 {m['ann']}%  回撤 {m['mdd']}%  夏普 {m['sharpe']}  换手 {m['turnover']}x")
        print(f"  分时段 {m['sub']}")

    # ── 保存 ──
    out = os.path.join(os.path.dirname(__file__), 'backtests', 'rcfw_v1_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")

    # ── 汇总表 ──
    print("\n" + "=" * 80)
    print("  全矩阵汇总")
    print("=" * 80)
    print(f"{'变体':<22s} {'年化':>7s} {'回撤':>7s} {'夏普':>6s} {'换手':>5s}  {'16-18':>7s} {'19-21':>7s} {'22-24':>7s} {'25-26':>7s}")
    print("-" * 80)
    for name in results:
        m = results[name]
        sub = m.get('sub', {})
        print(f"{name:<22s} {m['ann']:>6.1f}% {m['mdd']:>6.1f}% {m['sharpe']:>5.2f} {m['turnover']:>4.1f}x  "
              f"{sub.get('2016-18', 0):>+6.1f}% {sub.get('2019-21', 0):>+6.1f}% "
              f"{sub.get('2022-24', 0):>+6.1f}% {sub.get('2025-26', 0):>+6.1f}%")

    # ── 关键归因 ──
    print("\n── 关键归因 ──")
    if 'rcfw_45' in results and 'rcfw_fixedweight' in results:
        delta_sw = results['rcfw_45']['ann'] - results['rcfw_fixedweight']['ann']
        print(f"  制度切换效应(rcfw_45 - fixed): {delta_sw:+.1f}pp 年化")
    if 'rcfw_45_surge' in results and 'rcfw_45' in results:
        delta_surge = results['rcfw_45_surge']['ann'] - results['rcfw_45']['ann']
        print(f"  SURGE增量(rcfw_45): {delta_surge:+.1f}pp")
    if 'rcfw_35_surge' in results and 'rcfw_35' in results:
        delta_surge35 = results['rcfw_35_surge']['ann'] - results['rcfw_35']['ann']
        print(f"  SURGE增量(rcfw_35): {delta_surge35:+.1f}pp")
    if 'v4fixed' in results and 'rcfw_45' in results:
        delta_vs_v4 = results['rcfw_45']['ann'] - results['v4fixed']['ann']
        print(f"  vs V4.1基线(rcfw_45): {delta_vs_v4:+.1f}pp")
    if 'v4fixed_surge' in results and 'rcfw_45_surge' in results:
        delta_vs_v4s = results['rcfw_45_surge']['ann'] - results['v4fixed_surge']['ann']
        print(f"  vs V4.1+SURGE(rcfw_45_surge): {delta_vs_v4s:+.1f}pp")


if __name__ == '__main__':
    main()
