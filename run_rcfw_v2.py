#!/usr/bin/env python3
"""
RC-FW v2: V4.1 简单因子 + 制度切换权重 — 全试验矩阵

因子定义 = V4.1 原版(同 stock_data.compute_stock_factors)
唯一变量 = 因子权重按制度分档

变体:
  rcfw_v2             制度切换权重(进攻/均衡/防守)
  rcfw_v2_surge        +SURGE 日频加速
  rcfw_v2_tranche      +错峰双腿(15日/月末)
  rcfw_v2_fixed        NEUTRAL权重固定(应逼近V4.1, 校准用)
  v4fixed / v4fixed_surge   V4.1 基线

敏感性(在 rcfw_v2 上单点扰动):
  rcfw_v2_mom60       RISKON动量60%(更激进)
  rcfw_v2_mom45       RISKON动量45%(中间档)
  rcfw_v2_lv25        RISKOFF低波25%+动量25%(更保守的防守)

用法: venv/bin/python run_rcfw_v2.py [--variant all]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src.factors_rcfw_v2 import build_v4_select_fn, REGIME_WEIGHTS_V2
from run_final_backtest import load_etf_prices, load_index, load_stocks, compute_regime
from run_v5_backtest import (
    month_end_dates, mid_month_dates, build_surge_lock, metrics_from_values,
)

SURGE_KWARGS = dict(lookback=15, lock_days=21, s30_min=0.70, base_min=0.15)


def run_one(select_fn, df_close, df_open, regime_series, stock_data,
            rebal=None, surge=False, top_n=15):
    """单次回测封装"""
    cal = df_close.index[df_close.index >= START_DATE]
    rd = rebal if rebal is not None else month_end_dates(cal)
    forced = None
    if surge:
        extra, forced = build_surge_lock(
            load_index()['close'], df_close, regime_series, cal, rd, **SURGE_KWARGS)
        rd = pd.DatetimeIndex(sorted(set(rd) | set(extra)))
    return run_stock_backtest(
        df_close, regime_series, stock_data, top_n=top_n, verbose=False,
        execution='next_open', stamp_duty=True, ffill_valuation=True,
        df_open=df_open, rebalance_dates=rd, select_fn=select_fn,
        forced_regime=forced,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all')
    args = ap.parse_args()

    # ── 数据 ──
    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    stock_data = load_stocks(None)
    regime_series = compute_regime(index_df['close'])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = month_end_dates(cal)
    mm_dates = mid_month_dates(cal)

    print(f"个股 {len(stock_data)} 只 | 日历 {cal[0].date()}~{cal[-1].date()}")
    print(f"月度调仓 {len(me_dates)} 次")

    # ── select_fn 构建 ──
    sf = build_v4_select_fn(stock_data, cal)  # 默认参数

    # 敏感性变体: 修改 REGIME_WEIGHTS_V2
    import copy
    w_mom60 = copy.deepcopy(REGIME_WEIGHTS_V2)
    w_mom60["RISKON"] = {"low_vol": 0.10, "value": 0.05, "quality": 0.25, "momentum_6m": 0.60}
    w_mom45 = copy.deepcopy(REGIME_WEIGHTS_V2)
    w_mom45["RISKON"] = {"low_vol": 0.15, "value": 0.10, "quality": 0.30, "momentum_6m": 0.45}

    # ── 试验矩阵 ──
    variants = {}

    def register(name, fn):
        if args.variant == 'all' or name in args.variant.split(','):
            variants[name] = fn

    register('rcfw_v2', lambda: run_one(sf, df_close, df_open, regime_series, stock_data))
    register('rcfw_v2_surge', lambda: run_one(
        sf, df_close, df_open, regime_series, stock_data, surge=True))

    # 校准: NEUTRAL固定权重应≈V4.1
    # 修改 select_fn 使其忽略 regime, 永远用 NEUTRAL
    sf_fixed = build_v4_select_fn(stock_data, cal)
    # hack: 让所有 regime 都映射到 NEUTRAL 权重
    import src.factors_rcfw_v2 as fv2
    orig_w = fv2.REGIME_WEIGHTS_V2
    neutral_only = {"RISKON": orig_w["NEUTRAL"], "NEUTRAL": orig_w["NEUTRAL"],
                    "RISKOFF": orig_w["NEUTRAL"], "CRISIS": orig_w["NEUTRAL"]}
    fv2.REGIME_WEIGHTS_V2 = neutral_only
    sf_nonly = build_v4_select_fn(stock_data, cal)
    fv2.REGIME_WEIGHTS_V2 = orig_w  # restore
    register('rcfw_v2_fixed', lambda: run_one(
        sf_nonly, df_close, df_open, regime_series, stock_data))

    # 敏感性
    fv2.REGIME_WEIGHTS_V2 = w_mom60
    sf_m60 = build_v4_select_fn(stock_data, cal)
    fv2.REGIME_WEIGHTS_V2 = orig_w
    register('rcfw_v2_mom60', lambda: run_one(
        sf_m60, df_close, df_open, regime_series, stock_data))

    fv2.REGIME_WEIGHTS_V2 = w_mom45
    sf_m45 = build_v4_select_fn(stock_data, cal)
    fv2.REGIME_WEIGHTS_V2 = orig_w
    register('rcfw_v2_mom45', lambda: run_one(
        sf_m45, df_close, df_open, regime_series, stock_data))

    # V4.1 基线
    register('v4fixed', lambda: run_one(
        None, df_close, df_open, regime_series, stock_data))
    register('v4fixed_surge', lambda: run_one(
        None, df_close, df_open, regime_series, stock_data, surge=True))

    # ── 运行 ──
    results = {}
    for name, fn in variants.items():
        print(f"\n===== {name} =====", flush=True)
        r = fn()
        m = metrics_from_values(r['values'])
        m['turnover'] = round(r['metrics'].get('annual_turnover_x', np.nan), 1)
        results[name] = m
        print(f"  年化 {m['ann']}%  回撤 {m['mdd']}%  夏普 {m['sharpe']}  换手 {m['turnover']}x")
        print(f"  分时段 {m['sub']}")

    # ── 错峰双腿 ──
    for tag in ['rcfw_v2_tranche']:
        if tag not in variants:
            continue
        print(f"\n===== {tag} =====", flush=True)
        r_me = run_one(sf, df_close, df_open, regime_series, stock_data, rebal=me_dates)
        r_mm = run_one(sf, df_close, df_open, regime_series, stock_data, rebal=mm_dates)
        nav_a = r_me['values']['value'] / r_me['values']['value'].iloc[0]
        nav_b = r_mm['values']['value'] / r_mm['values']['value'].iloc[0]
        mix = pd.DataFrame({'value': (nav_a + nav_b) / 2 * 1e6})
        m = metrics_from_values(mix)
        m['turnover'] = round(
            (r_me['metrics']['annual_turnover_x'] + r_mm['metrics']['annual_turnover_x']) / 2, 1)
        results[tag] = m
        print(f"  年化 {m['ann']}%  回撤 {m['mdd']}%  夏普 {m['sharpe']}  换手 {m['turnover']}x")
        print(f"  分时段 {m['sub']}")

    # ── 保存 ──
    out = os.path.join(os.path.dirname(__file__), 'backtests', 'rcfw_v2_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")

    # ── 汇总 ──
    print("\n" + "=" * 85)
    print("  RC-FW v2 全矩阵汇总")
    print("=" * 85)
    print(f"{'变体':<24s} {'年化':>7s} {'回撤':>7s} {'夏普':>6s} {'换手':>5s}  {'16-18':>7s} {'19-21':>7s} {'22-24':>7s} {'25-26':>7s}")
    print("-" * 85)
    for name in results:
        m = results[name]
        sub = m.get('sub', {})
        print(f"{name:<24s} {m['ann']:>6.1f}% {m['mdd']:>6.1f}% {m['sharpe']:>5.2f} {m['turnover']:>4.1f}x  "
              f"{sub.get('2016-18', 0):>+6.1f}% {sub.get('2019-21', 0):>+6.1f}% "
              f"{sub.get('2022-24', 0):>+6.1f}% {sub.get('2025-26', 0):>+6.1f}%")

    # ── 归因 ──
    print("\n── 关键归因 ──")
    if 'rcfw_v2' in results and 'rcfw_v2_fixed' in results:
        d = results['rcfw_v2']['ann'] - results['rcfw_v2_fixed']['ann']
        print(f"  制度切换效应(v2 - fixed_neutral): {d:+.1f}pp 年化")
    if 'rcfw_v2' in results and 'v4fixed' in results:
        d = results['rcfw_v2']['ann'] - results['v4fixed']['ann']
        print(f"  vs V4.1(rcfw_v2): {d:+.1f}pp")
    if 'rcfw_v2_surge' in results and 'v4fixed_surge' in results:
        d = results['rcfw_v2_surge']['ann'] - results['v4fixed_surge']['ann']
        print(f"  vs V4.1+SURGE(rcfw_v2_surge): {d:+.1f}pp")
    if 'rcfw_v2_fixed' in results and 'v4fixed' in results:
        d = results['rcfw_v2_fixed']['ann'] - results['v4fixed']['ann']
        print(f"  校准偏差(fixed_neutral - V4.1): {d:+.1f}pp (应接近0)")
    if 'rcfw_v2_surge' in results and 'rcfw_v2' in results:
        d = results['rcfw_v2_surge']['ann'] - results['rcfw_v2']['ann']
        print(f"  SURGE增量: {d:+.1f}pp")


if __name__ == '__main__':
    main()
