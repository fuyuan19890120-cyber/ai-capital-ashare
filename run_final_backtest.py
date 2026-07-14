#!/usr/bin/env python3
"""最终个股版回测 — 直接跑，出结果"""
import os, sys, warnings, json
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
from config import START_DATE
from src.stock_backtest import run_stock_backtest


def load_etf_close():
    etf_files = {
        'sh510300': 'etf_510300.csv', 'sh510500': 'etf_510500.csv',
        'sz159915': 'etf_159915.csv', 'sh588000': 'etf_588000.csv',
        'sh511010': 'etf_511010.csv', 'sh511220': 'etf_511220.csv',
        'sh518880': 'etf_518880.csv', 'sh511880': 'etf_511880.csv',
    }
    etf_data = {}
    for code, fname in etf_files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if 'close' in df.columns and not df.empty:
                etf_data[code] = df['close']
    return pd.DataFrame(etf_data).sort_index().dropna()


def load_index():
    path = os.path.join(DATA_DIR, 'index_sh000300.csv')
    if not os.path.exists(path): return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_stocks(limit=400):
    d = os.path.join(DATA_DIR, "stocks")
    if not os.path.exists(d): return {}
    result = {}
    for f in sorted(os.listdir(d))[:limit]:
        if not f.endswith('.csv'): continue
        try:
            df = pd.read_csv(os.path.join(d, f), index_col=0, parse_dates=True)
            if len(df) > 250: result[f.replace('.csv', '')] = df
        except: pass
    return result


def compute_regime(benchmark_close):
    sma50 = benchmark_close.rolling(50).mean()
    sma250 = benchmark_close.rolling(250).mean()
    scores = pd.Series(index=benchmark_close.index, dtype=float)
    for i in range(252, len(benchmark_close)):
        p = benchmark_close.iloc[i]
        dev = (p - sma250.iloc[i]) / sma250.iloc[i]
        trend = 0.5 + 0.5 * np.tanh(dev * 10)
        golden = 1.0 if sma50.iloc[i] > sma250.iloc[i] else 0.0
        scores.iloc[i] = 0.6 * trend + 0.4 * golden
    return scores.dropna()


def main():
    print("=" * 55)
    print("  最终个股版回测")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # 数据
    print("\n加载数据...")
    df_close = load_etf_close()
    index_df = load_index()
    stock_data = load_stocks(400)

    print(f"  ETF: {df_close.index[0].date()} ~ {df_close.index[-1].date()} ({len(df_close)} 行)")
    print(f"  沪深300: {index_df.index[0].date()} ~ {index_df.index[-1].date()} ({len(index_df)} 行)")
    print(f"  个股: {len(stock_data)} 只")

    # 制度
    regime_series = compute_regime(index_df['close'])
    print(f"  制度序列: {regime_series.index[0].date()} ~ {regime_series.index[-1].date()} ({len(regime_series)} 期)")

    # 跑
    result = run_stock_backtest(
        df_close, regime_series, stock_data, top_n=15, verbose=True
    )

    m = result['metrics']
    print("\n" + "=" * 55)
    print("  回测结果")
    print("=" * 55)
    print(f"  区间: {START_DATE} ~ 2026-07-13")
    print(f"  股票池: {len(stock_data)} 只, Top-15 等权")
    print(f"  调仓次数: {len(result['rebalance_dates'])} 次")
    print()
    for k, v in m.items():
        if k == 'annual_returns':
            continue
        print(f"  {k}: {v}")

    if 'annual_returns' in m:
        print(f"\n  年度收益:")
        ar = m['annual_returns']
        if hasattr(ar, 'items'):
            for yr, ret in ar.items():
                bar = '🟢' if ret > 0 else '🔴'
                print(f"    {yr}: {bar} {ret:+.1f}%")

    # 保存
    out = os.path.join(os.path.dirname(__file__), "backtests", "final_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'metrics': {k: v for k, v in m.items() if k != 'annual_returns'},
            'annual_returns': dict(m['annual_returns']) if hasattr(m['annual_returns'], 'items') else {},
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果: {out}")


if __name__ == '__main__':
    main()
