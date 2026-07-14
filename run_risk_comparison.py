#!/usr/bin/env python3
"""
风控对比回测 — 基线 vs 四层风控（使用本地缓存数据）

用法：python run_risk_comparison.py
"""
import os, sys, warnings, json
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 使用原始目录的 data/（因为 worktree 没有 data 缓存）
DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")

from config import START_DATE
from src.risk_backtest import run_risk_backtest


def load_etf_close():
    """从本地缓存加载 ETF 对齐价格"""
    etf_files = {
        'sh510300': 'etf_510300.csv',
        'sh510500': 'etf_510500.csv',
        'sz159915': 'etf_159915.csv',
        'sh588000': 'etf_588000.csv',
        'sh511010': 'etf_511010.csv',
        'sh511220': 'etf_511220.csv',
        'sh518880': 'etf_518880.csv',
        'sh511880': 'etf_511880.csv',
    }

    etf_data = {}
    for code, fname in etf_files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if 'close' in df.columns and not df.empty:
                etf_data[code] = df['close']

    if not etf_data:
        return None
    df_close = pd.DataFrame(etf_data).sort_index().dropna()
    return df_close


def load_index_data():
    """加载沪深300缓存"""
    path = os.path.join(DATA_DIR, 'index_sh000300.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if 'close' not in df.columns:
        return None
    return df


def load_cached_stocks(limit=None):
    """从本地缓存加载个股数据"""
    cache_dir = os.path.join(DATA_DIR, "stocks")
    if not os.path.exists(cache_dir):
        return {}
    files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.csv')])[:limit]
    stock_data = {}
    for f in files:
        code = f.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(cache_dir, f), index_col=0, parse_dates=True)
            if len(df) > 250:
                stock_data[code] = df
        except:
            pass
    return stock_data


def compute_regime_series(benchmark_close):
    """计算制度时间序列 (SMA250 + 金叉)"""
    sma50 = benchmark_close.rolling(50).mean()
    sma250 = benchmark_close.rolling(250).mean()
    regime_scores = pd.Series(index=benchmark_close.index, dtype=float)
    for i in range(252, len(benchmark_close)):
        p = benchmark_close.iloc[i]
        s250 = sma250.iloc[i]
        s50 = sma50.iloc[i]
        dev = (p - s250) / s250
        trend = 0.5 + 0.5 * np.tanh(dev * 10)
        golden = 1.0 if s50 > s250 else 0.0
        regime_scores.iloc[i] = 0.6 * trend + 0.4 * golden
    return regime_scores.dropna()


def main():
    print("=" * 60)
    print("  风控对比回测（本地缓存数据）")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. 加载 ETF 数据
    print("\n[1/4] 加载 ETF 数据...")
    df_close = load_etf_close()
    if df_close is None:
        print("❌ ETF 数据加载失败")
        return
    print(f"  ETF: {len(df_close)} 行 x {len(df_close.columns)} 列")
    print(f"  区间: {df_close.index[0].date()} ~ {df_close.index[-1].date()}")

    # 2. 加载指数数据
    print("\n[2/4] 加载沪深300...")
    index_df = load_index_data()
    if index_df is None or 'close' not in index_df.columns:
        print("❌ 指数数据加载失败")
        return
    print(f"  沪深300: {len(index_df)} 行")

    # 3. 加载个股
    print("\n[3/4] 加载个股数据...")
    stock_data = load_cached_stocks(limit=400)
    print(f"  个股: {len(stock_data)} 只")

    # 4. 跑回测
    print("\n[4/4] 计算制度序列...")
    regime_series = compute_regime_series(index_df['close'])
    print(f"  制度序列: {len(regime_series)} 期")
    print(f"  有制度分的时间: {regime_series.index[0].date()} ~ {regime_series.index[-1].date()}")

    # 4a. 基线
    print("\n" + "=" * 55)
    print("  第 1/2 轮：基线（无风控）")
    print("=" * 55)
    baseline = run_risk_backtest(
        df_close, regime_series, stock_data, top_n=15, enable_risk=False
    )

    # 4b. 风控增强
    print("\n" + "=" * 55)
    print("  第 2/2 轮：风控增强版")
    print("=" * 55)
    risk_mgd = run_risk_backtest(
        df_close, regime_series, stock_data, top_n=15, enable_risk=True
    )

    # 5. 打印
    bm = baseline['metrics']
    rm = risk_mgd['metrics']

    print("\n")
    print("=" * 70)
    print("  📊 回测对比结果")
    print("=" * 70)
    print(f"  区间: {START_DATE} ~ 2026-07-13")
    print(f"  股票池: {len(stock_data)} 只, Top-15 等权")
    print()
    print(f"  {'指标':<14s} {'基线(无风控)':>14s} {'风控增强版':>14s} {'变化':>10s}")
    print(f"  {'-'*54}")

    def fmt_delta(new, old, precision='.1f', invert=False):
        d = new - old
        if invert:
            better = d < 0
        else:
            better = d > 0
        symbol = '✅' if better else '⚠️'
        return f"{symbol} {d:+{precision}}"

    print(f"  {'年化收益':<14s} {bm['annual_return']:>13.1f}% {rm['annual_return']:>13.1f}% {fmt_delta(rm['annual_return'], bm['annual_return']):>10s}")
    print(f"  {'最大回撤':<14s} {bm['max_drawdown']:>13.1f}% {rm['max_drawdown']:>13.1f}% {fmt_delta(rm['max_drawdown'], bm['max_drawdown'], invert=True):>10s}")
    print(f"  {'夏普比率':<14s} {bm['sharpe_ratio']:>13.2f} {rm['sharpe_ratio']:>13.2f} {fmt_delta(rm['sharpe_ratio'], bm['sharpe_ratio'], '.2f'):>10s}")
    print(f"  {'总收益':<14s} {bm['total_return']:>13.1f}% {rm['total_return']:>13.1f}% {fmt_delta(rm['total_return'], bm['total_return']):>10s}")
    print(f"  {'年化波动率':<14s} {bm['volatility']:>13.1f}% {rm['volatility']:>13.1f}% {fmt_delta(rm['volatility'], bm['volatility'], invert=True):>10s}")

    # 风控事件
    rs = risk_mgd.get('risk_stats', {})
    if rs:
        print(f"\n  🛡️ 风控事件统计:")
        print(f"    止损触发: {rs.get('stop_loss_count', 0)} 次")
        print(f"    回撤事件: {rs.get('drawdown_event_count', 0)} 次")
        print(f"    板块过滤: {rs.get('sector_filter_count', 0)} 次")

        if rs.get('drawdown_events'):
            print(f"\n  回撤事件明细:")
            for e in rs['drawdown_events']:
                print(f"    {e['date']} | 回撤 {e['drawdown']}% | {e['action']}")

        if rs.get('stop_loss_samples'):
            print(f"\n  止损示例（前 5 次）:")
            for e in rs['stop_loss_samples'][:5]:
                print(f"    {str(e['date'])[:10]}  {e['code']}  入场{e['entry']} → 退出{e['exit']}  ({e['loss']}%)")

    # 年度对比
    ba = bm.get('annual_returns', {})
    ra = rm.get('annual_returns', {})
    if ba or ra:
        print(f"\n  {'年度收益对比':^54s}")
        print(f"  {'年份':<8s} {'基线':>12s} {'风控增强':>12s} {'差值':>12s}")
        print(f"  {'-'*44}")
        for yr in sorted(set(list(ba.keys()) + list(ra.keys()))):
            b = ba.get(yr, 0)
            r = ra.get(yr, 0)
            d = r - b
            print(f"  {yr:<8d} {b:>11.1f}% {r:>11.1f}% {d:>+11.1f}%")

    print()
    print("=" * 70)

    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), "backtests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "risk_comparison.json")
    with open(out_path, 'w') as f:
        json.dump({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'baseline': bm,
            'risk_managed': rm,
            'risk_stats': rs,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"  结果保存: {out_path}")


if __name__ == '__main__':
    main()
