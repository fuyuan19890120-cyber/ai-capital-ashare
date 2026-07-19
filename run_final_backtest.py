#!/usr/bin/env python3
"""最终个股版回测 — 直接跑，出结果

2026-07-18 审计修复版, 三种模式用于逐项归因:
  --mode legacy    复现旧版(前400只宇宙+信号日收盘成交+无印花税+停牌计0) — 仅对照
  --mode universe  只修宇宙(全部缓存个股), 其余同旧版 — 隔离宇宙bug的影响
  --mode fixed     全修复(默认): 全宇宙 + T+1开盘成交 + 印花税 + 涨跌停约束 + 停牌估值修复
"""
import os, sys, warnings, json, argparse
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
from config import START_DATE
from src.stock_backtest import run_stock_backtest

ETF_FILES = {
    'sh510300': 'etf_510300.csv', 'sh510500': 'etf_510500.csv',
    'sz159915': 'etf_159915.csv', 'sh588000': 'etf_588000.csv',
    'sh511010': 'etf_511010.csv', 'sh511220': 'etf_511220.csv',
    'sh518880': 'etf_518880.csv', 'sh511880': 'etf_511880.csv',
}


def load_etf_prices(calendar_fix=True):
    """返回 (close宽表, open宽表)

    calendar_fix=True: dropna(how='all') 保留完整日历。
    False 复刻旧行为 dropna() —— etf_588000(2020-11起)存在时整个回测被静默截断,
    这是审计新发现的 bug, 仅供 legacy 模式忠实复现。
    """
    closes, opens = {}, {}
    for code, fname in ETF_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if 'close' in df.columns and not df.empty:
                closes[code] = df['close']
                if 'open' in df.columns:
                    opens[code] = df['open']
    wide = pd.DataFrame(closes).sort_index()
    df_close = wide.dropna(how='all') if calendar_fix else wide.dropna()
    df_open = pd.DataFrame(opens).sort_index().reindex(df_close.index)
    return df_close, df_open


def load_index():
    path = os.path.join(DATA_DIR, 'index_sh000300.csv')
    if not os.path.exists(path): return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_stocks(limit=None, index_only=False):
    """limit=None 加载全部缓存个股(修复: 旧版 limit=400 按文件名截断, 无沪市/科创票)

    index_only=True: 仅加载 CSI300+创业板指+科创50 成分股(与实盘 signal_generator 一致)。
    V4.2 审计修复(2026-07-19): 回测必须与实盘使用相同的股票池。
    """
    d = os.path.join(DATA_DIR, "stocks")
    if not os.path.exists(d): return {}

    # 获取指数成分股白名单(与 signal_generator.load_stock_universe 一致)
    whitelist = None
    if index_only:
        try:
            import akshare as ak
            from src.stock_data import get_csi300_constituents
            csi300 = set(get_csi300_constituents())
            chinext = set(ak.index_stock_cons(symbol="399006")['品种代码'].apply(lambda x: str(x).zfill(6)))
            star50 = set(ak.index_stock_cons(symbol="000688")['品种代码'].apply(lambda x: str(x).zfill(6)))
            whitelist = csi300 | chinext | star50
        except Exception:
            pass

    result = {}
    files = sorted(os.listdir(d))
    if limit:
        files = files[:limit]
    for f in files:
        if not f.endswith('.csv'): continue
        code = f.replace('.csv', '')
        if whitelist is not None and code not in whitelist:
            continue
        try:
            df = pd.read_csv(os.path.join(d, f), index_col=0, parse_dates=True)
            if len(df) > 250: result[code] = df
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


MODES = {
    # (calendar_fix, limit, execution, stamp_duty, ffill_valuation, index_only)
    'legacy':   (False, 400,  'same_close', False, False, False),  # 忠实复刻旧代码(被588000截断到2020-11)
    'calendar': (True,  400,  'same_close', False, False, False),  # +日历修复
    'universe': (True,  None, 'same_close', False, False, False),  # +宇宙修复(全部缓存个股)
    'fixed':    (True,  None, 'next_open',  True,  True,  True),   # V4.2审计修复: index_only=True对齐实盘宇宙
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=list(MODES), default='fixed')
    args = ap.parse_args()
    calendar_fix, limit, execution, stamp, ffill, index_only = MODES[args.mode]

    print("=" * 55)
    print(f"  最终个股版回测  [mode={args.mode}]")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    # 数据
    print("\n加载数据...")
    df_close, df_open = load_etf_prices(calendar_fix)
    index_df = load_index()
    stock_data = load_stocks(limit, index_only=index_only)

    print(f"  ETF: {df_close.index[0].date()} ~ {df_close.index[-1].date()} ({len(df_close)} 行)")
    print(f"  沪深300: {index_df.index[0].date()} ~ {index_df.index[-1].date()} ({len(index_df)} 行)")
    print(f"  个股: {len(stock_data)} 只")

    # 制度
    regime_series = compute_regime(index_df['close'])
    print(f"  制度序列: {regime_series.index[0].date()} ~ {regime_series.index[-1].date()} ({len(regime_series)} 期)")

    # 跑
    result = run_stock_backtest(
        df_close, regime_series, stock_data, top_n=15, verbose=True,
        execution=execution, stamp_duty=stamp, ffill_valuation=ffill, df_open=df_open,
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
    out = os.path.join(os.path.dirname(__file__), "backtests", f"final_result_{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'mode': args.mode,
            'metrics': {k: v for k, v in m.items() if k != 'annual_returns'},
            'annual_returns': dict(m['annual_returns']) if hasattr(m['annual_returns'], 'items') else {},
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  结果: {out}")


if __name__ == '__main__':
    main()
