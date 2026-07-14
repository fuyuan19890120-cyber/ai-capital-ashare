#!/usr/bin/env python3
"""
数据刷新脚本
每月运行前执行一次，更新个股缓存到最新日期
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')
import akshare as ak
import pandas as pd


def main():
    print("刷新数据...")

    # 获取股票池
    csi300 = set(ak.index_stock_cons(symbol="000300")['品种代码'].apply(lambda x: str(x).zfill(6)))
    chinext = set(ak.index_stock_cons(symbol="399006")['品种代码'].apply(lambda x: str(x).zfill(6)))
    star50 = set(ak.index_stock_cons(symbol="000688")['品种代码'].apply(lambda x: str(x).zfill(6)))
    pool = sorted(csi300 | chinext | star50)

    CACHE_DIR = 'data/stocks'
    updated = 0

    for i, code in enumerate(pool):
        cache_path = f'{CACHE_DIR}/{code}.csv'
        if not os.path.exists(cache_path):
            continue

        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            last_date = df.index[-1]
            days_since = (pd.Timestamp.now() - last_date).days

            # 只刷新超过5天的数据
            if days_since <= 5:
                continue

            # 获取增量数据（Sina源）
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            new_df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")

            if new_df is not None and not new_df.empty:
                new_df['date'] = pd.to_datetime(new_df['date'])
                new_df = new_df.set_index('date').sort_index()
                keep = ["open", "high", "low", "close", "volume"]
                new_df = new_df[[c for c in keep if c in new_df.columns]]
                # 合并
                combined = pd.concat([df[~df.index.isin(new_df.index)], new_df])
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                combined.to_csv(cache_path)
                updated += 1

        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pool)} checked, {updated} updated")

        time.sleep(0.3)

    # 刷新指数数据
    print("刷新指数数据...")
    for idx_symbol in ["sh000300"]:
        try:
            cache_path = f'data/index_{idx_symbol}.csv'
            if os.path.exists(cache_path):
                os.remove(cache_path)
            idx_df = ak.stock_zh_index_daily(symbol=idx_symbol)
            if idx_df is not None and not idx_df.empty:
                idx_df = idx_df.rename(columns={"date": "date", "open": "open", "close": "close",
                                                "high": "high", "low": "low", "volume": "volume"})
                idx_df["date"] = pd.to_datetime(idx_df["date"])
                idx_df = idx_df.set_index("date").sort_index()
                keep = ["open", "high", "low", "close", "volume"]
                idx_df = idx_df[[c for c in keep if c in idx_df.columns]]
                idx_df.to_csv(cache_path)
                print(f"  {idx_symbol}: refreshed")
        except Exception as e:
            print(f"  {idx_symbol}: {e}")

    print(f"\nDone: {updated}/{len(pool)} stocks refreshed")


if __name__ == '__main__':
    main()
