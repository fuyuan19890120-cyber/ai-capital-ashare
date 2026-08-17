#!/usr/bin/env python3
"""
GitHub Actions 用: 拉取 ETF-R4 43 只 ETF 最近 120 天日线前复权(东财)
输出 CSV 到 data/etf_incremental/etf_latest.csv, 由 workflow commit 回仓库。

背景: 本机直连东财行情 API 被反爬封禁(本机IP+代理IP均被拒),
      GitHub Actions 美国节点可直连, 故增量数据在 CI 上拉取。
本机: merge_etf_incremental.py 读取本 CSV, 重叠日校准后追加到 qmt_qfq.db。
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "etf_incremental", "etf_latest.csv")
DAYS = 120  # 拉最近 120 天, 确保覆盖一个月以上的停更窗口

# 43 只 ETF = 4 宽基 + 3 防御 + 36 行业 (与 paper_trade_r4.py 的 ALL_ETFS 一致)
BROAD = ["510300.SH", "510500.SH", "159915.SZ", "588000.SH"]
DEFENSE = ["511010.SH", "518880.SH", "511880.SH"]
SECTOR_ETFS = [
    "512880.SH","512800.SH","512070.SH","512000.SH","512010.SH","512170.SH","159992.SZ",
    "159928.SZ","512690.SH","512480.SH","159995.SZ","512760.SH","512660.SH","512710.SH",
    "515030.SH","515790.SH","516160.SH","512400.SH","515210.SH","159870.SZ",
    "512200.SH","516750.SH","512980.SH","159869.SZ","159939.SZ","515000.SH","512720.SH",
    "515050.SH","515880.SH","159967.SZ","562500.SH","159819.SZ","159611.SZ","512580.SH",
    "516670.SH","159865.SZ",
]
ALL_ETFS = BROAD + DEFENSE + SECTOR_ETFS


def fetch_one(code, start, end):
    sym = code.replace(".SH", "").replace(".SZ", "")
    df = ak.fund_etf_hist_em(symbol=sym, period="daily",
                             start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return None
    out = df[["日期", "开盘", "收盘", "最高", "最低", "成交量"]].rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    })
    out["date"] = out["date"].astype(str).str.replace("-", "")
    out["code"] = code
    out = out[out["close"] > 0]
    return out[["code", "date", "open", "high", "low", "close", "volume"]]


def main():
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=DAYS)).strftime("%Y%m%d")
    print(f"拉取区间 {start} ~ {end}  ({len(ALL_ETFS)} 只 ETF)")

    frames, fail = [], []
    for i, code in enumerate(ALL_ETFS, 1):
        df = None
        for attempt in range(3):
            try:
                df = fetch_one(code, start, end)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [{i}/{len(ALL_ETFS)}] {code} 失败: {str(e)[:70]}")
                time.sleep(2)
        if df is None or df.empty:
            fail.append(code)
            continue
        frames.append(df)
        time.sleep(0.4)

    if not frames:
        print("❌ 全部失败, 退出")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_df.to_csv(OUT_CSV, index=False)
    print(f"✅ 成功 {len(ALL_ETFS)-len(fail)}/{len(ALL_ETFS)}, 输出 {len(all_df)} 行 -> {OUT_CSV}")
    if fail:
        print(f"失败 {len(fail)} 只: {fail}")


if __name__ == "__main__":
    main()
