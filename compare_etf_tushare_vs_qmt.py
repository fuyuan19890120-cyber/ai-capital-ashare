#!/usr/bin/env python3
"""
交叉比对: Tushare 前复权数据 vs QMT qmt_qfq.db 前复权数据
验证 close 是否一致 + volume 单位差异
"""
import os, sqlite3
import pandas as pd

QMT_DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
TS_DB = os.path.expanduser("~/ai-capital-ashare/data/etf_tushare.db")

qcon = sqlite3.connect(QMT_DB)
tcon = sqlite3.connect(TS_DB)
qmt = pd.read_sql("SELECT code, date, close, volume FROM daily", qcon)
tsd = pd.read_sql("SELECT code, date, close, volume FROM daily", tcon)
qcon.close(); tcon.close()

qmt_codes = set(qmt["code"]); ts_codes = set(tsd["code"])
print(f"QMT {len(qmt_codes)} 只, Tushare {len(ts_codes)} 只")
print(f"仅 QMT 有: {sorted(qmt_codes - ts_codes)}")
print(f"仅 Tushare 有: {sorted(ts_codes - qmt_codes)}")
print()

common = sorted(qmt_codes & ts_codes)
rows = []
for code in common:
    q = qmt[qmt["code"] == code].set_index("date")
    t = tsd[tsd["code"] == code].set_index("date")
    cd = q.index.intersection(t.index)
    if len(cd) == 0:
        continue
    qc, tc = q.loc[cd, "close"], t.loc[cd, "close"]
    rel = ((tc - qc).abs() / qc)
    max_err = rel.max()
    big = int((rel > 0.01).sum())       # 误差>1% 的天数
    big5 = int((rel > 0.005).sum())     # 误差>0.5% 的天数
    # volume 单位比 (tushare / qmt), 取中位数, 排除 0
    qv, tv = q.loc[cd, "volume"], t.loc[cd, "volume"]
    valid = (qv > 0) & (tv > 0)
    vol_ratio = float((tv[valid] / qv[valid]).median()) if valid.any() else float('nan')
    rows.append((code, len(cd), max_err, big5, big, vol_ratio))

df = pd.DataFrame(rows, columns=["code", "公共天数", "close最大误差", "误差>0.5%天数", "误差>1%天数", "vol比(ts/qmt)"])
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df.to_string(index=False))
print()
print(f"=== 汇总 (共 {len(df)} 只可比) ===")
print(f"close 最大误差中位数: {df['close最大误差'].median()*100:.3f}%")
print(f"close 最大误差 > 1% 的只数: {(df['close最大误差']>0.01).sum()}")
print(f"误差>1% 天数总计: {df['误差>1%天数'].sum()}")
print(f"vol 比 (ts/qmt) 中位数: {df['vol比(ts/qmt)'].median():.3f}  (≈100=QMT按份/ts按手, ≈1=同单位)")
