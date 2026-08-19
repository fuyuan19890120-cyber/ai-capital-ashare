#!/usr/bin/env python3
"""
ETF-R4 数据源: Tushare 全量拉取 + pct_chg 重构前复权 (替代 QMT, 免 Windows)

为什么用 pct_chg 而非 fund_adj:
  tushare fund_adj(复权因子) 对场内 ETF 的份额折算处理不准(实测 -71%~+84% 偏差),
  而 pct_chg(涨跌幅) 已正确处理拆分/折算(实测: 拆分日 close 跳空 -48% 但 pct_chg 正常 +2%)。
  前复权 = 锚定最新收盘价, 用 pct_chg 从最新往前递推历史价, 收益率与真实回报一致。

用法: 每月末全量重跑(前复权锚点随最新价漂移, 不宜增量追加, 全量重拉仅 ~1-2 分钟)。
输出: ~/ai-capital-ashare/data/etf_tushare.db  (daily 表, 与 qmt_qfq.db 同 schema)
"""
import os, sys, time, warnings, sqlite3
from datetime import datetime
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import tushare as ts

TOKEN = "2d7e92d4cda95afe932389b4bee184fd9fabcdf8851ff79d68fe933b"  # 可改为 os.environ.get("TUSHARE_TOKEN")
DB_OUT = os.path.expanduser("~/ai-capital-ashare/data/etf_tushare.db")
START = "20150101"

# 43 只 ETF = 4 宽基 + 3 防御 + 36 行业 (与 paper_trade_r4.py 的 ALL_ETFS 一致)
BROAD = ["510300.SH", "510500.SH", "159915.SZ", "588000.SH"]
DEFENSE = ["511010.SH", "518880.SH", "511880.SH"]
SECTOR_ETFS = [
    "512880.SH","512800.SH","512070.SH","512000.SH","512010.SH","512170.SH","159992.SZ",
    "159928.SZ","512690.SH","512480.SH","159995.SZ","512760.SH","512660.SH","512710.SH",
    "515030.SH","515790.SH","516160.SH","512400.SH","515210.SH","159870.SZ",
    "512200.SH","512980.SH","159869.SZ","159939.SZ","515000.SH",
    "515050.SH","515880.SH","159967.SZ","562500.SH","159819.SZ","159611.SZ",
    "159865.SZ",
    "159227.SZ","159206.SZ",  # 商业航天: 航空航天ETF华夏 + 卫星ETF永赢
    "159770.SZ","588730.SH",  # 机器人备选(天弘) + 人工智能备选(科创AI易方达)
    "159857.SZ","516020.SH","561560.SH",  # 光伏备选(天弘)+化工备选(华宝)+电力备选(华泰柏瑞)
    "560860.SH",  # 有色备选(万家工业有色, 4.5亿流动性)
    "159707.SZ",  # 房地产备选(华宝中证800地产, 0.7亿流动性)
]
ALL_ETFS = BROAD + DEFENSE + SECTOR_ETFS


def rebuild_qfq(df):
    """用 pct_chg 重构前复权价(锚定最新收盘价), open/high/low 按同比例缩放"""
    df = df.sort_values("trade_date").reset_index(drop=True)
    n = len(df)
    close = df["close"].values.astype(float)
    pct = df["pct_chg"].values.astype(float)

    qfq = np.zeros(n)
    qfq[n - 1] = close[n - 1]  # 锚定最新不复权收盘价
    for i in range(n - 2, -1, -1):
        r = pct[i + 1] / 100.0 if not np.isnan(pct[i + 1]) else 0.0
        qfq[i] = qfq[i + 1] / (1.0 + r)

    factor = qfq / close  # 复权比例因子 (每股)
    for col in ["open", "high", "low"]:
        df[col] = df[col].values.astype(float) * factor
    df["close"] = qfq
    return df


def fetch_one(pro, code, end):
    daily = pro.fund_daily(ts_code=code, start_date=START, end_date=end)
    if daily is None or daily.empty:
        return None
    daily = rebuild_qfq(daily)
    out = pd.DataFrame({
        "code": code,
        "date": daily["trade_date"].astype(str),
        "open": daily["open"], "high": daily["high"],
        "low": daily["low"], "close": daily["close"],
        "volume": daily["vol"],  # tushare vol 单位: 手 (与 QMT 一致, 实测 ratio=1)
    })
    return out[out["close"] > 0]


def main():
    ts.set_token(TOKEN)
    pro = ts.pro_api()
    end = datetime.now().strftime("%Y%m%d")

    os.makedirs(os.path.dirname(DB_OUT), exist_ok=True)
    # 写临时库, 完成后原子替换 (崩溃不会留空库/半库)
    DB_TMP = DB_OUT + ".tmp"
    if os.path.exists(DB_TMP):
        os.remove(DB_TMP)
    con = sqlite3.connect(DB_TMP)
    con.execute("CREATE TABLE daily (code TEXT, date TEXT, open REAL, high REAL, "
                "low REAL, close REAL, volume REAL)")
    con.commit()

    ok, fail = [], []
    total_rows = 0
    print(f"开始全量拉取 {len(ALL_ETFS)} 只 ETF (tushare fund_daily + pct_chg 重构前复权) ...")
    for i, code in enumerate(ALL_ETFS, 1):
        df = None
        for attempt in range(3):
            try:
                df = fetch_one(pro, code, end)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [{i}/{len(ALL_ETFS)}] {code} 失败: {str(e)[:80]}")
                time.sleep(1.5)
        if df is None or df.empty:
            fail.append(code)
            continue
        df.to_sql("daily", con, if_exists="append", index=False)
        rows = len(df)
        total_rows += rows
        ok.append(code)
        rng = f"{df['date'].iloc[0]}~{df['date'].iloc[-1]}"
        print(f"  [{i}/{len(ALL_ETFS)}] {code}  {rows:5d} 行  {rng}")
        time.sleep(0.4)

    con.commit()
    con.close()

    # 原子替换: 全部成功才替换旧库; 部分失败保留旧库(本次数据不写入, 避免静默缺 ETF)
    if not fail:
        os.replace(DB_TMP, DB_OUT)
        print(f"\n✅ 完成: 成功 {len(ok)}/{len(ALL_ETFS)}, 总 {total_rows} 行, 已替换 {DB_OUT}")
    else:
        if os.path.exists(DB_TMP):
            os.remove(DB_TMP)
        print(f"\n⚠️ 成功 {len(ok)}/{len(ALL_ETFS)}, 失败 {len(fail)} 只: {fail}")
        print(f"   保留旧库 {DB_OUT} 未动(避免缺 ETF), 请重跑")


if __name__ == "__main__":
    main()
