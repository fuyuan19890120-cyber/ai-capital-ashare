"""
QMT 数据库统一适配层。
从 qmt_qfq.db (ETF前复权) + qmt_full.db (个股) 加载日线数据，供回测策略使用。

功能：
- 代码格式双向转换（策略格式 ↔ QMT 格式）
- ETF 批量加载（前复权数据，自动过滤前视偏误 volume==0）
- 基准指数加载
- 个股数据加载（用于 ETF-R1 的 V4.1 因子投票）
"""
import os
import sqlite3
import pandas as pd
import numpy as np

# 两个数据库：ETF 使用前复权库，个股使用全量库
DB_ETF_PATH = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
DB_STOCK_PATH = os.path.expanduser("~/ai-capital-ashare/data/qmt_full.db")

# ── 代码格式转换 ──────────────────────────────────────────────

def to_qmt_code(strategy_code: str) -> str:
    """
    sh512880 → 512880.SH
    sz159915 → 159915.SZ
    512880   → 512880.SH (bare code, 非 000/002/300/600 开头默认沪市)
    """
    code = str(strategy_code).strip()
    if "." in code:
        return code  # already QMT format
    if code.startswith("sh"):
        return code[2:] + ".SH"
    if code.startswith("sz"):
        return code[2:] + ".SZ"
    # Bare code: infer exchange
    if code.startswith(("000", "001", "002", "003", "159", "16", "300", "301")):
        return code + ".SZ"
    return code + ".SH"


def from_qmt_code(qmt_code: str) -> str:
    """512880.SH → sh512880, 159915.SZ → sz159915"""
    code = str(qmt_code).strip()
    if code.endswith(".SH"):
        return "sh" + code[:-3]
    if code.endswith(".SZ"):
        return "sz" + code[:-3]
    return code  # unknown format, return as-is


def bare_code(code: str) -> str:
    """sh512880 → 512880, 512880.SH → 512880"""
    c = str(code).strip()
    for prefix in ("sh", "sz"):
        if c.startswith(prefix):
            c = c[2:]
    for suffix in (".SH", ".SZ"):
        if c.endswith(suffix):
            c = c[:-3]
    return c


# ── ETF 替代映射 ────────────────────────────────────────────────
# 当 QMT 中某 ETF 数据不可用（全 -1.0 或缺失）时的替代品

ETF_SUBSTITUTES = {
    "510300.SH": "510330.SH",  # 沪深300ETF → 沪深300ETF(广发)
    "511010.SH": "511220.SH",  # 国债ETF → 城投债ETF
    "510500.SH": "510510.SH",  # 中证500ETF → 中证500ETF(广发)
    "512690.SH": "159928.SZ",  # 酒ETF → 消费ETF
}


# ── 拆分修正 ────────────────────────────────────────────────────
# 复制自 etf_r1_backtest.py 的 _fix_splits 逻辑

def fix_splits(close, open_=None):
    """
    修正 ETF 份额拆分导致的价格跳空。
    检测 |单日变动| > 25% 的跳空（A股 ETF 涨跌停板 ±20%，超过必然是拆分/折算），
    将跳空前全部历史按比例缩放，实现后复权式连续化。
    """
    c = close.copy()
    o = open_.copy() if open_ is not None else None
    r = c.ffill().pct_change()
    for d in r.index[r.abs() > 0.25]:
        i = c.index.get_loc(d)
        prev_idx = c.iloc[:i].last_valid_index()
        if prev_idx is None:
            continue
        factor = c.loc[d] / c.loc[prev_idx]
        c.iloc[:i] = c.iloc[:i] * factor
        if o is not None:
            o.iloc[:i] = o.iloc[:i] * factor
    if o is not None:
        return c, o
    return c


# ── 数据加载 ────────────────────────────────────────────────────

def load_qmt_etfs(codes, start=None, end=None,
                  fix_splits_enabled=False,  # qfq 数据不需要拆分修正
                  verbose=True):
    """
    批量加载 ETF 日线数据（前复权）。

    Args:
        codes: 策略格式的 ETF 代码列表，如 ['sh512880', 'sz159915', '511880']
        start: 起始日期 '2015-01-01'
        end: 截止日期 '2026-07-25'
        fix_splits_enabled: 默认 False（qfq 数据已处理拆分）
        verbose: 是否打印状态信息

    Returns:
        {strategy_code: DataFrame} 其中 DataFrame 列: open/high/low/close/volume/amount
    """
    conn = sqlite3.connect(DB_ETF_PATH)
    result = {}
    missing = []

    for sc in codes:
        qmt_c = to_qmt_code(sc)

        # 检查是否存在且有效
        row = conn.execute(
            "SELECT COUNT(*) as n, SUM(CASE WHEN close > 0 THEN 1 ELSE 0 END) as valid "
            "FROM daily WHERE code=?", (qmt_c,)
        ).fetchone()
        total, valid = row[0], row[1]

        if total == 0:
            missing.append((sc, qmt_c, "不存在"))
            continue
        if valid == 0:
            missing.append((sc, qmt_c, f"全 {total} 行 close≤0"))
            continue

        # 加载数据
        # QMT 日期格式为 YYYYMMDD，需转换过滤条件
        where = [f"code='{qmt_c}'"]
        if start:
            s_date = start.replace('-', '')
            where.append(f"date>='{s_date}'")
        if end:
            e_date = end.replace('-', '')
            where.append(f"date<='{e_date}'")
        sql = f"SELECT date, open, high, low, close, volume, amount FROM daily WHERE {' AND '.join(where)} ORDER BY date"
        df = pd.read_sql(sql, conn, index_col='date', parse_dates=['date'])

        # 过滤前视偏误：上市前 volume == 0 的数据
        real_start = (df['volume'] > 0).idxmax() if (df['volume'] > 0).any() else None
        if real_start is not None:
            df = df.loc[real_start:]

        if len(df) == 0:
            missing.append((sc, qmt_c, "过滤后无有效数据"))
            continue

        result[sc] = df

    conn.close()

    if verbose:
        print(f"[QMT] ETF 加载: {len(result)}/{len(codes)} 可用 (qfq前复权)")
        for sc, code, reason in missing:
            print(f"  [QMT] ⚠ {sc} ({code}): {reason}")

    return result


def load_qmt_benchmark(code: str = "000300.SH", start: str = None, end: str = None) -> pd.DataFrame:
    """
    加载基准指数（默认沪深300）。

    Args:
        code: QMT 格式指数代码，如 '000300.SH'
        start, end: 日期范围

    Returns:
        DataFrame 列: open/high/low/close/volume/amount
    """
    conn = sqlite3.connect(DB_ETF_PATH)
    where = [f"code='{code}'"]
    if start:
        where.append(f"date>='{start.replace('-','')}'")
    if end:
        where.append(f"date<='{end.replace('-','')}'")
    sql = f"SELECT date, open, high, low, close, volume, amount FROM daily WHERE {' AND '.join(where)} ORDER BY date"
    df = pd.read_sql(sql, conn, index_col='date', parse_dates=['date'])
    conn.close()

    if len(df) == 0:
        raise ValueError(f"基准 {code} 在 QMT(qfq) 中无数据")

    # 过滤 volume == 0 的回填数据
    real_start = (df['volume'] > 0).idxmax() if (df['volume'] > 0).any() else None
    if real_start is not None:
        df = df.loc[real_start:]

    return df


def load_qmt_stocks(start=None, end=None, min_history=250):
    """
    加载 QMT 中所有 A 股个股日线数据（用于 ETF-R1 因子投票）。

    Args:
        start, end: 日期范围
        min_history: 最少历史天数要求

    Returns:
        {bare_code: DataFrame} 例如 {'600519': DataFrame}
        bare_code 格式匹配 compute_stock_factors 的输入
    """
    conn = sqlite3.connect(DB_STOCK_PATH)

    # 获取所有股票代码（排除 ETF/指数/基金）
    all_codes = [row[0] for row in conn.execute(
        "SELECT DISTINCT code FROM daily"
    ).fetchall()]

    # 过滤：只保留 A 股个股
    stock_codes = []
    for c in all_codes:
        bare = c.split(".")[0]
        if bare.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")):
            stock_codes.append(c)

    print(f"[QMT] 个股: {len(stock_codes)} 只 (总 {len(all_codes)} 只标的中筛选)")

    result = {}
    skipped_short = 0
    skipped_no_data = 0

    for code in stock_codes:
        bare = code.split(".")[0]

        where = [f"code='{code}'", "close > 0"]
        if start:
            where.append(f"date>='{start}'")
        if end:
            where.append(f"date<='{end}'")
        sql = f"SELECT date, open, high, low, close, volume, amount FROM daily WHERE {' AND '.join(where)} ORDER BY date"
        df = pd.read_sql(sql, conn, index_col='date', parse_dates=['date'])

        # 过滤 volume == 0
        df = df[df['volume'] > 0]

        if len(df) < min_history:
            skipped_short += 1
            continue

        result[bare] = df

    conn.close()
    print(f"[QMT] 个股加载: {len(result)} 只 (跳过 {skipped_short} 只不足{min_history}天)")
    return result


def get_etf_stats(codes: list = None):
    """快速查询 ETF 数据可用性"""
    conn = sqlite3.connect(DB_PATH)
    if codes is None:
        rows = conn.execute(
            "SELECT code, COUNT(*) as n, SUM(CASE WHEN close>0 THEN 1 ELSE 0 END) as valid, "
            "SUM(CASE WHEN volume>0 THEN 1 ELSE 0 END) as traded, "
            "MIN(date) as d1, MAX(date) as d2 "
            "FROM daily WHERE code LIKE '5%.SH' OR code LIKE '15%.SZ' OR code LIKE '16%.SZ' "
            "GROUP BY code HAVING valid > 0 ORDER BY code"
        ).fetchall()
    else:
        qmt_codes = [to_qmt_code(c) for c in codes]
        placeholders = ",".join(["?"] * len(qmt_codes))
        rows = conn.execute(
            f"SELECT code, COUNT(*) as n, SUM(CASE WHEN close>0 THEN 1 ELSE 0 END) as valid, "
            f"SUM(CASE WHEN volume>0 THEN 1 ELSE 0 END) as traded, "
            f"MIN(date) as d1, MAX(date) as d2 "
            f"FROM daily WHERE code IN ({placeholders}) "
            f"GROUP BY code HAVING valid > 0 ORDER BY code",
            qmt_codes
        ).fetchall()
    conn.close()
    return [
        {"code": r[0], "total": r[1], "valid": r[2], "traded": r[3], "first": r[4], "last": r[5]}
        for r in rows
    ]


# 模块加载时打印数据质量提醒
print("[QMT Adapter] 已加载。ETF: qmt_qfq.db (前复权), 个股: qmt_full.db")
print("[QMT Adapter]   - 自动过滤 volume==0 的前视偏误数据")
