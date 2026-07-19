"""
个股数据模块：CSI300 成分股 + 多因子数据
"""
import os
import time
import pickle
import warnings
import pandas as pd
import numpy as np
import akshare as ak
from config import DATA_DIR

STOCK_CACHE_DIR = os.path.join(DATA_DIR, "stocks")
os.makedirs(STOCK_CACHE_DIR, exist_ok=True)

# Factor definitions with weights (based on 3-year factor research)
# Low vol: only factor that worked all 3 years (2024-2026)
# Value: strong 2024 and 2026
# Quality (ROE): recovering 2024-2026
# Momentum: for trend following during RISKON
FACTOR_WEIGHTS = {
    "low_vol": 0.30,     # 1/volatility (63d)
    "value": 0.25,       # 1/PE
    "quality": 0.20,     # ROE
    "momentum_6m": 0.25, # 6-month return
}


def get_csi300_constituents():
    """获取当前沪深300成分股"""
    df = ak.index_stock_cons(symbol="000300")
    codes = [str(c).zfill(6) for c in df['品种代码'].tolist()]
    return codes


def fetch_stock_daily(code, start_date="20150101", end_date="20260714", force_refresh=False):
    """获取单只个股日线数据"""
    cache_path = os.path.join(STOCK_CACHE_DIR, f"{code}.csv")

    if not force_refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 200:
                return df
        except Exception:
            pass

    # Determine sh/sz prefix for Sina
    prefix = "sh" if code.startswith(("6", "9")) else "sz"

    for attempt in range(3):
        try:
            # Try Sina source first (more reliable)
            df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", adjust="qfq")
            if df is not None and not df.empty:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                df = df.sort_index()
                # Filter date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                if len(df) > 200:
                    # Keep essential columns
                    keep = ["open", "high", "low", "close", "volume"]
                    df = df[[c for c in keep if c in df.columns]]
                    df.to_csv(cache_path)
                    return df
            break  # Don't retry if Sina returned empty
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            continue

    # Fallback: East Money source
    try:
        time.sleep(1)
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date,
            adjust="qfq"
        )
        if df is not None and not df.empty:
            df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            keep = ["open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            df.to_parquet(cache_path)
            return df
    except Exception:
        pass

    return None


def fetch_all_stocks(codes, max_stocks=None, verbose=True):
    """批量获取个股数据，自动缓存"""
    results = {}
    fetch_list = codes[:max_stocks] if max_stocks else codes

    success = 0
    for i, code in enumerate(fetch_list):
        if verbose and i % 50 == 0:
            print(f"  Progress: {i}/{len(fetch_list)} ({success} loaded)")

        # Check cache first
        cache_path = os.path.join(STOCK_CACHE_DIR, f"{code}.parquet")
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty and len(df) > 200:
                    results[code] = df
                    success += 1
                    continue
            except Exception:
                pass

        # Fetch with delay
        time.sleep(0.8)

        df = fetch_stock_daily(code)
        if df is not None and not df.empty and len(df) > 200:
            results[code] = df
            success += 1

    if verbose:
        print(f"  Done: {success}/{len(fetch_list)} stocks loaded")
    return results


def get_st_filters():
    """获取ST股票列表，用于过滤"""
    try:
        st_df = ak.stock_zh_a_st_em()
        st_codes = set(st_df['代码'].tolist())
        return st_codes
    except Exception:
        return set()


def compute_stock_factors(stock_data, date, pe_data=None):
    """
    计算个股多因子得分

    stock_data: {code: DataFrame}
    date: 当前调仓日期
    pe_data: {code: PE_series}（可选，从估值数据获取）

    ⚠️ 审计警示(2026-07-18): 生产链路(signal_generator/stock_backtest)均未传 pe_data,
    此时 value 退化为 low_vol*0.5、quality 是 250 日动量 —— 实际敞口 ≈ 低波43% + 动量45%,
    并非宣称的"低波30/价值25/质量20/动量25"。接入真实 PE/ROE(按披露日对齐)前,
    对本策略的一切讨论按"低波+动量组合"理解。详见 reports/v4_audit_fix_report.md。

    返回: {code: score}
    """
    scores = {}

    for code, df in stock_data.items():
        if date not in df.index:
            continue

        hist = df[df.index <= date].dropna()

        # 需要至少 1 年数据
        if len(hist) < 250:
            continue

        factor_scores = {}

        # === 低波动率因子 (30%) ===
        ret = hist['close'].pct_change().dropna().iloc[-63:]
        if len(ret) >= 20:
            vol = ret.std() * np.sqrt(252)
            factor_scores["low_vol"] = 1.0 / (vol + 0.01)  # 波动率越低分越高
        else:
            factor_scores["low_vol"] = 0

        # === 价值因子 (25%) ===
        if pe_data and code in pe_data:
            pe = pe_data[code]
            if pe > 0 and pe < 500:
                factor_scores["value"] = 1.0 / pe  # PE越低分越高
            else:
                factor_scores["value"] = 0
        else:
            # 用价格/账面做简单替代：用波动率倒数中的信息
            factor_scores["value"] = factor_scores.get("low_vol", 0) * 0.5

        # === 质量因子 (20%) ===
        # 简化：用股价长期趋势（能持续上涨的股票质量大概率不差）
        if len(hist) >= 500:
            long_ret = hist['close'].iloc[-1] / hist['close'].iloc[-250] - 1
            # 正收益 = 质量好
            factor_scores["quality"] = max(0, long_ret) * 2  # Scale up positive returns
        else:
            factor_scores["quality"] = 0

        # === 动量因子 (25%) ===
        if len(hist) >= 126:
            ret_6m = hist['close'].iloc[-1] / hist['close'].iloc[-126] - 1
            vol_6m = hist['close'].pct_change().dropna().iloc[-126:].std()
            if vol_6m > 0:
                factor_scores["momentum_6m"] = ret_6m / vol_6m
            else:
                factor_scores["momentum_6m"] = ret_6m
        else:
            factor_scores["momentum_6m"] = 0

        # === 加权合成 ===
        total = 0
        for factor, weight in FACTOR_WEIGHTS.items():
            val = factor_scores.get(factor, 0)
            if factor in ("value", "low_vol"):
                val = np.log1p(max(val, 0))  # Log transform for skewed distributions
            total += val * weight

        scores[code] = total

    return scores


def filter_reversal_stocks(stock_data, date, exclude_pct=0.15):
    """
    V4.2 反转硬过滤(2026-07-19): 剔除近20日涨幅前 exclude_pct 的候选股。

    A股截面动量为负/反转效应是最稳健的量价因子(roadmap P1, CH-3/CH-4),
    但多头端弱——最佳用法是作剔除项: 追涨买入短期涨幅最高的票几乎必亏。

    stock_data: {code: DataFrame}
    date: 当前调仓日期
    exclude_pct: 剔除分位(默认 0.25 = 前25%)
    返回: 过滤后的 {code: DataFrame}
    """
    returns_20d = {}
    for code, df in stock_data.items():
        if date in df.index:
            idx = df.index.get_loc(date)
            if idx >= 20:
                ret = float(df['close'].iloc[idx]) / float(df['close'].iloc[idx - 20]) - 1
                returns_20d[code] = ret

    if len(returns_20d) < 10:
        return stock_data  # 样本太少, 不过滤

    threshold = np.percentile(list(returns_20d.values()), (1 - exclude_pct) * 100)
    filtered = {code: stock_data[code] for code, ret in returns_20d.items()
                if ret < threshold}
    return filtered


def select_top_stocks(scores, top_n=10):
    """从评分中选出 Top N 股票"""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [code for code, score in sorted_scores[:top_n] if score > 0]
