"""
数据获取与缓存模块
封装 AKShare，自动重试 + CSV 缓存，统一输出格式
"""
import os
import time
import pandas as pd
import akshare as ak
from config import DATA_DIR

CACHE_DIR = DATA_DIR


def _get_cache_path(symbol):
    """获取 CSV 缓存文件路径"""
    return os.path.join(CACHE_DIR, f"{symbol}.csv")


def _load_cache(symbol):
    """从 CSV 缓存加载数据"""
    path = _get_cache_path(symbol)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not df.empty:
            return df
    return None


def _save_cache(symbol, df):
    """保存数据到 CSV 缓存"""
    path = _get_cache_path(symbol)
    df.to_csv(path)


def fetch_etf_daily(code, start_date="20150101", end_date="20260713", force_refresh=False):
    """
    获取单只 ETF 日线前复权数据
    返回标准 DataFrame：[date, open, high, low, close, volume]
    """
    cache_key = f"etf_{code}"

    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    retries = 3
    for attempt in range(retries):
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df.empty:
                raise ValueError(f"ETF {code} returned empty data")

            # 标准化列名
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            df = df.sort_index()
            df = df.astype(float)

            _save_cache(cache_key, df)
            return df

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                pass  # fall through to Sina fallback

    # ========== Fallback: 尝试 Sina 源 ==========
    try:
        # Determine sh/sz prefix
        shsz_code = f"sh{code}" if code.startswith("51") else f"sz{code}"
        df = ak.fund_etf_hist_sina(symbol=shsz_code)
        if df is not None and not df.empty:
            df = _standardize_sina_etf(df)
            _save_cache(cache_key, df)
            return df
    except Exception:
        pass

    print(f"⚠️ ETF {code} 获取失败（东财+Sina双源）")
    return None


def _standardize_sina_etf(df):
    """标准化 Sina 源 ETF 数据格式"""
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if '日' in col or 'date' in cl:
            col_map[col] = 'date'
        elif '开' in col or 'open' in cl:
            col_map[col] = 'open'
        elif '高' in col or 'high' in cl:
            col_map[col] = 'high'
        elif '低' in col or 'low' in cl:
            col_map[col] = 'low'
        elif 'prevclose' in cl or '前收' in col:
            continue  # skip prevclose column
        elif '收' in col or 'close' in cl:
            col_map[col] = 'close'
        elif '量' in col or 'vol' in cl:
            col_map[col] = 'volume'

    df = df.rename(columns=col_map)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    std_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in std_cols if c in df.columns]]
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')

    # 处理重复列（Sina源货币ETF可能有 close + close.1 双列）
    if 'close' in df.columns:
        # 找所有 close.* 列，取第一个非NaN值
        close_cols = [c for c in df.columns if c.startswith('close')]
        if len(close_cols) > 1:
            df['close'] = df[close_cols].bfill(axis=1).iloc[:, 0]
            extra_cols = [c for c in close_cols if c != 'close']
            df = df.drop(columns=extra_cols)

    df = df.dropna(subset=['close'])
    return df


def fetch_index_daily(symbol="sh000300", start_date="20150101", end_date="20260713", force_refresh=False):
    """
    获取指数日线数据（用于制度检测参考）
    返回标准 DataFrame：[date, open, high, low, close, volume]
    """
    cache_key = f"index_{symbol}"

    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    retries = 3
    for attempt in range(retries):
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df.empty:
                raise ValueError(f"Index {symbol} returned empty data")

            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            df = df.sort_index()
            df = df.astype(float)

            # 过滤日期范围
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            df = df[(df.index >= start) & (df.index <= end)]

            _save_cache(cache_key, df)
            return df

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"⚠️ Index {symbol} 获取失败: {e}")
                return None


def fetch_all_etfs(etf_list, start_date="20150101", end_date="20260713", force_refresh=False):
    """
    批量获取 ETF 数据
    etf_list: list of dict, 每项含 code 和 name
    返回 dict: {code: DataFrame}
    """
    results = {}
    for etf in etf_list:
        code = etf["ak_code"]
        name = etf["name"]
        print(f"  获取 {name} ({code})...", end=" ")
        df = fetch_etf_daily(code, start_date, end_date, force_refresh)
        if df is not None and not df.empty:
            results[etf["code"]] = df
            print(f"✓ {len(df)} 行")
        else:
            print(f"✗ 失败")
        time.sleep(0.5)  # 控制请求频率
    return results
