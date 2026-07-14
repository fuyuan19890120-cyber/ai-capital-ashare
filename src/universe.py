"""
资产池定义与统一数据接口
"""
import pandas as pd
from config import UNIVERSE, START_DATE, END_DATE
from src.data_fetcher import fetch_all_etfs, fetch_index_daily


_PRICING_CACHE = {}
_BENCHMARK_CACHE = None


def get_universe():
    """返回资产池定义"""
    return UNIVERSE


def get_risk_assets():
    """返回风险资产列表"""
    return [a for a in UNIVERSE if a["type"] == "risk"]


def get_defensive_assets():
    """返回防御资产列表"""
    return [a for a in UNIVERSE if a["type"] == "defensive"]


def load_price_data(force_refresh=False):
    """
    加载 8 只 ETF 价格数据
    返回 dict: {code: DataFrame}
    有缓存则直接用缓存，消除重复网络请求
    """
    global _PRICING_CACHE

    if _PRICING_CACHE and not force_refresh:
        return _PRICING_CACHE

    start = START_DATE.replace("-", "")
    end = END_DATE.replace("-", "")

    print("📡 加载 ETF 价格数据...")
    _PRICING_CACHE = fetch_all_etfs(UNIVERSE, start, end, force_refresh)
    print(f"✅ 加载完成，{len(_PRICING_CACHE)} 只 ETF")
    return _PRICING_CACHE


def load_benchmark(force_refresh=False):
    """
    加载基准指数数据（沪深300）
    """
    global _BENCHMARK_CACHE

    if _BENCHMARK_CACHE is not None and not force_refresh:
        return _BENCHMARK_CACHE

    start = START_DATE.replace("-", "")
    end = END_DATE.replace("-", "")

    print("📡 加载沪深300指数数据...")
    _BENCHMARK_CACHE = fetch_index_daily("sh000300", start, end, force_refresh)
    print(f"✅ 沪深300加载完成")
    return _BENCHMARK_CACHE


def get_aligned_prices(prices_dict):
    """
    将所有 ETF 价格对齐到共同的交易日索引
    只保留所有 ETF 都有数据的日期
    前向填充停牌期间的数据
    """
    if not prices_dict:
        return pd.DataFrame()

    closes = {}
    for code, df in prices_dict.items():
        closes[code] = df["close"]

    df_close = pd.DataFrame(closes)
    df_close = df_close.sort_index()
    # 前向填充（处理停牌）
    df_close = df_close.ffill()
    # 删除全 NaN 的行
    df_close = df_close.dropna(how="all")
    return df_close


def get_aligned_volumes(prices_dict):
    """获取对齐的成交量数据"""
    if not prices_dict:
        return pd.DataFrame()
    volumes = {}
    for code, df in prices_dict.items():
        volumes[code] = df["volume"]
    df_vol = pd.DataFrame(volumes)
    df_vol = df_vol.sort_index()
    df_vol = df_vol.ffill()
    return df_vol
