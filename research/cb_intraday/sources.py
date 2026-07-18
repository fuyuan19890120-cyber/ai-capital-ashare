# -*- coding: utf-8 -*-
"""
可转债日内研究 - 数据源适配层

三个免费源的统一封装(均绕过 akshare 的硬编码限制,直连行情 API):
  - 新浪: 5/15/30/60 分钟 K 线(上限 1023 根,主力历史源)+ 在市转债清单
  - 东财: 1 分钟当日分时(唯一的 1 分钟源)+ 5-60 分钟 K 线(限频激进,备用)
  - 腾讯: 1/5 分钟(上限 320 根,应急备用)

已知约束(见 DATA_QUALITY.md):
  - 任何免费源都没有 >1 年的分钟历史,历史靠每日采集滚动积累
  - 本机直连东财会被断连(IPv6/风控),需 SOCKS 代理;GitHub Actions 美国节点可直连
  - 新浪分钟"成交量"与日线单位差 10 倍 → 流动性一律用成交额(amount, 元)
"""
import json
import os
import random
import re
import time

import pandas as pd
import requests

# 东财代理: 直连失败时依次尝试(本机 VPN 的 SOCKS 端口)
EM_PROXY_CANDIDATES = [None, os.environ.get("CB_PROXY", "socks5h://127.0.0.1:1081")]
EM_UT = "f057cbcbce2a86e2866ab8877db1d059"

_last_request_ts = 0.0


def _throttle(min_interval: float = 0.6):
    """全局限速: 相邻请求至少间隔 min_interval 秒(带随机抖动, 避免整齐节奏触发风控)"""
    global _last_request_ts
    wait = _last_request_ts + min_interval + random.uniform(0, 0.3) - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


def _get(url, params=None, headers=None, proxies=None, tries=3, timeout=20):
    last_err = None
    for i in range(tries):
        _throttle()
        try:
            r = requests.get(url, params=params, headers=headers, proxies=proxies, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - 网络层错误统一重试
            last_err = e
            time.sleep(2.0 * (i + 1))
    raise last_err


# ============================================================
# 新浪
# ============================================================

_SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}


def sina_universe(include_bj: bool = False) -> pd.DataFrame:
    """在市转债清单(新浪行情中心 hskzz_z 节点), 返回 [symbol, name, price, amount]"""
    rows, page = [], 1
    while True:
        url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"Market_Center.getHQNodeData?page={page}&num=80&sort=symbol&asc=1&node=hskzz_z")
        text = _get(url, headers=_SINA_HEADERS).text
        if not text or text.strip() in ("null", "[]"):
            break
        data = json.loads(re.sub(r"(?<={|,)(\w+):", r'"\1":', text))
        rows.extend(data)
        if len(data) < 80:
            break
        page += 1
    df = pd.DataFrame(rows)[["symbol", "name", "trade", "amount"]]
    df.columns = ["symbol", "name", "price", "amount"]
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if not include_bj:
        df = df[~df["symbol"].str.startswith("bj")].reset_index(drop=True)
    return df


def sina_kline(symbol: str, scale: int, datalen: int = 1023) -> pd.DataFrame:
    """
    新浪分钟 K 线。symbol 如 'sh113037'; scale ∈ {5,15,30,60,240(日线)}。
    返回 [datetime, open, high, low, close, volume, amount], 旧→新。
    注意: volume 单位与日线口径不一致, 流动性请用 amount(元)。
    """
    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/x=/CN_MarketDataService.getKLineData"
           f"?symbol={symbol}&scale={scale}&ma=no&datalen={datalen}")
    text = _get(url, headers=_SINA_HEADERS).text
    m = re.search(r"\((.*)\)", text, re.S)
    if not m or m.group(1) in ("null", ""):
        return pd.DataFrame()
    data = json.loads(m.group(1))
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["day"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cols = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    return df[[c for c in cols if c in df.columns]].sort_values("datetime").reset_index(drop=True)


# ============================================================
# 东财
# ============================================================

def _em_secid(symbol: str) -> str:
    return {"sh": "1", "sz": "0"}[symbol[:2]] + "." + symbol[2:]


def _em_get_json(url, params):
    last_err = None
    for proxies in EM_PROXY_CANDIDATES:
        try:
            p = {"http": proxies, "https": proxies} if proxies else None
            return _get(url, params=params, proxies=p, tries=2).json()
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


def em_trends_1min(symbol: str) -> pd.DataFrame:
    """东财当日 1 分钟分时(241 根, 含 9:30 集合竞价)。仅当日, 需每日采集积累。"""
    j = _em_get_json("https://push2.eastmoney.com/api/qt/stock/trends2/get", {
        "secid": _em_secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "iscr": "0", "iscca": "0", "ut": EM_UT, "ndays": "1"})
    trends = (j.get("data") or {}).get("trends") or []
    if not trends:
        return pd.DataFrame()
    df = pd.DataFrame([t.split(",") for t in trends],
                      columns=["datetime", "open", "close", "high", "low", "volume", "amount", "avg_price"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def em_kline(symbol: str, klt: int, lmt: int = 1000000) -> pd.DataFrame:
    """东财分钟 K 线(klt ∈ {5,15,30,60}), 直连拿全服务端深度(绕过 akshare 的 lmt=66)。"""
    j = _em_get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {
        "secid": _em_secid(symbol), "klt": str(klt), "fqt": "0", "lmt": str(lmt),
        "beg": "0", "end": "20500000", "iscca": "1",
        "fields1": "f1,f2,f3,f4,f5",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61", "ut": EM_UT})
    klines = (j.get("data") or {}).get("klines") or []
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame([k.split(",") for k in klines],
                      columns=["datetime", "open", "close", "high", "low", "volume", "amount",
                               "amplitude", "pct_chg", "chg", "turnover"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]


# ============================================================
# 腾讯(备用)
# ============================================================

def tencent_mkline(symbol: str, freq: str = "m5", count: int = 320) -> pd.DataFrame:
    """腾讯分钟 K 线, freq ∈ {m1, m5, m15, m30, m60}, 上限约 320 根。应急备用。"""
    j = _get("https://ifzq.gtimg.cn/appstock/app/kline/mkline",
             params={"param": f"{symbol},{freq},,{count}"}).json()
    k = (j.get("data", {}).get(symbol, {}) or {}).get(freq) or []
    if not k:
        return pd.DataFrame()
    df = pd.DataFrame([r[:6] for r in k],
                      columns=["datetime", "open", "close", "high", "low", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
