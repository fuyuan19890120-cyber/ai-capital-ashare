# -*- coding: utf-8 -*-
"""
可转债日内研究 - 分钟数据本地存储

parquet 按 频率/代码 分文件: data/cb_min/{freq}/{symbol}.parquet
增量合并去重(upsert): 新旧数据按 datetime 去重, 保留最新一次抓取。
免费源没有长历史, 档案靠每日采集滚动积累 —— 存储层是整个基建的地基。
"""
import os

import pandas as pd

from . import config


def _path(symbol: str, freq: str) -> str:
    d = os.path.join(config.MIN_DATA_DIR, freq)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{symbol}.parquet")


def load(symbol: str, freq: str) -> pd.DataFrame:
    """读取单券单频率的全部已积累数据; 无档案返回空表"""
    p = _path(symbol, freq)
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_parquet(p)


def upsert(symbol: str, freq: str, new_df: pd.DataFrame) -> int:
    """增量写入: 与已有档案按 datetime 去重合并, 返回新增行数"""
    if new_df is None or new_df.empty:
        return 0
    old = load(symbol, freq)
    if old.empty:
        merged = new_df.copy()
        added = len(merged)
    else:
        merged = pd.concat([old, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset="datetime", keep="last")
        added = len(merged) - len(old)
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged.to_parquet(_path(symbol, freq), index=False)
    return max(added, 0)


def load_universe(freq: str, symbols=None) -> pd.DataFrame:
    """读取某频率下全部(或指定)券, 拼成长表 [symbol, datetime, o/h/l/c/v/amount]"""
    d = os.path.join(config.MIN_DATA_DIR, freq)
    if not os.path.isdir(d):
        return pd.DataFrame()
    frames = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".parquet"):
            continue
        sym = fn[:-8]
        if symbols is not None and sym not in symbols:
            continue
        df = pd.read_parquet(os.path.join(d, fn))
        df.insert(0, "symbol", sym)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def archive_summary(freq: str) -> pd.DataFrame:
    """档案盘点: 每券行数与时间范围, 用于检查采集完整性"""
    d = os.path.join(config.MIN_DATA_DIR, freq)
    if not os.path.isdir(d):
        return pd.DataFrame()
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(d, fn), columns=["datetime"])
            rows.append({"symbol": fn[:-8], "bars": len(df),
                         "first": df["datetime"].min(), "last": df["datetime"].max()})
    return pd.DataFrame(rows)
