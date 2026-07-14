"""
动量评分模块 (v2 优化版)

四因子复合：反转(1m) + 动量(3m) + 动量(6m) + 低波偏好
"""
import pandas as pd
import numpy as np
from config import (
    LOOKBACK_1M, LOOKBACK_3M, LOOKBACK_6M, VOL_LOOKBACK,
    REVERSAL_WEIGHT_1M, MOMENTUM_WEIGHT_3M, MOMENTUM_WEIGHT_6M, LOWVOL_WEIGHT,
    TS_WEIGHT, CS_WEIGHT,
)


def compute_returns(prices_df):
    """计算日收益率"""
    return prices_df.pct_change()


def compute_volatility(returns_df, lookback):
    """计算滚动年化波动率"""
    return returns_df.rolling(lookback).std() * np.sqrt(252)


def compute_period_return(prices_df, lookback):
    """
    计算回溯期收益率
    (close_t - close_{t-lookback}) / close_{t-lookback}
    """
    return prices_df.pct_change(lookback)


def compute_reversal_signal(prices_df, lookback=LOOKBACK_1M):
    """
    反转信号 (v2): 买最近跌了的
    反转 = -(回溯期收益) / 波动率标准化
    值越大越应该买（跌多了反弹机会大）
    """
    ret = compute_period_return(prices_df, lookback)
    vol = compute_volatility(prices_df.pct_change(), lookback)
    # 反转：取负号，跌得多的得分高
    signal = -ret / (vol + 1e-10)
    return signal


def compute_momentum_signal(prices_df, lookback):
    """
    动量信号: 买最近涨了的
    动量 = 回溯期收益 / 波动率标准化
    值越大越应该买（趋势强）
    """
    ret = compute_period_return(prices_df, lookback)
    vol = compute_volatility(prices_df.pct_change(), lookback)
    signal = ret / (vol + 1e-10)
    return signal


def compute_lowvol_signal(prices_df, lookback=VOL_LOOKBACK):
    """
    低波动率偏好: 波动率越低得分越高
    低波偏好 = 1 / 年化波动率
    """
    ret = prices_df.pct_change()
    vol = compute_volatility(ret, lookback)
    signal = 1.0 / (vol + 1e-10)
    return signal


def compute_composite_score(prices_df):
    """
    计算四因子复合时序得分 (v2 优化版)

    返回 DataFrame，每行一个日期，每列一个 ETF
    """
    # 三个信号
    reversal_1m = compute_reversal_signal(prices_df, LOOKBACK_1M)
    momentum_3m = compute_momentum_signal(prices_df, LOOKBACK_3M)
    momentum_6m = compute_momentum_signal(prices_df, LOOKBACK_6M)
    lowvol = compute_lowvol_signal(prices_df, VOL_LOOKBACK)

    # 对每个信号做截面标准化（z-score），使跨资产可比
    def cross_sectional_zscore(df):
        """各行（截面）z-score 标准化"""
        return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1) + 1e-10, axis=0)

    reversal_z = cross_sectional_zscore(reversal_1m)
    momentum_3m_z = cross_sectional_zscore(momentum_3m)
    momentum_6m_z = cross_sectional_zscore(momentum_6m)
    lowvol_z = cross_sectional_zscore(lowvol)

    # 加权合成时序得分
    ts_score = (
        REVERSAL_WEIGHT_1M * reversal_z +
        MOMENTUM_WEIGHT_3M * momentum_3m_z +
        MOMENTUM_WEIGHT_6M * momentum_6m_z +
        LOWVOL_WEIGHT * lowvol_z
    )

    return ts_score


def compute_cross_sectional_score(ts_score):
    """
    计算截面排名得分（0-1）
    在每个截面（日期）上，对 ETF 做百分位排名
    """
    n = ts_score.shape[1]
    ranks = ts_score.rank(axis=1, ascending=True)
    cs_score = (ranks - 1) / (n - 1)  # 0 到 1
    return cs_score


def compute_final_scores(prices_df):
    """
    计算最终选股得分 (v2 优化版)

    最终得分 = 0.50 × 时序得分(z-score) + 0.50 × 截面排名(0-1)
    返回完整 DataFrame
    """
    ts_score = compute_composite_score(prices_df)
    cs_score = compute_cross_sectional_score(ts_score)

    # 时序得分 z-score 标准化后与截面得分合成
    ts_z = ts_score.subtract(ts_score.mean(axis=1), axis=0).divide(
        ts_score.std(axis=1) + 1e-10, axis=0)

    final_score = TS_WEIGHT * ts_z + CS_WEIGHT * cs_score

    return final_score


def select_top_assets(scores, prices_df, date, top_n=3, max_corr=0.65):
    """
    从得分中选出 Top-N 资产（带相关性过滤）

    scores: 最终得分 DataFrame
    prices_df: 用于计算相关性的价格数据
    date: 当前调仓日期
    top_n: 选几只
    max_corr: 相关性阈值

    返回：选中的 ETF 代码列表
    """
    if date not in scores.index:
        return []

    # 获取当前截面得分排序
    current_scores = scores.loc[date].dropna().sort_values(ascending=False)

    if current_scores.empty:
        return []

    selected = [current_scores.index[0]]  # 最高分一定入选

    if top_n == 1 or len(current_scores) == 1:
        return selected

    # 计算收益率用于相关性
    returns = prices_df.pct_change().dropna()

    # 相关性过滤
    for asset in current_scores.index[1:]:
        if asset in selected:
            continue

        # 与已选资产的最大相关性
        corr_series = returns[selected].corrwith(returns[asset])
        max_correlation = corr_series.max() if not corr_series.empty else 0

        if max_correlation < max_corr:
            selected.append(asset)

        if len(selected) >= top_n:
            break

    return selected[:top_n]
