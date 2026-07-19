# -*- coding: utf-8 -*-
"""
RC-FW v1: 制度条件因子权重 (Regime-Conditional Factor Weights)

复用 V5 因子计算管线(factors_v5), 仅权重按制度分档切换。

预注册设计(2026-07-19):
  因子池: low_vol_250 / sharpe_mom / PMO / ROE_quality / EP
  权重矩阵: 见 REGIME_WEIGHTS_45 / REGIME_WEIGHTS_35
  硬过滤: isST / 近20日涨幅前25%剔除

用法:
  select_fn = build_select_fn(stock_data, calendar, aux, qroe,
                               momentum_45=True, top_n=15)
  codes = select_fn(date, regime)
"""
import numpy as np
import pandas as pd

from src.factors_v5 import (
    load_aux_panels, load_quarterly_roe, quality_scores_at,
    _pct_rank, REVERSAL_EXCLUDE_PCT, MAX20_NEUTRAL_PCT,
)

# ── 制度-因子权重矩阵 ──────────────────────────────────
# 两个版本: RISKON动量 45% vs 35%

REGIME_WEIGHTS_45 = {
    #         low_vol  mom  PMO  ROE_qual  EP
    "RISKON":  (0.15, 0.45, 0.15, 0.15, 0.10),
    "NEUTRAL": (0.30, 0.25, 0.15, 0.20, 0.10),
    "RISKOFF": (0.45, 0.10, 0.15, 0.25, 0.05),
    "CRISIS":  (0.00, 0.00, 0.00, 0.00,  0.00),  # 不选股, 权重无意义
}

REGIME_WEIGHTS_35 = {
    "RISKON":  (0.20, 0.35, 0.15, 0.20, 0.10),
    "NEUTRAL": (0.30, 0.25, 0.15, 0.20, 0.10),
    "RISKOFF": (0.45, 0.10, 0.15, 0.25, 0.05),
    "CRISIS":  (0.00, 0.00, 0.00, 0.00,  0.00),
}

# 对照: 固定权重(用于归因制度切换效应)
FIXED_WEIGHTS = (0.25, 0.25, 0.15, 0.25, 0.10)
# low_vol=25, mom=25(相当于V5的15%动量+10%流动性中非动量的部分...
# 这里做一个合理的固定基准: 低波25/动量25/PMO15/质量25/EP10)


def build_select_fn(stock_data, calendar, aux, qroe, top_n=15,
                    momentum_45=True, low_vol_window=250,
                    reversal_pct=REVERSAL_EXCLUDE_PCT):
    """
    返回 select_fn(date, regime) -> [code, ...]

    参数:
      momentum_45: True 用 45% 动量(RISKON), False 用 35%
      reversal_pct: 近月涨幅剔除分位(默认 25%)
    """
    regime_w = REGIME_WEIGHTS_45 if momentum_45 else REGIME_WEIGHTS_35

    # ── 预计算面板(与 V5 完全一致) ──
    close = pd.DataFrame({c: sdf["close"] for c, sdf in stock_data.items()}).reindex(calendar)
    ret = close.pct_change()

    vol = ret.rolling(low_vol_window, min_periods=int(low_vol_window * 0.8)).std()
    ret20 = close / close.shift(20) - 1.0
    max20 = ret.rolling(20).max()

    mom_ret = close.shift(20) / close.shift(120) - 1.0
    mom_vol = ret.shift(20).rolling(100, min_periods=80).std()
    sharpe_mom = mom_ret / mom_vol.replace(0, np.nan)

    turn = aux["turn"]
    amount = aux["amount"]
    pe = aux["peTTM"]
    isst = aux["isST"]

    pmo = turn.rolling(20, min_periods=15).sum() / turn.rolling(250, min_periods=200).sum()
    # Amihud: 剔除涨跌停日(与 V5 复审修复一致)
    ret_nolimit = ret.where(ret.abs() < 0.095)
    amihud = (ret_nolimit.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()

    bars_count = close.notna().cumsum()

    def select_fn(date, regime="NEUTRAL"):
        if date not in close.index:
            return []

        alive = close.loc[date].notna() & (bars_count.loc[date] >= 250)
        idx = alive[alive].index
        if len(idx) < top_n:
            return []

        # ── 硬过滤 1: ST ──
        st_row = isst.loc[date].reindex(idx)
        idx = idx[(st_row != 1).fillna(True).values]

        # ── 硬过滤 2: 近月涨幅前 N% 剔除 ──
        r20 = ret20.loc[date].reindex(idx)
        cut = r20.quantile(1 - reversal_pct)
        idx = idx[(r20 < cut).fillna(False).values]
        if len(idx) < top_n:
            return []

        # ── 取当前制度的权重 ──
        w_lv, w_mom, w_pmo, w_q, w_ep = regime_w.get(regime, regime_w["NEUTRAL"])

        # ── 因子截面排名(与 V5 完全相同) ──
        # 低波
        r_lowvol = _pct_rank(vol.loc[date].reindex(idx), good_high=False)

        # 动量(夏普动量, MAX20 中性化)
        r_mom = _pct_rank(sharpe_mom.loc[date].reindex(idx), good_high=True)
        m20 = max20.loc[date].reindex(idx)
        lottery = m20 >= m20.quantile(1 - MAX20_NEUTRAL_PCT)
        r_mom[lottery.fillna(False)] = 0.5

        # 流动性桶: PMO + Amihud 等权
        r_pmo = _pct_rank(pmo.loc[date].reindex(idx), good_high=False)
        r_amihud = _pct_rank(amihud.loc[date].reindex(idx), good_high=True)
        r_liq = pd.concat([r_pmo, r_amihud], axis=1).mean(axis=1)

        # 质量桶: ROE水平 + ΔROE + 稳定性
        qs = quality_scores_at(qroe, date)
        qdf = pd.DataFrame.from_dict(qs, orient="index",
                                     columns=["roe", "droe", "stab"]).reindex(idx)
        r_q = pd.concat([
            _pct_rank(qdf["roe"], good_high=True),
            _pct_rank(qdf["droe"], good_high=True),
            _pct_rank(qdf["stab"], good_high=False),
        ], axis=1).mean(axis=1)

        # 价值: EP
        pe_row = pe.loc[date].reindex(idx)
        ep = 1.0 / pe_row.where(pe_row > 0)
        r_ep = _pct_rank(ep, good_high=True).fillna(0.0)

        # ── 制度条件加权合成(与 V5 的唯一差异) ──
        comp = (w_lv * r_lowvol.fillna(0.5)
                + w_mom * r_mom.fillna(0.5)
                + w_pmo * r_liq.fillna(0.5)
                + w_q * r_q.fillna(0.5)
                + w_ep * r_ep)
        # 注: PMO权重作用在流动性桶整体上, 桶内PMO+Amihud仍等权

        return list(comp.sort_values(ascending=False).head(top_n).index)

    return select_fn
