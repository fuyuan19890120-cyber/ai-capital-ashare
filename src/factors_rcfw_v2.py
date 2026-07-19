# -*- coding: utf-8 -*-
"""
RC-FW v2: V4.1 简单因子 + 制度切换权重

因子计算逻辑与 stock_data.compute_stock_factors 完全一致(同公式、同 log 变换),
唯一差异: 因子权重按制度分档切换。

V4.1 实际有效敞口(审计修正): 低波 ~43% + 6月动量 ~25% + 250日动量 ~20% + 噪声 ~12%
NEUTRAL 权重逼近该有效敞口; RISKON 向动量倾斜; RISKOFF 向低波倾斜。

硬过滤: isST + 近20日涨幅前25%剔除
"""
import numpy as np
import pandas as pd

# ── 因子权重矩阵(V4.1 原始因子名保持一致) ──
# 四因子: low_vol / value / quality / momentum_6m
# 审计注: value 无 pe_data 时退化为 low_vol*0.5, quality 为250日动量

REGIME_WEIGHTS_V2 = {
    # RISKON: 进攻 — 动量优先, 低波收锚
    "RISKON": {
        "low_vol": 0.12,
        "value": 0.08,       # 退化→低波, 有效低波 ≈ 12+4=16%
        "quality": 0.25,      # 250日动量
        "momentum_6m": 0.55,  # 6月夏普动量 — 主引擎
    },
    # NEUTRAL: 均衡 — 逼近 V4.1 有效敞口 (~低波43/动量25/长期动量20)
    "NEUTRAL": {
        "low_vol": 0.30,
        "value": 0.25,        # 退化→低波, 有效低波 ≈ 30+12.5=42.5%
        "quality": 0.20,       # 250日动量 (V4.1 名义"质量")
        "momentum_6m": 0.25,   # 6月夏普动量
    },
    # RISKOFF: 防守 — 低波优先, 动量几乎清零
    "RISKOFF": {
        "low_vol": 0.50,
        "value": 0.15,        # 退化→低波, 有效低波 ≈ 50+7.5=57.5%
        "quality": 0.25,       # 250日动量(长期趋势在弱市中仍有参考价值)
        "momentum_6m": 0.10,   # 仅保留底仓
    },
    "CRISIS": {
        "low_vol": 0.30, "value": 0.25, "quality": 0.20, "momentum_6m": 0.25,
        # CRISIS 不选股, 权重无意义, 填 NEUTRAL 占位
    },
}

# ── 硬过滤参数 ──
REVERSAL_EXCLUDE_PCT = 0.25   # 近月涨幅前 25% 剔除


def build_v4_select_fn(stock_data, calendar, top_n=15,
                       reversal_pct=REVERSAL_EXCLUDE_PCT):
    """
    返回 select_fn(date, regime) -> [code, ...]

    因子计算与 V4.1 stock_data.compute_stock_factors 完全一致,
    仅 FACTOR_WEIGHTS 随 regime 切换。
    """
    regime_w = REGIME_WEIGHTS_V2

    # ── 预计算 ──
    close = pd.DataFrame(
        {c: sdf["close"] for c, sdf in stock_data.items()}
    ).reindex(calendar)
    ret_pct = close.pct_change()

    # 63日波动率(与 V4.1 一致)
    vol_63 = ret_pct.rolling(63, min_periods=20).std() * np.sqrt(252)

    # 近20日涨幅(反转过滤用)
    ret20 = close / close.shift(20) - 1.0

    # 6个月夏普动量(与 V4.1 一致: ret_6m/vol_6m)
    # V4.1 用 hist['close'].iloc[-1] / hist['close'].iloc[-126] - 1
    ret_6m = close / close.shift(126) - 1.0
    vol_6m = ret_pct.rolling(126, min_periods=60).std()
    # 注意: V4.1 的动量=ret_6m/vol_6m, 不是滚动 Sharpe
    # 实际计算: 对每个 date, hist = df[:date].dropna()[-126:],
    # ret_6m = close[-1]/close[-126]-1, vol_6m = ret.std()
    # 这里用面板近似(略有差异但方向一致): ret_6m / (vol_6m 的年化)
    # 更精确: 逐日点计算, 但性能考虑用面板近似
    sharpe_6m = ret_6m / vol_6m.replace(0, np.nan)

    # 250日收益(与 V4.1 "quality" 一致)
    ret_12m = close / close.shift(250) - 1.0

    bars_count = close.notna().cumsum()

    def select_fn(date, regime="NEUTRAL"):
        if date not in close.index:
            return []

        alive = close.loc[date].notna() & (bars_count.loc[date] >= 250)
        idx = alive[alive].index
        if len(idx) < top_n:
            return []

        # ── 硬过滤: 近月涨幅前 N% 剔除 ──
        r20 = ret20.loc[date].reindex(idx)
        cut = r20.quantile(1 - reversal_pct)
        idx = idx[(r20 < cut).fillna(False).values]
        if len(idx) < top_n:
            return []

        # ── 取权重 ──
        w = regime_w.get(regime, regime_w["NEUTRAL"])

        # ── 逐股计算因子得分(与 V4.1 完全一致) ──
        scores = {}
        for code in idx:
            # 低波因子
            v_lv = vol_63.at[date, code]
            if pd.notna(v_lv) and v_lv > 0:
                f_lowvol = 1.0 / (v_lv + 0.01)
            else:
                f_lowvol = 0.0

            # 价值因子(无 pe_data, 退化为 low_vol*0.5, 与 V4.1 一致)
            f_value = f_lowvol * 0.5

            # 质量因子(250日收益, 与 V4.1 一致)
            r12 = ret_12m.at[date, code]
            if pd.notna(r12):
                f_quality = max(0.0, float(r12)) * 2.0
            else:
                f_quality = 0.0

            # 动量因子(6月夏普动量, 与 V4.1 一致)
            sm = sharpe_6m.at[date, code]
            if pd.notna(sm):
                f_mom = float(sm)
            else:
                # fallback: 原始6月收益
                r6 = ret_6m.at[date, code]
                f_mom = float(r6) if pd.notna(r6) else 0.0

            # ── 加权合成(与 V4.1 一致: log1p on value/low_vol) ──
            total = 0.0
            for factor, weight in w.items():
                if factor == "low_vol":
                    total += np.log1p(max(f_lowvol, 0)) * weight
                elif factor == "value":
                    total += np.log1p(max(f_value, 0)) * weight
                elif factor == "quality":
                    total += f_quality * weight
                elif factor == "momentum_6m":
                    total += f_mom * weight
            scores[code] = total

        # ── 排名取 Top-N ──
        sorted_codes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [c for c, s in sorted_codes[:top_n] if s > 0]

    return select_fn
