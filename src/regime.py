"""
制度检测模块 (v2 优化版)

8 因子加权评分 → 四制度分类（RISKON / NEUTRAL / RISKOFF / CRISIS）

以沪深300为参考基准
"""
import pandas as pd
import numpy as np
from config import REGIME_WEIGHTS, REGIME_THRESHOLDS, BENCHMARK_INDEX


def compute_regime_scores(benchmark_prices, etf_prices_dict, benchmark_volume=None):
    """
    计算制度评分时序

    参数:
        benchmark_prices: 沪深300价格 Series（index=date）
        etf_prices_dict: {code: price_series}，用于广度和相关性计算
        benchmark_volume: 沪深300成交量 Series（可选，用于成交额动能）

    返回:
        regime_df: DataFrame，含各因子得分 + 总分 + 制度标签
    """
    bench = benchmark_prices.dropna()
    if bench.empty:
        return pd.DataFrame()

    results = []

    # 需要至少 300 个交易日预热
    for i in range(300, len(bench)):
        date = bench.index[i]
        hist = bench.iloc[:i+1]  # 截至当日的所有历史数据

        scores = {}
        scores["date"] = date

        # ===== 因子1: 价格 vs SMA200 (10%) =====
        sma200 = hist.rolling(200).mean().iloc[-1]
        price = hist.iloc[-1]
        scores["sma200_vs_price"] = 1.0 if price > sma200 else 0.0

        # ===== 因子2: SMA200 斜率/21日 (8%) =====
        if len(hist) >= 221:
            # 计算完整历史的 SMA200
            sma200_full = hist.rolling(200).mean()
            sma200_now = sma200_full.iloc[-1]
            sma200_21d_ago = sma200_full.iloc[-22] if len(sma200_full) >= 22 else sma200_now
            if pd.notna(sma200_now) and pd.notna(sma200_21d_ago):
                # 21日斜率 > 0.5% 视为上升
                slope_pct = (sma200_now - sma200_21d_ago) / sma200_21d_ago
                scores["sma200_slope"] = 1.0 if slope_pct > 0.005 else (0.0 if slope_pct < -0.005 else 0.5)
            else:
                scores["sma200_slope"] = 0.5
        else:
            scores["sma200_slope"] = 0.5

        # ===== 因子3: 市场广度 (15%) =====
        scores["market_breadth"] = _compute_breadth(etf_prices_dict, date)

        # ===== 因子4: 年化波动率 vs 3年滚动中位数 (15%) =====
        ret = hist.pct_change().dropna().iloc[-21:]
        if len(ret) >= 5:
            ann_vol = ret.std() * np.sqrt(252)
            # 3年滚动波动率中位数
            if len(hist) >= 756:  # 3 years
                rolling_vol = hist.pct_change().rolling(21).std() * np.sqrt(252)
                median_vol = rolling_vol.rolling(756).median().iloc[-1]
                scores["volatility"] = 0.0 if ann_vol > median_vol * 1.3 else (1.0 if ann_vol < median_vol else 0.5)
            else:
                scores["volatility"] = 0.5
        else:
            scores["volatility"] = 0.5

        # ===== 因子5: 21日最大回撤 (15%) =====
        if len(hist) >= 21:
            recent = hist.iloc[-21:]
            peak = recent.expanding().max()
            dd = (recent - peak) / peak
            max_dd = dd.min()
            scores["max_drawdown"] = 0.0 if max_dd < -0.08 else (1.0 if max_dd > -0.03 else 0.5)
        else:
            scores["max_drawdown"] = 0.5

        # ===== 因子6: 平均相关系数 vs 3年中位数 (5%) =====
        scores["correlation"] = _compute_correlation_score(etf_prices_dict, date)

        # ===== 因子7: 成交额动能 (15%) =====
        if benchmark_volume is not None and date in benchmark_volume.index:
            vol_hist = benchmark_volume[benchmark_volume.index <= date]
            if len(vol_hist) >= 60:
                vol_20 = vol_hist.iloc[-20:].mean()
                vol_60 = vol_hist.iloc[-60:].mean()
                ratio = vol_20 / vol_60 if vol_60 > 0 else 1
                if ratio > 1.15:
                    scores["volume_momentum"] = 1.0
                elif ratio < 0.85:
                    scores["volume_momentum"] = 0.0
                else:
                    scores["volume_momentum"] = 0.5
            else:
                scores["volume_momentum"] = 0.5
        else:
            scores["volume_momentum"] = 0.5

        # ===== 因子8: 60日均线乖离率 (17%) =====
        if len(hist) >= 60:
            ma60 = hist.iloc[-60:].mean()
            deviation = (price - ma60) / ma60
            if 0.05 <= deviation <= 0.15:
                scores["deviation_60ma"] = 1.0
            elif deviation > 0.20:
                scores["deviation_60ma"] = 0.0  # 极端超买
            elif deviation < -0.10:
                scores["deviation_60ma"] = 0.0  # 极端超卖
            elif -0.05 <= deviation < 0.05:
                scores["deviation_60ma"] = 0.5
            else:
                scores["deviation_60ma"] = 0.5
        else:
            scores["deviation_60ma"] = 0.5

        # ===== 加权总得分 =====
        total = 0.0
        for factor, weight in REGIME_WEIGHTS.items():
            total += scores.get(factor, 0.5) * weight
        scores["total_score"] = total

        # ===== 制度分类 =====
        scores["regime"] = _classify_regime(total)
        scores["regime_numeric"] = _regime_to_num(scores["regime"])

        results.append(scores)

    df = pd.DataFrame(results)
    df = df.set_index("date")
    return df


def _classify_regime(score):
    if score >= REGIME_THRESHOLDS["RISKON"]:
        return "RISKON"
    elif score >= REGIME_THRESHOLDS["NEUTRAL"]:
        return "NEUTRAL"
    elif score >= REGIME_THRESHOLDS["RISKOFF"]:
        return "RISKOFF"
    else:
        return "CRISIS"


def _regime_to_num(regime):
    return {"RISKON": 3, "NEUTRAL": 2, "RISKOFF": 1, "CRISIS": 0}.get(regime, 2)


def _compute_breadth(etf_prices_dict, date):
    """计算 ETF 广度：有多少 ETF 站上 SMA50"""
    count_above = 0
    count_total = 0
    for code, prices in etf_prices_dict.items():
        if isinstance(prices, pd.Series):
            hist = prices[prices.index <= date]
        elif isinstance(prices, pd.DataFrame):
            hist = prices[prices.index <= date]
            if 'close' in hist.columns:
                hist = hist['close']
        else:
            continue

        if len(hist) < 50:
            continue
        sma50 = hist.iloc[-50:].mean()
        if hist.iloc[-1] > sma50:
            count_above += 1
        count_total += 1

    if count_total == 0:
        return 0.5
    pct = count_above / count_total
    return 1.0 if pct > 0.5 else (0.0 if pct < 0.25 else 0.5)


def _compute_correlation_score(etf_prices_dict, date):
    """计算平均相关系数"""
    returns_dict = {}
    for code, prices in etf_prices_dict.items():
        if isinstance(prices, pd.Series):
            hist = prices[prices.index <= date].iloc[-63:]
        elif isinstance(prices, pd.DataFrame):
            hist = prices[prices.index <= date].iloc[-63:]
            if 'close' in hist.columns:
                hist = hist['close']
        else:
            continue
        if len(hist) < 21:
            continue
        returns_dict[code] = hist.pct_change().dropna()

    if len(returns_dict) < 3:
        return 0.5

    df_ret = pd.DataFrame(returns_dict)
    corr_matrix = df_ret.corr()
    # 平均成对相关
    n = len(corr_matrix)
    if n < 2:
        return 0.5
    avg_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))

    # 高相关性 = 系统性风险 = 熊市信号
    if avg_corr > 0.6:
        return 0.0
    elif avg_corr < 0.3:
        return 1.0
    else:
        return 0.5
