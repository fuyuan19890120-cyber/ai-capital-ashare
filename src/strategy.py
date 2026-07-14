"""
Backtrader 策略类
"""
import backtrader as bt
import pandas as pd
import numpy as np
from config import (
    TOP_N_SELECTION, MAX_CORRELATION, COMMISSION, SLIPPAGE,
    REGIME_ALLOCATION, REGIME_THRESHOLDS,
    CRISIS_VOL_THRESHOLD, CRISIS_DEVIATION_THRESHOLD,
)
from src.momentum import compute_final_scores, select_top_assets


class NaiveMomentumStrategy(bt.Strategy):
    """
    Phase 2 基线策略：裸动量选股（无制度过滤）

    月度调仓，选 Top-3，等权配置
    目的：验证裸动量在 A 股是否亏损
    """

    params = dict(
        rebalance_day=1,        # 每月第1个交易日调仓（近似月末）
        top_n=TOP_N_SELECTION,
        max_corr=MAX_CORRELATION,
        lookback_days=252,      # 动量计算需要的历史数据
    )

    def __init__(self):
        self.month_counter = 0
        self.last_rebalance_month = -1
        # 价格缓存用于动量计算
        self.price_history = {}
        for data in self.datas:
            symbol = data._name
            self.price_history[symbol] = []

    def next(self):
        # 收集每日价格
        for data in self.datas:
            symbol = data._name
            self.price_history[symbol].append(data.close[0])

        # 每月调仓
        current_month = self.datas[0].datetime.date(0).month
        if current_month == self.last_rebalance_month:
            return
        self.last_rebalance_month = current_month

        self.month_counter += 1

        # 需要足够历史数据
        if self.month_counter < 12:
            return

        # 构建价格 DataFrame
        prices_dict = {}
        for data in self.datas:
            symbol = data._name
            prices = pd.Series(self.price_history[symbol])
            if len(prices) > self.p.lookback_days:
                prices_dict[symbol] = prices

        if len(prices_dict) < 3:
            return

        prices_df = pd.DataFrame(prices_dict)

        # 计算动量得分
        try:
            scores = compute_final_scores(prices_df)
        except Exception:
            return

        # 获取最后一天日期对应的得分
        last_idx = scores.index[-1]
        date_str = str(last_idx.date()) if hasattr(last_idx, 'date') else str(last_idx)

        # 选股
        try:
            selected = select_top_assets(scores, prices_df, last_idx, self.p.top_n, self.p.max_corr)
        except Exception:
            return

        if not selected:
            return

        # 等权下单
        weight = 0.95 / len(selected)  # 留5%现金缓冲

        # 平掉不在选中的仓位
        for data in self.datas:
            symbol = data._name
            pos = self.getposition(data)
            if symbol not in selected and pos.size > 0:
                self.close(data)

        # 开/调选中仓位
        for data in self.datas:
            symbol = data._name
            if symbol in selected:
                self.order_target_percent(data, target=weight)


class AdaptiveMomentumStrategy(bt.Strategy):
    """
    Phase 3+ 完整策略：制度检测 + 动量选股 + 波动率仓位管理

    月度调仓 + 每日 CRISIS 检测
    """
    # 暂时沿用 Naive 的逻辑，Phase 3 会大幅扩展
    params = dict(
        rebalance_day=1,
        top_n=TOP_N_SELECTION,
        max_corr=MAX_CORRELATION,
        lookback_days=252,
    )

    def __init__(self):
        self.month_counter = 0
        self.last_rebalance_month = -1
        self.current_regime = "NEUTRAL"  # 默认
        self.price_history = {}
        self.benchmark_history = []  # 沪深300价格（用于制度检测）
        for data in self.datas:
            symbol = data._name
            self.price_history[symbol] = []

    def log(self, msg):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt} | {msg}')

    def next(self):
        for data in self.datas:
            symbol = data._name
            self.price_history[symbol].append(data.close[0])

        current_month = self.datas[0].datetime.date(0).month
        if current_month == self.last_rebalance_month:
            return
        self.last_rebalance_month = current_month
        self.month_counter += 1

        if self.month_counter < 12:
            return

        prices_dict = {}
        for data in self.datas:
            symbol = data._name
            prices = pd.Series(self.price_history[symbol])
            if len(prices) > self.p.lookback_days:
                prices_dict[symbol] = prices

        if len(prices_dict) < 3:
            return

        prices_df = pd.DataFrame(prices_dict)
        try:
            scores = compute_final_scores(prices_df)
            last_idx = scores.index[-1]
            selected = select_top_assets(scores, prices_df, last_idx, self.p.top_n, self.p.max_corr)
        except Exception:
            return

        if not selected:
            return

        weight = 0.95 / len(selected)

        for data in self.datas:
            symbol = data._name
            pos = self.getposition(data)
            if symbol not in selected and pos.size > 0:
                self.close(data)

        for data in self.datas:
            symbol = data._name
            if symbol in selected:
                self.order_target_percent(data, target=weight)
