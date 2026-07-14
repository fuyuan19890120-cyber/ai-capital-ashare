"""
个股级风控模块

提供四层微观保护（在制度检测的宏观仓位之外）：
1. 单只个股权重上限 — 任何股票不超过组合的 max_position_pct
2. 单板块权重上限 — 科创板/创业板不过度集中
3. 组合回撤熔断 — 从峰值回撤超阈值 → 降仓/停买/强平
4. 个股止损 — 从买入价下跌超阈值 → 强制卖出

所有参数在 config.py 中配置。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    """风控参数配置"""
    # 单只个股权重上限（占组合总值的百分比）
    max_position_pct: float = 0.15

    # 单板块权重上限
    # A股板块分类：沪市主板 / 深市主板 / 创业板 / 科创板
    max_sector_pct: float = 0.40

    # 组合回撤熔断
    drawdown_warning: float = 0.15    # -15%：新仓位减半
    drawdown_halt: float = 0.20       # -20%：禁止新买
    drawdown_liquidate: float = 0.25  # -25%：强制减仓至50%

    # 个股止损（从买入价计算）
    stop_loss_pct: float = 0.15       # -15% 止损

    # 最小现金保留（元）
    min_cash: float = 1000.0


def classify_sector(code: str) -> str:
    """
    A股板块分类。

    - 688xxx → 科创板
    - 300xxx, 301xxx → 创业板
    - 600xxx, 601xxx, 603xxx, 605xxx → 沪市主板
    - 000xxx, 001xxx, 002xxx, 003xxx → 深市主板
    - 其他 → 未知
    """
    if code.startswith('688'):
        return '科创板'
    elif code.startswith(('300', '301')):
        return '创业板'
    elif code.startswith(('600', '601', '603', '605')):
        return '沪市主板'
    elif code.startswith(('000', '001', '002', '003')):
        return '深市主板'
    else:
        return '其他'


class DrawdownController:
    """
    组合回撤控制器。

    追踪组合历史峰值，根据回撤深度发出不同级别的风控信号：
    - Warning (-15%): 新仓位减半
    - Halt (-20%): 禁止新买入
    - Liquidate (-25%): 强制减仓至 50% 现金
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.peak_value: float = 0.0
        self.current_drawdown: float = 0.0

    def update(self, current_value: float):
        """更新峰值和回撤"""
        if current_value > self.peak_value:
            self.peak_value = current_value
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

    @property
    def position_multiplier(self) -> float:
        """新买入仓位的缩放系数"""
        if self.current_drawdown >= self.limits.drawdown_halt:
            return 0.0   # 停买
        elif self.current_drawdown >= self.limits.drawdown_warning:
            return 0.5   # 减半
        return 1.0

    @property
    def should_liquidate(self) -> bool:
        """是否需要强制减仓"""
        return self.current_drawdown >= self.limits.drawdown_liquidate

    def status_text(self) -> str:
        """人类可读的回撤状态"""
        if self.current_drawdown >= self.limits.drawdown_liquidate:
            return f"🚨 强平 ({self.current_drawdown:.1%})"
        elif self.current_drawdown >= self.limits.drawdown_halt:
            return f"🛑 停买 ({self.current_drawdown:.1%})"
        elif self.current_drawdown >= self.limits.drawdown_warning:
            return f"⚠️ 减半 ({self.current_drawdown:.1%})"
        return f"✅ 正常 ({self.current_drawdown:.1%})"


class RiskManager:
    """
    组合层面的风险管理器。

    职责：
    - 验证每笔买单是否符合权重/板块/回撤约束
    - 追踪个股止损
    - 计算板块分布
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.drawdown = DrawdownController(self.limits)
        # 个股止损追踪: {code: entry_price}
        self.entry_prices: Dict[str, float] = {}

    # ============================================================
    # 买入前验证
    # ============================================================

    def validate_buy(
        self,
        code: str,
        proposed_weight: float,
        current_holdings: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        cash_balance: float,
    ) -> Tuple[bool, str]:
        """
        验证一笔买入是否合规。

        Args:
            code: 股票代码
            proposed_weight: 提议仓位权重（0-1）
            current_holdings: {code: shares}
            current_prices: {code: price}
            portfolio_value: 组合总市值
            cash_balance: 当前现金

        Returns:
            (is_valid, reason)
        """
        trade_value = proposed_weight * portfolio_value

        # 1. 现金检查
        if trade_value > cash_balance:
            return False, f"现金不足: 需¥{trade_value:,.0f}, 有¥{cash_balance:,.0f}"

        # 2. 单只权重上限
        if proposed_weight > self.limits.max_position_pct:
            return False, f"单只权重 {proposed_weight:.1%} 超上限 {self.limits.max_position_pct:.1%}"

        # 3. 板块权重上限
        sector = classify_sector(code)
        sector_weight = self._sector_weight(sector, current_holdings, current_prices, portfolio_value)
        new_sector_weight = sector_weight + proposed_weight
        if new_sector_weight > self.limits.max_sector_pct:
            return False, (
                f"板块「{sector}」权重将达 {new_sector_weight:.1%}，"
                f"超上限 {self.limits.max_sector_pct:.1%}"
            )

        # 4. 回撤熔断
        if self.drawdown.should_liquidate:
            return False, "组合回撤已触发强平线，禁止买入"

        if self.drawdown.position_multiplier == 0.0:
            return False, "组合回撤已触发停买线，禁止买入"

        # 5. 最小现金保留
        remaining = cash_balance - trade_value
        if remaining < self.limits.min_cash:
            return False, f"交易后现金 ¥{remaining:,.0f} 低于最低 ¥{self.limits.min_cash:,.0f}"

        return True, "合规"

    # ============================================================
    # 止损检查
    # ============================================================

    def check_stop_loss(
        self, code: str, current_price: float
    ) -> Tuple[bool, str]:
        """
        检查某只股票是否触及止损。

        Returns:
            (should_sell, reason)
        """
        if code not in self.entry_prices:
            return False, ""
        entry = self.entry_prices[code]
        if entry <= 0:
            return False, ""
        loss = (current_price - entry) / entry
        if loss <= -self.limits.stop_loss_pct:
            return True, f"止损触发: {loss:.1%} (入场 {entry:.2f}, 现价 {current_price:.2f})"
        return False, ""

    def check_all_stop_losses(
        self, current_prices: Dict[str, float]
    ) -> List[str]:
        """返回所有触发止损的股票代码"""
        triggers = []
        for code in self.entry_prices:
            if code in current_prices:
                hit, _ = self.check_stop_loss(code, current_prices[code])
                if hit:
                    triggers.append(code)
        return triggers

    def record_entry(self, code: str, price: float):
        """记录买入价"""
        self.entry_prices[code] = price

    def remove_entry(self, code: str):
        """移除记录（卖出时调用）"""
        self.entry_prices.pop(code, None)

    # ============================================================
    # 板块分析
    # ============================================================

    def _sector_weight(
        self,
        sector: str,
        holdings: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
    ) -> float:
        """计算某板块在组合中的权重"""
        if portfolio_value <= 0:
            return 0.0
        value = 0.0
        for code, shares in holdings.items():
            if classify_sector(code) == sector:
                value += shares * prices.get(code, 0.0)
        return value / portfolio_value

    def sector_breakdown(
        self, holdings: Dict[str, float], prices: Dict[str, float], portfolio_value: float
    ) -> Dict[str, float]:
        """返回各板块权重分布"""
        sectors = {}
        for code, shares in holdings.items():
            s = classify_sector(code)
            val = shares * prices.get(code, 0.0)
            sectors[s] = sectors.get(s, 0.0) + val
        if portfolio_value <= 0:
            return {}
        return {s: v / portfolio_value for s, v in sorted(sectors.items(), key=lambda x: -x[1])}

    # ============================================================
    # 综合风控报告
    # ============================================================

    def risk_report(
        self,
        holdings: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float,
        cash_balance: float,
    ) -> dict:
        """生成风控状态报告"""
        self.drawdown.update(portfolio_value)

        # 个股止损
        stops = self.check_all_stop_losses(prices)

        # 板块分布
        sectors = self.sector_breakdown(holdings, prices, portfolio_value)

        # 最大单票权重
        max_weight = 0.0
        max_code = ""
        for code, shares in holdings.items():
            w = shares * prices.get(code, 0.0) / portfolio_value if portfolio_value > 0 else 0
            if w > max_weight:
                max_weight = w
                max_code = code

        return {
            'drawdown': {
                'current': round(self.drawdown.current_drawdown * 100, 1),
                'status': self.drawdown.status_text(),
                'position_multiplier': self.drawdown.position_multiplier,
                'peak': round(self.drawdown.peak_value, 0),
            },
            'concentration': {
                'max_single_stock': f"{max_weight:.1%}",
                'max_code': max_code,
                'n_holdings': len(holdings),
            },
            'sectors': {s: f"{w:.1%}" for s, w in sectors.items()},
            'stop_losses': stops,
            'alerts': self._generate_alerts(portfolio_value, cash_balance, stops),
        }

    def _generate_alerts(
        self, portfolio_value: float, cash_balance: float, stops: List[str]
    ) -> List[str]:
        """生成风控警报列表"""
        alerts = []
        if self.drawdown.current_drawdown >= self.limits.drawdown_warning:
            alerts.append(f"回撤 {self.drawdown.current_drawdown:.1%} ≥ 警告线 {self.limits.drawdown_warning:.0%}")
        if stops:
            alerts.append(f"{len(stops)}只触发止损: {', '.join(stops)}")
        cash_pct = cash_balance / portfolio_value if portfolio_value > 0 else 0
        if cash_pct < 0.02:
            alerts.append(f"现金占比仅 {cash_pct:.1%}，无缓冲空间")
        return alerts
