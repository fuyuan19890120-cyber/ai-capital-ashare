# -*- coding: utf-8 -*-
"""
可转债日内研究 - 交易成本模型

可转债 T+0、无印花税, 成本 = 佣金 + 冲击(滑点):
  - 佣金: 万0.5~万2/边(券商可谈), 默认万1
  - 冲击: 与下单量/盘口深度相关; bar 级回测用固定 bp 近似, 敏感性扫描 0~15bp
教训回顾: 隔夜策略研究里 15 类信号"费前全活、费后全灭" —— 日内换手更高,
成本假设必须比隔夜研究更保守, 所有结论默认展示三档成本下的结果。
"""


class CostModel:
    def __init__(self, commission_bps: float = 1.0, slippage_bps: float = 5.0):
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    @property
    def one_side(self) -> float:
        """单边成本(小数), 如 6bp -> 0.0006"""
        return (self.commission_bps + self.slippage_bps) / 10000.0

    @property
    def round_trip(self) -> float:
        """双边成本(小数)"""
        return 2.0 * self.one_side

    def __repr__(self):
        return f"CostModel(佣金{self.commission_bps}bp+滑点{self.slippage_bps}bp/边, 双边{self.round_trip:.4%})"


# 敏感性分析三档: 乐观 / 基准 / 保守
COST_SCENARIOS = {
    "乐观(万0.5佣金+2bp滑点)": CostModel(0.5, 2.0),
    "基准(万1佣金+5bp滑点)": CostModel(1.0, 5.0),
    "保守(万2佣金+10bp滑点)": CostModel(2.0, 10.0),
}
