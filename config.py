"""
全局参数配置
"""

# ============================================================
# 回测区间
# ============================================================
START_DATE = "2015-01-01"
END_DATE = "2026-07-13"

# ============================================================
# 数据缓存目录
# ============================================================
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "backtests")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKTEST_DIR, exist_ok=True)

# ============================================================
# 选股因子权重 (v2 优化版)
# ============================================================
# 时序信号：反转 + 动量 + 低波 四因子复合
REVERSAL_WEIGHT_1M = 0.30   # 1个月反转（抄底）
MOMENTUM_WEIGHT_3M = 0.35   # 3个月动量（趋势跟随）
MOMENTUM_WEIGHT_6M = 0.25   # 6个月动量（长期趋势）
LOWVOL_WEIGHT = 0.10        # 低波动率偏好

# 时序 vs 截面合成
TS_WEIGHT = 0.50             # 时序得分权重
CS_WEIGHT = 0.50             # 截面排名权重

# 动量回溯窗口（交易天数）
LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126

# ============================================================
# 制度检测因子权重 (v2 优化版)
# ============================================================
REGIME_WEIGHTS = {
    "sma200_vs_price": 0.10,     # 价格 vs SMA200
    "sma200_slope": 0.08,        # SMA200 斜率
    "market_breadth": 0.15,      # 市场广度
    "volatility": 0.15,          # 年化波动率异常
    "max_drawdown": 0.15,        # 最大回撤
    "correlation": 0.05,         # 平均相关系数
    "volume_momentum": 0.15,     # 成交额动能
    "deviation_60ma": 0.17,      # 60日均线乖离率
}

# 制度 → 仓位映射
REGIME_ALLOCATION = {
    "RISKON":    {"risk": 1.00, "defensive": 0.00, "cash": 0.00},
    "NEUTRAL":   {"risk": 0.55, "defensive": 0.35, "cash": 0.10},
    "RISKOFF":   {"risk": 0.15, "defensive": 0.60, "cash": 0.25},
    "CRISIS":    {"risk": 0.00, "defensive": 0.95, "cash": 0.05},
}

# 制度分数阈值
REGIME_THRESHOLDS = {
    "RISKON": 0.70,
    "NEUTRAL": 0.50,
    "RISKOFF": 0.30,
    # below 0.30 = CRISIS
}

# ============================================================
# 仓位风控参数 (v4 最终版 — 微观风控升级)
# ============================================================

# --- 个股级风控（新增！2026-07-14）---
# 单只个股权重上限（占组合总值）
MAX_SINGLE_POSITION = 0.15
# 单板块权重上限（科创板/创业板/沪市主板/深市主板）
MAX_SECTOR_PCT = 0.40
# 最小现金保留（元）
MIN_CASH = 1000.0

# 个股止损（从买入价下跌超过此值强制卖出）
STOP_LOSS_PCT = 0.15

# --- 组合回撤熔断（新增！）---
# 从历史峰值回撤超过此值 → 新仓位减半
DRAWDOWN_WARNING = 0.15
# 回撤超过此值 → 禁止新买入
DRAWDOWN_HALT = 0.20
# 回撤超过此值 → 强制减仓至 50% 现金
DRAWDOWN_LIQUIDATE = 0.25

# --- 原有参数 ---
VOL_TARGET = 0.12               # 组合波动率目标（12%）
MAX_LEVERAGE = 1.0              # 不杠杆
MAX_CORRELATION = 0.65          # 相关性过滤阈值
TOP_N_SELECTION = 15            # 选股数量（更新为个股版 Top-15）
VOL_LOOKBACK = 63               # 波动率计算窗口

# CRISIS 熔断条件
CRISIS_VOL_THRESHOLD = 0.25     # 年化波动率 > 25%
CRISIS_DEVIATION_THRESHOLD = -0.15  # 乖离率 < -15%

# ============================================================
# 交易成本
# ============================================================
COMMISSION = 0.00025            # 佣金 万2.5
STAMP_DUTY = 0.0                # ETF 免印花税
SLIPPAGE = 0.001               # 滑点 0.1%

# ============================================================
# 资产池定义
# ============================================================
UNIVERSE = [
    {"name": "沪深300",    "code": "sh510300", "type": "risk",     "ak_code": "510300", "market": "sh"},
    {"name": "中证500",    "code": "sh510500", "type": "risk",     "ak_code": "510500", "market": "sh"},
    {"name": "创业板",     "code": "sz159915", "type": "risk",     "ak_code": "159915", "market": "sz"},
    {"name": "科创50",     "code": "sh588000", "type": "risk",     "ak_code": "588000", "market": "sh"},
    {"name": "国债ETF",    "code": "sh511010", "type": "defensive","ak_code": "511010", "market": "sh"},
    {"name": "企业债ETF",  "code": "sh511220", "type": "defensive","ak_code": "511220", "market": "sh"},
    {"name": "黄金ETF",    "code": "sh518880", "type": "defensive","ak_code": "518880", "market": "sh"},
    {"name": "货币ETF",    "code": "sh511880", "type": "defensive","ak_code": "511880", "market": "sh"},
]

# 制度检测参考指数
BENCHMARK_INDEX = "sh000300"    # 沪深300作为基准
