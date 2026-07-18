# -*- coding: utf-8 -*-
"""可转债日内研究 - 全局配置"""
import os

# 项目根目录(worktree/主仓通用: 从本文件向上两级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 分钟数据档案目录(gitignore, 每日采集滚动积累)
MIN_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cb_min")

# 研究输出目录(报告/图表, 入库)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ============================================================
# 交易成本(单边, bp; 可转债 T+0 无印花税)
# ============================================================
COMMISSION_BPS = 1.0   # 佣金: 万1(转债佣金普遍万0.5~万2, 按券商实际调整)
SLIPPAGE_BPS = 5.0     # 冲击成本: 默认 5bp/边, 敏感性分析扫 0~15bp

# ============================================================
# 采集参数
# ============================================================
FREQS_SINA = {"5": 5, "15": 15, "30": 30, "60": 60}   # 新浪 scale
MIN_AMOUNT_FILTER = 0  # 采集不做流动性过滤, 研究时再过滤
