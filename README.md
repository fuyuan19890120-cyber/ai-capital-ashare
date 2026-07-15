# AI Capital A-Share — 量化策略 v4

> SMA250 制度择时 + 广度SMA30锁仓进场 + 四因子精选个股 + 月度调仓

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 策略概述

以 **SMA250** 判断 A 股牛熊，**广度SMA30** 提前捕捉趋势反转，**四因子**（低波+价值+质量+动量）从 CSI300+创业板+科创50 的 ~380 只股票中精选 Top-15，月度调仓。

| 指标 | 数值 |
|------|:---:|
| 年化收益 | **25.3%** |
| 最大回撤 | −29.2% |
| 夏普比率 | **0.94** |
| 总收益（2015-2026） | **1124%** |
| 跑赢沪深300年数 | 8/11 |

---

## 快速开始

```bash
git clone https://github.com/fuyuan19890120-cyber/ai-capital-ashare.git
cd ai-capital-ashare
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 生成月度信号
python run_monthly.py

# 启动 Dashboard
streamlit run dashboard/app.py --server.port 8501
```

---

## 项目结构

```
├── run_monthly.py            # ⭐ 月度信号主入口
├── refresh_data.py           # 数据刷新
├── config.py                 # 全局参数
├── dashboard/                # Streamlit 仪表盘
│   └── app.py
├── src/
│   ├── signal_generator.py   # 信号生成引擎
│   ├── stock_backtest.py     # 个股回测引擎
│   ├── stock_data.py         # 个股数据+因子计算
│   ├── factor_monitor.py     # 因子监控 (Quant-Zero)
│   ├── backtest_engine.py    # ETF回测引擎
│   ├── regime.py             # 8因子制度检测
│   ├── obsidian_logger.py    # Obsidian 自动记录
│   └── return_tracker.py     # 收益追踪
├── data/stocks/              # 个股CSV缓存
└── signals/                  # 信号输出
```

---

## 策略逻辑

### 第一层：制度检测

| 制度 | 条件 | 权益仓位 |
|------|:--:|:--:|
| 🟢 RISKON | SMA250趋势向上 | 95% |
| 🟡 NEUTRAL | 趋势震荡 | 60% |
| 🟠 RISKOFF | 趋势走弱 | 30% |
| 🔴 CRISIS | 暴跌 | 0% |

### 第二层：广度SMA30锁仓

当市场大面积回暖（3只ETF站上SMA50的比例15天内从<33%跳到>67%）且SMA30确认趋势时，强制提前进场，锁定21个交易日。

### 第三层：多因子选股

| 因子 | 权重 | 计算方式 |
|------|:--:|------|
| 低波动率 | 30% | 1/波动率(63日) |
| 价值 | 25% | 1/股价 |
| 质量 | 20% | 过去一年涨幅 |
| 动量 | 25% | 6月收益/波动率 |

从 CSI300+创业板+科创50 ≈380只中，四因子打分选出 Top-15，等权配置。

---

## 自动化

- **每月28日** launchd 自动运行 `auto_monthly.sh`
- 数据刷新 → 制度检测 → 选股 → 调仓清单 → 收益追踪 → 因子监控
- 信号/持仓/收益率自动写入 Obsidian

---

## 实盘对接

当前支持：
- **国金证券 MiniQMT**：10万门槛，Python xtquant SDK
- **东方财富掘金量化**：专业投资者认证，掘金SDK

---

## 致谢

- 策略灵感来自 [kabNath/AI-Capital](https://github.com/kabNath/AI-Capital)
- 因子监控方法论来自 [marcohwlam/quant-zero](https://github.com/marcohwlam/quant-zero)
- 数据源：AKShare + Sina

---

> ⚠️ **免责声明**：历史回测不代表未来收益。量化策略存在失效风险，实盘前请充分验证。
