# 可转债日内策略研究基建 (cb_intraday)

可转债是 A 股散户唯一现实的日内标的(T+0、无印花税),本模块提供从数据采集到成本敏感回测的完整研究管线。与已验证的[双低周频策略](../../backtests/)共享标的池,是其向日内频段的自然延伸。

## 架构

```
sources.py   数据源适配层  新浪(主力)/东财/腾讯 直连行情API, 全局限速+重试+代理回退
store.py     存储层        parquet 按 频率/代码 分文件, 增量upsert去重 → data/cb_min/
collect.py   采集器        CLI, 全市场在市转债, 单券失败不阻塞
costs.py     成本模型      佣金+滑点(bp), 三档敏感性场景
engine.py    回测引擎      bar级、无前视(信号bar收盘→次bar开盘成交)、纯日内T+0收盘强平
signals_demo.py  基线信号  首小时动量/反转、尾盘动量 —— 管线验证+基准线
run_demo.py  全链路演示    三策略 × 三档成本 → output/demo_results.md
```

关键设计约束见 [DATA_QUALITY.md](DATA_QUALITY.md):**免费源没有长分钟历史,档案靠每日采集滚动积累**。

## 快速开始

依赖:`pandas / pyarrow / requests`(venv 已有)+ `pysocks`(2026-07-18 已装入 venv,本机东财走 SOCKS 代理必需,代理地址用 `CB_PROXY` 环境变量覆盖,默认 `socks5h://127.0.0.1:1081`)。

```bash
# 1. 采集(首次约15分钟, 之后增量很快)
venv/bin/python -m research.cb_intraday.collect --freq 60,15,5

# 2. 跑演示回测(三个基线信号 × 三档成本)
venv/bin/python -m research.cb_intraday.run_demo

# 3. 档案盘点
venv/bin/python -c "from research.cb_intraday import store; print(store.archive_summary('60').describe())"
```

## 采集节奏(重要)

| 频率 | 源窗口 | 不断档要求 |
|---|---|---|
| 5分钟 | 21个交易日 | ≤2周跑一次 `--freq 5` |
| 15/60分钟 | 3个月/1年 | 每月一次即可 |
| 1分钟 | **仅当日** | 每个交易日收盘后 `--freq 1`(东财源,本机需代理,建议部署 GitHub Actions) |

建议 crontab(工作日 15:30):
```
30 15 * * 1-5 cd ~/ai-capital-ashare && venv/bin/python -m research.cb_intraday.collect --freq 60,15,5 >> data/cb_min/collect.log 2>&1
```

## 回测引擎约定

- 信号函数: `signal_fn(bars) -> DataFrame[symbol, datetime, strength]`,datetime 为信号产生的 bar
- 引擎自动: 次 bar 开盘买入 → 当日最后 bar 收盘卖出(纯日内,不留隔夜)
- 当日信号按 strength 取前 `max_positions` 只等权
- 所有结论输出三档成本对照(乐观/基准/保守),口径与隔夜策略研究一致

## 研究路线(下一步)

**基准线已立(2026-07-18,319 只 × 60 分钟 × ~1 年)**:首小时动量 / 首小时反转 / 尾盘动量三个朴素信号,三档成本下**费后全部为负**(每笔 -0.25%~-0.5%,详见 [output/demo_results.md](output/demo_results.md))。与隔夜研究"简单信号全灭"的结论一致——日内换手更高,成本门槛更狠。后续信号研究以打赢这条负基线为最低要求。

1. **60分钟×1年 信号粗筛**(现在就能做):日内动量族、正股-转债联动(转债分钟跟正股)、双低池内日内择时
2. **成本细化**:用 5 分钟档案估计真实滑点分布,替换固定 bp 假设
3. **1分钟档案**:积累 3 个月后做入场时点优化;若 60 分钟级别信号全灭,及时止损这条线
4. 教训前置([隔夜研究](../../backtests/overnight_gap_breadth_report.md)):"每天交易"是亏钱根源,日内策略同样优先找稀疏高置信度触发条件

## 已知局限

- 未建模转债涨跌停(±20%)与临停机制,强信号日的可成交性偏乐观
- 滑点为固定 bp 近似,小盘妖债实际冲击远大于此 → 用成交额过滤 + 保守档成本兜底
- 新浪分钟成交量单位与日线不一致,流动性一律用成交额(代码已遵守)
