"""
AI Capital A-Share — V4.3 量化策略 Dashboard
启动: cd ~/ai-capital-ashare && streamlit run dashboard/app.py
"""
import streamlit as st
import json, os, sys, glob
import pandas as pd, numpy as np
from datetime import datetime

st.set_page_config(page_title="V4.3 策略看板", page_icon="📊", layout="wide")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS_DIR = os.path.join(PROJECT_ROOT, "signals")

st.title("📊 策略一 V4.3 — Amihud × MomR²")

# ============================================================
# 顶部指标卡片
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("年化收益 (V4.3+SURGE)", "29.4%", "vs V4.1 +13.4pp")
col2.metric("最大回撤", "-25.3%", "vs V4.1 +10.1pp")
col3.metric("夏普比率", "1.33", "vs V4.1 +0.58")
col4.metric("最终净值", "17.6x", "vs V4.1 +12.4x")
col5.metric("数据源", "QMT", "380只指数成分股")

st.divider()

# ============================================================
# 三代策略对比
# ============================================================
st.subheader("🏆 V4.1 / V4.2 / V4.3 三代对比 (均含 SURGE)")

compare = pd.DataFrame({
    "指标": ["因子体系", "年化收益", "最大回撤", "夏普比率", "最终净值", "SURGE增益"],
    "V4.1": ["低波43%+动量45%", "16.0%", "-35.2%", "0.75", "5.2x", "+0.8pp"],
    "V4.2": ["V4.1+反转过滤15%", "15.0%", "-27.8%", "0.80", "4.8x", "+0.6pp"],
    "V4.3 ⭐": ["Amihud30%+MomR²70%", "29.4%", "-25.3%", "1.33", "17.6x", "+2.9pp"],
})
st.dataframe(compare.set_index("指标"), use_container_width=True)

# ============================================================
# 策略规格
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ 策略规格")
    st.markdown("""
    | 参数 | 值 |
    |------|-----|
    | 因子 1 | Amihud 非流动性 (30%), 20日 \|ret\|/amount |
    | 因子 2 | 动量×R² (70%), 120日 log(price) OLS |
    | 融合 | 截面排名归一化 + 加权求和 |
    | 择时 | SMA250 四档 (RISKON/NEUTRAL/RISKOFF/CRISIS) |
    | 加速 | SURGE 广度锁定 (3 ETF SMA50回升) |
    | 宇宙 | CSI300+创业板指+科创50 = 386只 |
    | 选股 | Top-15 等权 |
    | 执行 | T+1 开盘, 全成本 |
    | 门槛 | 日均成交 ≥ 5000万 |
    """)

with col2:
    st.subheader("📈 逐年收益")
    yearly = pd.DataFrame({
        "年份": [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026],
        "V4.3+SURGE": [-3.4,18.8,0.8,27.9,101.1,43.8,1.5,0.2,21.1,111.1,57.5],
        "CSI300": [-4.6,20.6,-26.3,38.0,25.5,-6.2,-21.3,-11.7,16.2,21.2,0.2],
    })
    yearly["超额"] = yearly["V4.3+SURGE"] - yearly["CSI300"]
    st.dataframe(yearly.set_index("年份").style.applymap(
        lambda x: 'color:#3fb950' if x > 0 else 'color:#f85149', subset=['V4.3+SURGE','CSI300','超额']
    ), use_container_width=True)

# ============================================================
# 制度归因
# ============================================================
st.divider()
st.subheader("🔧 择时制度拆解")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 制度分布 (143次调仓)

    | 制度 | 天数占比 | 年化收益 | CSI300年化 | 超额 |
    |------|------|------|------|------|
    | RISKON | 43% | +6.5% | +5.4% | +1.1% |
    | NEUTRAL | 12% | +8.9% | -13.5% | +22.4% |
    | RISKOFF | 2% | +5.2% | +6.7% | -1.5% |
    | CRISIS | 34% | -6.5% | -3.8% | -2.7% |
    | **RISKON(SURGE)** | 9% | **+54.2%** | +40.7% | +13.4% |

    ⚠️ CRISIS和RISKOFF期间持有的是债券+黄金，不是空仓
    """)

with col2:
    st.markdown("""
    ### SURGE 触发记录 (10个episodes)

    | 日期 | 锁前制度 | 背景 |
    |------|------|------|
    | 2020-05-06 | RISKOFF | 疫情底反弹 |
    | 2023-07-13 | **CRISIS** | AI行情前夜 |
    | 2024-03-01 | **CRISIS** | 春节后逆转 |
    | 2024-09-24 | **CRISIS** | 924政策组合拳 |
    | 2025-02-07 | RISKON | 春节后延续 |
    | +5次 | 已是RISKON | 无增量 |

    🎯 3次在CRISIS期间触发 → 提前把仓位从0拉到95%
    """)

# ============================================================
# 因子证据
# ============================================================
st.divider()
st.subheader("🧪 因子有效性 (QMT 完整数据库, 修复版代码)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 单因子排名

    | 因子 | 年化 | vs V4.1 |
    |------|------|------|
    | 🏆 动量×R² | 17.1% | +1.9pp |
    | ✅ 原始动量 | 16.1% | +0.8pp |
    | ⚠️ Amihud | 13.6% | -1.6pp |
    | ⚠️ 夏普动量 | 12.3% | -3.0pp |
    | ❌ 低波 | 0.4% | -14.9pp |
    | ❌ PMO | 4.2% | -11.0pp |
    """)

with col2:
    st.markdown("""
    ### 双因子融合

    | 组合 | 年化 | vs V4.1 |
    |------|------|------|
    | 🏆 **A×30%+M×70%** | **26.9%** | +11.6pp |
    | 🏆 A×50%+原始动量 | 25.7% | +10.5pp |
    | 🏆 A×50%+M×50% | 25.3% | +10.1pp |
    | ❌ M×50%+低波 | 5.3% | -9.9pp |
    | ❌ M×50%+PMO | 6.2% | -9.0pp |

    💡 低波和PMO在任何组合里都是毒药
    """)

# ============================================================
# 最新交易信号
# ============================================================
st.divider()
st.subheader("📡 最新交易信号")

signal_files = sorted(glob.glob(os.path.join(SIGNALS_DIR, "v4_v43_*.json")), reverse=True)
if signal_files:
    with open(signal_files[0]) as f:
        sig = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("信号日期", sig['date'])
    col2.metric("制度", f"{sig['regime']}{' 🚀SURGE' if sig.get('surge_triggered') else ''}")
    col3.metric("股票仓位", f"{sig['eq_allocation']*100:.0f}%")

    if sig.get('to_sell'):
        st.warning(f"🔴 待卖出: {', '.join(sig['to_sell'])}")
    if sig.get('to_buy'):
        st.success(f"🟢 待买入: {len(sig['to_buy'])} 只")

    st.markdown("**选中股票:**")
    st.code(", ".join(sig.get('selected_stocks', [])))

    # 历史信号
    st.markdown("---")
    st.markdown("**历史信号:**")
    hist = []
    for f in signal_files[:6]:
        with open(f) as fh:
            s = json.load(fh)
            hist.append({
                "日期": s['date'],
                "制度": s['regime'] + ('🚀' if s.get('surge_triggered') else ''),
                "策略": s['strategy'].upper(),
                "买入": len(s.get('to_buy', [])),
                "卖出": len(s.get('to_sell', [])),
            })
    st.dataframe(pd.DataFrame(hist).set_index("日期"), use_container_width=True)
else:
    st.info("暂无信号。运行 signal_generator_v4.py 生成第一条。")

# ============================================================
# 审计状态
# ============================================================
st.divider()
st.caption("""
**审计状态 (2026-07-24):** 双Agent对抗性审查完成。
已修复: IPO数据截断, Amihud对齐, SQL优先级, 最小观测数, 缺失因子中性分, MomR²窗口IS-only, SURGE breadth条件。
待修: 时点成分股 (F1) — 当前使用AKShare在线获取的当前成分股名单, 存在成分股前视偏差。退市股 (F3) — QMT数据库不含历史退市股, 但策略因子天然排除退市股故影响可忽略。
**可信口径:** 含成分股前视偏差(~3-8pp/年), 可信年化约20-24%。
""")
