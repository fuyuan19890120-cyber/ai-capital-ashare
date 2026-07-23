"""
策略一看板 — V4.1/V4.2/V4.3 三代对比 + 每日收益跟踪
启动: cd ~/ai-capital-ashare && streamlit run dashboard/app.py
每日更新: python3 ~/quant/daily_tracker.py
"""
import streamlit as st
import json, os
from datetime import datetime

st.set_page_config(page_title="策略一 V4 对比看板", page_icon="📊", layout="wide")

TRACKER_FILE = os.path.join(os.path.dirname(__file__), "tracker.json")

# === 每日跟踪数据 ===
tracker = None
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE) as f:
        tracker = json.load(f)

st.title("📊 策略一 · V4.1 → V4.2 → V4.3 三代对比")
st.caption(f"QMT 完整数据库 · 380只指数成分股 · 均含 SURGE · 更新: {tracker.get('updated', '—') if tracker else '—'}")

# === 三列核心指标 ===
c1, c2, c3 = st.columns(3)
c1.metric("V4.1 · 低波+动量", "16.0% 年化", "5.2x 净值")
c2.metric("V4.2 · +反转过滤", "15.0% 年化", "4.8x 净值")
c3.metric("⭐ V4.3 · Amihud×MomR²", "29.4% 年化", "17.6x 净值", delta_color="off")

# === 实盘跟踪 ===
st.divider()
st.subheader("📡 实盘跟踪 · 2026-06-30 起")

if tracker:
    td = tracker['daily']
    tc = tracker.get('csi300', {})

    col1, col2, col3, col4 = st.columns(4)
    for col, key, label in [
        (col1, 'v41', 'V4.1'), (col2, 'v42', 'V4.2'),
        (col3, 'v43', 'V4.3'), (col4, None, 'CSI300')
    ]:
        if key:
            d = td.get(key, {})
            col.metric(
                f"{label} 收益",
                f"{d.get('return_pct', 0):+.1f}%",
                f"回撤 {d.get('max_dd_pct', 0):.1f}% · 净值 {d.get('final_nav', 1):.3f}"
            )
        else:
            col.metric("CSI300", f"{tc.get('return_pct', 0):+.1f}%")

    # 逐日净值表
    with st.expander("📈 逐日净值"):
        if td:
            dates = [n['date'] for n in td['v43']['navs']]
            rows = []
            for i, d in enumerate(dates):
                row = {'日期': d}
                for k in ['v41', 'v42', 'v43']:
                    navs = td[k]['navs']
                    row[k.upper()] = f"{navs[i]['nav']:.4f}" if i < len(navs) else '—'
                if 'csi300' in tracker:
                    csi_navs = tracker['csi300'].get('navs', [])
                    row['CSI300'] = f"{csi_navs[i]['nav']:.4f}" if i < len(csi_navs) else '—'
                rows.append(row)
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index('日期'), use_container_width=True)

    # 当前持仓
    st.subheader("📋 6/30 选股 (持有至今)")
    rb = tracker['rebalances'][0]
    cols = st.columns(3)
    for col, key, label in [
        (cols[0], 'v41', '⬜ V4.1'),
        (cols[1], 'v42', '🟡 V4.2'),
        (cols[2], 'v43', '🔵 V4.3'),
    ]:
        stocks = rb[key]
        col.markdown(f"**{label}**")
        for s in stocks:
            col.caption(s)

    # 后续调仓
    if len(tracker['rebalances']) > 1:
        st.subheader("🔄 后续调仓")
        for rb in tracker['rebalances'][1:]:
            st.caption(f"**{rb['date']}**")
            cols = st.columns(3)
            for col, key in [(cols[0], 'v41'), (cols[1], 'v42'), (cols[2], 'v43')]:
                col.caption(', '.join(rb[key][:5]) + ('...' if len(rb[key]) > 5 else ''))
else:
    st.warning("⚠️ 尚未生成跟踪数据。运行 `python3 ~/quant/daily_tracker.py`")

# === 静态对比内容 ===
st.divider()
st.subheader("📈 逐年收益 (均含 SURGE)")
import pandas as pd
yearly = pd.DataFrame({
    "年份": [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026],
    "V4.1": [-1.9,12.5,-6.0,19.4,33.5,35.8,1.5,10.7,-2.7,49.2,23.4],
    "V4.2": [6.9,27.7,-8.7,18.3,42.9,31.4,1.5,3.7,5.4,22.7,12.8],
    "V4.3": [-3.4,18.8,0.8,27.9,101.1,43.8,1.5,0.2,21.1,111.1,57.5],
    "CSI300": [-4.6,20.6,-26.3,38.0,25.5,-6.2,-21.3,-11.7,16.2,21.2,0.2],
}).set_index("年份")
st.dataframe(yearly.style.applymap(lambda x: 'color:#3fb950' if x>0 else 'color:#f85149'), use_container_width=True)

st.subheader("📊 演进归因")
st.markdown("""
| 改动 | 版本 | 效果 |
|------|------|------|
| 基线 | V4.1 | 低波43%是死代码(年化0.4%)，纯靠动量14%撑着 |
| +反转过滤 | V4.2 | 降7pp回撤，代价-1pp年化。熊市保护，牛市拖累 |
| 因子替换 | V4.3 | 低波→Amihud，原始动量→MomR²。信息源正交，排名融合1+1>2 |
| 反转移除 | V4.3 | MomR²的R²已做趋势质量过滤，再反转=双重过滤砍有效信号 |
| SURGE协同 | V4.3 | V4.3因子在SURGE加速期选股更准(+2.9pp vs +0.6pp) |
""")

st.subheader("🔬 单因子有效性")
st.dataframe(pd.DataFrame({
    "因子": ["动量×R²","原始动量","Amihud","低波63日"],
    "年化": ["17.1%","16.1%","13.6%","0.4%"],
    "V4.1": ["","✅45%","","✅43%"],
    "V4.2": ["","✅45%","","✅43%"],
    "V4.3": ["✅70%","","✅30%","❌"],
}).set_index("因子"), use_container_width=True)

st.caption("数据源: QMT完整数据库 · 380只指数成分股 · IPO截断 · 双Agent审计(2026-07-23/24)")
st.caption("可信口径: V4.3约20-24%(扣除成分股前视偏差后)")
