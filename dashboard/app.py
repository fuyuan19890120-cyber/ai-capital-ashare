"""
策略一 三代对比看板
启动: cd ~/ai-capital-ashare && streamlit run dashboard/app.py
每日更新: python3 ~/quant/daily_tracker.py
"""
import streamlit as st
import json, os
from datetime import datetime

st.set_page_config(page_title="策略一 · V4 三代对比", page_icon="📊", layout="wide")

TRACKER_FILE = os.path.join(os.path.dirname(__file__), "tracker.json")

# ═══════════════════════════════════════════
# 股票名称缓存 (只加载一次)
# ═══════════════════════════════════════════
@st.cache_data(ttl=86400)
def get_stock_names():
    names = {}
    try:
        import akshare as ak
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for idx in ["000300", "399006", "000688"]:
            try:
                df = ak.index_stock_cons(symbol=idx)
                for _, row in df.iterrows():
                    names[str(row['品种代码']).zfill(6)] = row['品种名称']
            except:
                pass
    except:
        pass
    return names

STOCK_NAMES = get_stock_names()

def stock_display(code):
    """带名称展示: 300001.SZ → 300001 特锐德"""
    num = code.split('.')[0]
    name = STOCK_NAMES.get(num, '')
    return f"{code}  {name}" if name else code

# ═══════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════
tracker = None
if os.path.exists(TRACKER_FILE):
    with open(TRACKER_FILE) as f:
        tracker = json.load(f)

# ═══════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════
st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
    .main-sub { color: #8b949e; font-size: 13px; margin-bottom: 24px; }
    .tracking-start { color: #58a6ff; font-size: 13px; }
    .stMetric { padding: 12px !important; }
    div[data-testid="stMetricValue"] { font-size: 22px !important; }
    div[data-testid="stMetricDelta"] { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# 头部
# ═══════════════════════════════════════════
st.markdown('<div class="main-header">策略一 · V4.1 → V4.2 → V4.3</div>', unsafe_allow_html=True)
st.caption(f"QMT 完整数据库 · 380只指数成分股 · 均含 SURGE · 上月调仓: 2026-06-30 · 下次: 2026-07-31")

# ═══════════════════════════════════════════
# 实盘跟踪区 (从上次调仓日起)
# ═══════════════════════════════════════════
st.divider()

tb = tracker['daily'] if tracker else {}
tc = tracker.get('csi300', {}) if tracker else {}

if tracker:
    start_d = tracker.get('start', '2026-06-30')
    updated = tracker.get('updated', '—')
else:
    start_d, updated = '2026-06-30', '—'

st.subheader(f"📡 实盘跟踪 · {start_d} → {updated}")

col1, col2, col3, col4 = st.columns(4)

strategy_config = [
    ('v41', '⬜ V4.1 · 低波+动量', '#8b949e'),
    ('v42', '🟡 V4.2 · +反转过滤', '#d29922'),
    ('v43', '🔵 V4.3 · Amihud×MomR²', '#58a6ff'),
]

for col, (key, label, color) in zip([col1, col2, col3], strategy_config):
    with col:
        d = tb.get(key, {}) if tracker else {}
        ret = d.get('return_pct', 0)
        dd = d.get('max_dd_pct', 0)
        nav = d.get('final_nav', 1.0)
        st.metric(
            label,
            f"{ret:+.1f}%",
            f"回撤 {dd:.1f}% · 净值 {nav:.4f}"
        )

with col4:
    csi_ret = tc.get('return_pct', 0) if tracker else 0
    st.metric("CSI300 基准", f"{csi_ret:+.1f}%", "沪深300")

# 逐日净值可展开
if tracker:
    with st.expander("📈 逐日净值明细"):
        dates = [n['date'] for n in tb['v43']['navs']]
        rows = []
        for i, d in enumerate(dates):
            row = {'日期': d}
            for k in ['v41', 'v42', 'v43']:
                navs = tb[k]['navs']
                row[k.upper()] = f"{navs[i]['nav']:.4f}" if i < len(navs) else '—'
            if 'csi300' in tracker:
                cnavs = tracker['csi300'].get('navs', [])
                row['CSI300'] = f"{cnavs[i]['nav']:.4f}" if i < len(cnavs) else '—'
            rows.append(row)
        import pandas as pd
        df = pd.DataFrame(rows).set_index('日期')
        st.dataframe(df.style.applymap(
            lambda x: 'color:#3fb950' if isinstance(x, str) and x > '1.0000' else 'color:#f85149' if isinstance(x, str) and x < '1.0000' else '',
            subset=[k.upper() for k in ['v41','v42','v43']]
        ), use_container_width=True)

# 当前持仓
if tracker:
    st.subheader(f"📋 持仓 ({start_d} 选入)")
    rb = tracker['rebalances'][0]

    cols = st.columns(3)
    for col, (key, label, _) in zip(cols, strategy_config):
        with col:
            st.markdown(f"**{label}**")
            for s in rb[key]:
                st.caption(stock_display(s))

# ═══════════════════════════════════════════
# 回测区
# ═══════════════════════════════════════════
st.divider()
st.header("📊 历史回测")

# 核心指标
c1, c2, c3 = st.columns(3)
c1.metric("V4.1+SURGE", "16.0% 年化", "-35.2% 回撤 · 0.75 夏普 · 5.2x 净值")
c2.metric("V4.2+SURGE", "15.0% 年化", "-27.8% 回撤 · 0.80 夏普 · 4.8x 净值")
c3.metric("⭐ V4.3+SURGE", "29.4% 年化", "-25.3% 回撤 · 1.33 夏普 · 17.6x 净值", delta_color="off")

# 逐年收益
st.subheader("📈 逐年收益 (均含 SURGE)")
import pandas as pd
yearly = pd.DataFrame({
    "年份": [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026],
    "V4.1": [-1.9,12.5,-6.0,19.4,33.5,35.8,1.5,10.7,-2.7,49.2,23.4],
    "V4.2": [6.9,27.7,-8.7,18.3,42.9,31.4,1.5,3.7,5.4,22.7,12.8],
    "V4.3": [-3.4,18.8,0.8,27.9,101.1,43.8,1.5,0.2,21.1,111.1,57.5],
    "CSI300": [-4.6,20.6,-26.3,38.0,25.5,-6.2,-21.3,-11.7,16.2,21.2,0.2],
}).set_index("年份")

st.dataframe(
    yearly.style.applymap(lambda x: 'color:#3fb950;font-weight:600' if x > 0 else 'color:#f85149',
                          subset=['V4.1','V4.2','V4.3','CSI300'])
    .format("{:+.1f}%"),
    use_container_width=True, height=420
)

# 归因 + 因子
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 演进归因")
    st.markdown("""
| 改动 | 效果 |
|------|------|
| V4.1 基线 | 低波43%是死代码(年化0.4%) |
| → V4.2 反转过滤 | 降7pp回撤, 代价-1pp年化 |
| → V4.3 因子替换 | 低波→Amihud, 动量→MomR² |
| → V4.3 排名融合 | 正交信息源, 1+1>2 |
| → V4.3 SURGE协同 | +2.9pp (vs V4.1的+0.8pp) |
""")

with col2:
    st.subheader("🔬 因子有效性")
    st.dataframe(pd.DataFrame({
        "因子": ["动量×R²", "原始动量", "Amihud", "夏普动量", "低波63日"],
        "年化": ["17.1%","16.1%","13.6%","12.3%","0.4%"],
        "V4.1": ["","✅45%","","","✅43%"],
        "V4.2": ["","✅45%","","","✅43%"],
        "V4.3": ["✅70%","","✅30%","","❌"],
    }).set_index("因子"), use_container_width=True)

# 制度
st.subheader("⏱️ 择时制度 (三代共用)")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(pd.DataFrame({
        "制度": ["RISKON","NEUTRAL","RISKOFF","CRISIS","SURGE RISKON"],
        "占比": ["43%","12%","2%","34%","9%"],
        "V4.3年化": ["+6.5%","+8.9%","+5.2%","-6.5%","+54.2%"],
    }).set_index("制度"), use_container_width=True)
with c2:
    st.markdown("""
**SURGE 关键触发:**
- 2023-07-13 · CRISIS→RISKON · AI行情前夜
- 2024-03-01 · CRISIS→RISKON · 春节后逆转
- 2024-09-24 · CRISIS→RISKON · 924组合拳
""")

st.divider()
st.caption("数据源: QMT完整数据库 · 380指数成分股 · IPO截断 · 双Agent审计 2026-07-23/24")
st.caption("⚠️ 含成分股前视偏差约3-8pp/年 · V4.3可信口径约20-24%")
