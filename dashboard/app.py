"""
策略一 三代对比看板
启动: cd ~/ai-capital-ashare && venv/bin/streamlit run dashboard/app.py
每日更新: python3 ~/quant/daily_tracker.py
"""
import streamlit as st
import json, os
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="策略一 · V4 三代对比", page_icon="📊", layout="wide")

TRACKER_FILE = os.path.join(os.path.dirname(__file__), "tracker.json")

# === 股票名称 ===
@st.cache_data(ttl=86400)
def get_names():
    names = {}
    try:
        import akshare as ak, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for idx in ["000300", "399006", "000688"]:
            try:
                df = ak.index_stock_cons(symbol=idx)
                for _, r in df.iterrows():
                    names[str(r['品种代码']).zfill(6)] = r['品种名称']
            except: pass
    except: pass
    return names
NM = get_names()
def nd(code):
    n = code.split('.')[0]
    name = NM.get(n, '')
    return f"{code}  {name}" if name else code

# === 数据 ===
tracker = json.load(open(TRACKER_FILE)) if os.path.exists(TRACKER_FILE) else None
tb = tracker['daily'] if tracker else {}
tc = tracker.get('csi300', {}) if tracker else {}

st.markdown('<div style="font-size:24px;font-weight:700;margin-bottom:2px">📊 策略一 · V4.1 → V4.2 → V4.3</div>', unsafe_allow_html=True)
st.caption("QMT 数据库 · 380只指数成分股 · 均含 SURGE · 每月末调仓")

# ═════════════════════════════════════════
# 持仓 + 收益 (三列一一对应)
# ═════════════════════════════════════════
st.divider()
st.subheader("📋 当前持仓 & 实盘收益")

if tracker:
    rb = tracker['rebalances'][0]
    start_d = tracker.get('start', '2026-06-30')
    updated = tracker.get('updated', '—')
    st.caption(f"选股日 {start_d}  ·  初始 50万  ·  更新至 {updated}")

    cols = st.columns(3)
    config = [
        ('v41', '⬜ V4.1 · 低波+动量', '#8b949e'),
        ('v42', '🟡 V4.2 · +反转过滤', '#d29922'),
        ('v43', '🔵 V4.3 · Amihud×MomR²', '#58a6ff'),
    ]

    for col, (key, label, accent) in zip(cols, config):
        with col:
            st.markdown(f"**{label}**")

            d = tb.get(key, {})
            pct = d.get('return_pct', 0)
            profit = d.get('profit', 0)
            val = d.get('current_value', 0)
            dd = d.get('max_dd_pct', 0)

            c = "#3fb950" if pct >= 0 else "#f85149"
            st.markdown(f"""
            <div style="background:#161b22;border-left:3px solid{accent};border-radius:6px;padding:12px;margin:6px 0">
              <div style="font-size:24px;font-weight:700;color:{c}">{pct:+.1f}%</div>
              <div style="font-size:12px;color:#8b949e;margin-top:4px;line-height:1.6">
                盈亏 <b style="color:{c}">{profit:+,.0f}元</b><br>
                市值 <b>{val:,.0f}元</b> · 回撤 <b style="color:#f85149">{dd:.1f}%</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(f"持仓 {len(rb[key])} 只：")
            for i, s in enumerate(rb[key], 1):
                st.caption(f"{i:2d}. {nd(s)}")
else:
    st.warning("运行 `python3 ~/quant/daily_tracker.py` 生成数据 · 首次需等待选股计算")

# 逐日净值
if tracker:
    with st.expander("📈 逐日净值明细"):
        dates = [n['date'] for n in tb['v43']['navs']]
        rows = []
        for i, d in enumerate(dates):
            row = {'日期': d}
            for k in ['v41','v42','v43']:
                navs = tb[k]['navs']
                row[k.upper()] = f"{navs[i]['nav']:.4f}" if i < len(navs) else '—'
            if 'csi300' in tracker:
                cnavs = tracker['csi300'].get('navs', [])
                row['CSI300'] = f"{cnavs[i]['nav']:.4f}" if i < len(cnavs) else '—'
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index('日期'), use_container_width=True)

# ═════════════════════════════════════════
# 历史回测
# ═════════════════════════════════════════
st.divider()
st.header("📊 历史回测 (2015-2026)")

c1, c2, c3 = st.columns(3)
c1.metric("V4.1+SURGE", "16.0% 年化, 5.2x", "-35.2% 回撤 · 0.75 夏普")
c2.metric("V4.2+SURGE", "15.0% 年化, 4.8x", "-27.8% 回撤 · 0.80 夏普")
c3.metric("⭐ V4.3+SURGE", "29.4% 年化, 17.6x", "-25.3% 回撤 · 1.33 夏普")

st.subheader("📈 逐年收益")
yearly = pd.DataFrame({
    "年份": [2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026],
    "V4.1": [-1.9,12.5,-6.0,19.4,33.5,35.8,1.5,10.7,-2.7,49.2,23.4],
    "V4.2": [6.9,27.7,-8.7,18.3,42.9,31.4,1.5,3.7,5.4,22.7,12.8],
    "V4.3": [-3.4,18.8,0.8,27.9,101.1,43.8,1.5,0.2,21.1,111.1,57.5],
    "CSI300": [-4.6,20.6,-26.3,38.0,25.5,-6.2,-21.3,-11.7,16.2,21.2,0.2],
}).set_index("年份")
st.dataframe(yearly.style.applymap(
    lambda x: 'color:#3fb950' if x>0 else 'color:#f85149',
    subset=['V4.1','V4.2','V4.3','CSI300']
).format("{:+.1f}%"), use_container_width=True)

ca, cb = st.columns(2)
with ca:
    st.subheader("📊 演进归因")
    st.markdown("""
| 改动 | 效果 |
|------|------|
| V4.1 基线 | 低波43%是死代码(0.4%年化) |
| → V4.2 反转过滤 | 降7pp回撤, 代价-1pp年化 |
| → V4.3 因子替换 | Amihud+MomR²信息源正交 |
| → V4.3 排名融合 | 1+1>2协同放大 |
| SURGE增益 | V4.3 +2.9pp / V4.1 +0.8pp |
""")
with cb:
    st.subheader("🔬 因子有效性")
    st.dataframe(pd.DataFrame({
        "因子": ["动量×R²","原始动量","Amihud","低波63日"],
        "年化": ["17.1%","16.1%","13.6%","0.4%"],
        "V4.1": ["","✅","","✅"],
        "V4.2": ["","✅","","✅"],
        "V4.3": ["✅70%","","✅30%","❌"],
    }).set_index("因子"), use_container_width=True)

st.subheader("⏱️ 择时制度")
cc, cd = st.columns(2)
with cc:
    st.dataframe(pd.DataFrame({
        "": ["RISKON","NEUTRAL","RISKOFF","CRISIS","SURGE RISKON"],
        "占比": ["43%","12%","2%","34%","9%"],
        "年化": ["+6.5%","+8.9%","+5.2%","-6.5%","+54.2%"],
    }).set_index(""), use_container_width=True)
with cd:
    st.markdown("""
**SURGE 关键触发:**
- 2023-07 · AI行情前夜 · CRISIS→RISKON
- 2024-03 · 春节后逆转 · CRISIS→RISKON
- 2024-09 · 924组合拳 · CRISIS→RISKON
""")

st.divider()
st.caption("数据: QMT完整数据库 · IPO截断 · 双Agent审计 2026-07-24 · 含成分股前视偏差~3-8pp/年")
