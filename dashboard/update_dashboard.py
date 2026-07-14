#!/usr/bin/env python3
"""Update dashboard with factor monitor panel"""
import re

with open('dashboard/app.py', 'r') as f:
    content = f.read()

# 1. Add FACTOR_MONITOR_FILE path
old = 'TRACKER_FILE = os.path.join(PROJECT_ROOT, "signals", "portfolio_tracker.json")'
new = '''TRACKER_FILE = os.path.join(PROJECT_ROOT, "signals", "portfolio_tracker.json")
FACTOR_MONITOR_FILE = os.path.join(PROJECT_ROOT, "signals", "factor_monitor.json")'''
content = content.replace(old, new)

# 2. Add load_factor_monitor function
old = '@st.cache_data(ttl=3600)\ndef load_hs300():'
new = '''@st.cache_data(ttl=300)
def load_factor_monitor():
    """加载因子监控数据"""
    if not os.path.exists(FACTOR_MONITOR_FILE):
        return None
    with open(FACTOR_MONITOR_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_hs300():'''
content = content.replace(old, new)

# 3. Add factor display names
old = 'REGIME_COLORS = {"RISKON": "#2ecc71", "NEUTRAL": "#f1c40f", "RISKOFF": "#e67e22", "CRISIS": "#e74c3c"}'
new = '''REGIME_COLORS = {"RISKON": "#2ecc71", "NEUTRAL": "#f1c40f", "RISKOFF": "#e67e22", "CRISIS": "#e74c3c"}

FACTOR_NAMES = {"low_vol": "低波动率", "value": "价值", "quality": "质量", "momentum_6m": "动量"}
FACTOR_W = {"low_vol": 30, "value": 25, "quality": 20, "momentum_6m": 25}'''
content = content.replace(old, new)

# 4. Load factor_data in main
old = "    signal = load_signal()\n    tracker = load_tracker()"
new = "    signal = load_signal()\n    tracker = load_tracker()\n    factor_data = load_factor_monitor()"
content = content.replace(old, new)

# 5. Add factor section before trade history
old = '    st.subheader("📜 调仓历史")'
new = '''    st.divider()

    # ============================================================
    # 因子监控 (Quant-Zero)
    # ============================================================
    st.subheader("🔬 因子监控 (Quant-Zero)")

    if factor_data and factor_data.get("records"):
        latest = factor_data["records"][-1]["factors"]
        cols = st.columns(4)
        for i, fn in enumerate(["low_vol", "value", "quality", "momentum_6m"]):
            m = latest.get(fn, {})
            ic = m.get("ic", 0)
            wr = m.get("win_rate", 50)
            status = "✅" if ic > 0 else "⚠️"
            with cols[i]:
                st.metric(
                    f"{status} {FACTOR_NAMES[fn]}",
                    f"IC {ic:+.3f}",
                    delta=f"权重 {FACTOR_W[fn]}% | 胜率 {wr:.0f}%",
                )

        alerts = factor_data["records"][-1].get("alerts", [])
        if alerts:
            st.warning(f"⚠️ 退化预警: {', '.join(alerts)}")
        else:
            st.success("✅ 所有因子正常")

        records = factor_data["records"]
        if len(records) >= 2:
            st.caption("📈 滚动 IC 历史")
            rows = []
            for r in records[-6:]:
                row = {"日期": r["date"]}
                for fn in FACTOR_NAMES:
                    row[FACTOR_NAMES[fn]] = f"{r['factors'].get(fn, {}).get('ic', 0):+.3f}"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("🔬 因子监控数据将在月度运行后自动生成")

    st.divider()

    # ============================================================
    # 底部：调仓历史
    # ============================================================
    st.subheader("📜 调仓历史")'''
content = content.replace(old, new)

with open('dashboard/app.py', 'w') as f:
    f.write(content)
print("Dashboard updated with factor monitor panel")
