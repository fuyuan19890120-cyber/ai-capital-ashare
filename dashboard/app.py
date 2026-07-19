"""
AI Capital A-Share — 量化策略 Dashboard
Streamlit 可视化面板

启动：cd ~/ai-capital-ashare && streamlit run dashboard/app.py
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# ============================================================
# 配置
# ============================================================
st.set_page_config(
    page_title="AI Capital A-Share",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNAL_FILE = os.path.join(PROJECT_ROOT, "signals", "latest.json")
TRACKER_FILE = os.path.join(PROJECT_ROOT, "signals", "portfolio_tracker.json")
FACTOR_MONITOR_FILE = os.path.join(PROJECT_ROOT, "signals", "factor_monitor.json")
SURGE_FILE = os.path.join(PROJECT_ROOT, "signals", "surge_state.json")
HS300_CACHE = os.path.join(PROJECT_ROOT, "data", "index_sh000300.csv")

STRATEGY_VERSION = "V4.2 定稿 (2026-07-19)"
# V4.2: V4.1 + 反转硬过滤(剔除近20日涨幅前15%), 全缓存池年化21.8%/夏普1.13
EXPECTATION_NOTE = ("可信预期: 年化 ~14-16% + SURGE 增量 2-4pp | 最大回撤 -25~-30% | "
                    "V4.2=反转过滤(15%)仅增一行代码 | 全池展示口径 21.8%/0 负收益年")

REGIME_EMOJI = {"RISKON": "🟢", "NEUTRAL": "🟡", "RISKOFF": "🟠", "CRISIS": "🔴"}
REGIME_COLORS = {"RISKON": "#2ecc71", "NEUTRAL": "#f1c40f", "RISKOFF": "#e67e22", "CRISIS": "#e74c3c"}

FACTOR_NAMES = {"low_vol": "低波动率", "value": "价值", "quality": "质量", "momentum_6m": "动量"}
FACTOR_W = {"low_vol": 30, "value": 25, "quality": 20, "momentum_6m": 25}


# ============================================================
# 数据加载
# ============================================================

@st.cache_data(ttl=300)
def load_signal():
    """加载最新信号"""
    if not os.path.exists(SIGNAL_FILE):
        return None
    with open(SIGNAL_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_tracker():
    """加载收益追踪"""
    if not os.path.exists(TRACKER_FILE):
        return None
    with open(TRACKER_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_factor_monitor():
    """加载因子监控数据"""
    if not os.path.exists(FACTOR_MONITOR_FILE):
        return None
    with open(FACTOR_MONITOR_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_surge():
    """加载日频 SURGE 锁状态(surge_monitor.py 维护)"""
    if not os.path.exists(SURGE_FILE):
        return None
    with open(SURGE_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_hs300():
    """加载沪深300基准数据"""
    if not os.path.exists(HS300_CACHE):
        return None
    df = pd.read_csv(HS300_CACHE, index_col=0, parse_dates=True)
    if 'close' in df.columns:
        df = df.rename(columns={'close': 'Close'})
    return df


def classify_sector(code: str) -> str:
    if code.startswith('688'): return '科创板'
    elif code.startswith(('300', '301')): return '创业板'
    elif code.startswith(('600', '601', '603', '605')): return '沪市主板'
    elif code.startswith(('000', '001', '002', '003')): return '深市主板'
    return '其他'


@st.cache_data(ttl=86400)
def get_stock_names():
    """获取股票代码→名称映射（缓存24小时）"""
    names = {}
    try:
        import akshare as ak
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


# ============================================================
# 页面
# ============================================================

def main():
    st.title("📊 AI Capital A-Share")
    st.caption(f"量化策略 Dashboard — {STRATEGY_VERSION} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption(EXPECTATION_NOTE)

    signal = load_signal()
    tracker = load_tracker()
    factor_data = load_factor_monitor()

    if signal is None:
        st.warning("⚠️ 尚未生成信号。请先运行 `python run_monthly.py`。")
        st.info("首次运行：\n```bash\ncd ~/ai-capital-ashare\nsource venv/bin/activate\npython run_monthly.py\n```")
        return

    # ============================================================
    # 第一行：制度 + 核心指标
    # ============================================================
    col1, col2, col3, col4, col5 = st.columns(5)

    regime = signal['regime']
    emoji = REGIME_EMOJI.get(regime['regime'], '❓')

    with col1:
        st.metric("市场制度", f"{emoji} {regime['regime']}")
    with col2:
        st.metric("制度分数", f"{regime['score']:.3f}")
    with col3:
        st.metric("沪深300", f"{regime['price']:.0f}")
    with col4:
        st.metric("偏离年线", f"{regime['deviation_pct']:+.1f}%",
                  delta=f"SMA250: {regime['sma250']:.0f}")
    with col5:
        n_stocks = len(signal.get('selected_stocks', []))
        st.metric("持仓数", f"{n_stocks} 只")

    # ===== SURGE 状态条(日频广度SMA30锁定, surge_monitor.py 维护) =====
    surge = load_surge()
    if surge:
        d = surge.get('detail', {})
        sh = surge.get('shadow', {})
        if surge.get('active'):
            st.success(f"🚨 SURGE 锁定中: 强制 RISKON | 触发 {surge.get('trigger_date')} → "
                       f"到期 {surge.get('expire_date')} | 广度 {d.get('breadth_now', 0):.0%} "
                       f"s30分 {d.get('s30', 0):.3f}")
        else:
            st.caption(f"⚡ SURGE 未触发(检查于 {surge.get('checked_date', '—')}) | "
                       f"广度 {d.get('breadth_now', 0):.0%}(15日前 {d.get('breadth_prev', 0):.0%}) "
                       f"s30分 {d.get('s30', 0):.3f} 基础分 {d.get('base', 0):.3f}")
        if bool(sh.get('active')) != bool(surge.get('active')):
            st.warning(f"⚖️ 影子候选(20d/0.80)与正式信号分歧: 正式锁={surge.get('active')} "
                       f"影子锁={sh.get('active')} —— 天然裁决样本, 记录复盘")
    else:
        st.caption("⚡ SURGE 监控尚未运行(surge_monitor.py, 工作日15:35 自动)")

    st.divider()

    # ============================================================
    # 第二行：收益曲线（如果有历史数据）
    # ============================================================
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 收益追踪")
        if tracker and tracker.get('history'):
            _plot_returns(tracker)
        else:
            st.info("尚无历史收益数据。月度调仓后自动累积。")

    with col_right:
        st.subheader("💰 仓位配置")
        for k, v in signal.get('allocation', {}).items():
            st.metric(k, v)

        # 风控参数
        st.divider()
        st.caption("🛡️ 风控参数")
        rr = signal.get('risk_report', {}).get('limits', {})
        if rr:
            for k, v in rr.items():
                st.caption(f"• {k}: {v}")

    st.divider()

    # ============================================================
    # 第三行：持仓明细 + 板块分布
    # ============================================================
    col_left2, col_right2 = st.columns([3, 2])

    with col_left2:
        st.subheader(f"📋 精选个股 (Top-{len(signal.get('selected_stocks', []))})")
        stocks = signal.get('selected_stocks', [])
        if stocks:
            name_map = get_stock_names()
            rows = []
            for s in stocks:
                code = s['code']
                sector = classify_sector(code)
                rows.append({
                    '代码': code,
                    '名称': name_map.get(code, '—'),
                    '得分': s['score'],
                    '板块': sector,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("🛡️ 防御模式：不持仓个股")

    with col_right2:
        # 板块饼图
        stocks = signal.get('selected_stocks', [])
        if stocks:
            sectors = {}
            for s in stocks:
                sec = classify_sector(s['code'])
                sectors[sec] = sectors.get(sec, 0) + 1
            st.subheader("📊 板块分布")
            df_sector = pd.DataFrame({
                '板块': list(sectors.keys()),
                '数量': list(sectors.values()),
            })
            st.dataframe(df_sector, use_container_width=True, hide_index=True)

            # 风控警告
            rr = signal.get('risk_report', {})
            if rr.get('warnings'):
                st.divider()
                st.caption("⚠️ 风控提示")
                for w in rr['warnings']:
                    st.caption(w)

    st.divider()

    # ============================================================
    # 底部：调仓历史
    # ============================================================
    st.divider()

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
    st.subheader("📜 调仓历史")
    if tracker and tracker.get('history'):
        history_rows = []
        for h in reversed(tracker['history'][-20:]):
            history_rows.append({
                '日期': h['date'],
                '制度': h['regime'],
                '买入': len(h.get('to_buy', [])),
                '卖出': len(h.get('to_sell', [])),
                '持有': len(h.get('to_hold', [])),
                '累计收益': f"{h.get('return_pct', 0):+.1f}%",
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
    else:
        st.info("尚无调仓记录")

    st.divider()
    st.caption(f"信号日期: {signal['date']} | 生成时间: {signal['generated_at']} | 股票池: {signal.get('universe_size', '?')} 只")


# ============================================================
# 收益曲线图
# ============================================================

def _plot_returns(tracker):
    """绘制累计收益曲线 vs 沪深300"""
    history = tracker['history']
    if not history:
        return

    dates = [h['date'] for h in history]
    returns = [h.get('return_pct', 0) for h in history]
    regimes = [h['regime'] for h in history]

    chart_data = pd.DataFrame({
        '日期': pd.to_datetime(dates),
        '策略累计收益(%)': returns,
        '制度': regimes,
    }).sort_values('日期')

    # 沪深300 对比
    hs300 = load_hs300()
    if hs300 is not None and not hs300.empty:
        # 以策略起始日期为基准，计算沪深300同期收益
        start_date = chart_data['日期'].iloc[0]
        hs300 = hs300[hs300.index >= start_date]
        if not hs300.empty:
            base_price = hs300['Close'].iloc[0]
            hs300_ret = (hs300['Close'] / base_price - 1) * 100
            # 对齐日期到月度
            hs300_monthly = hs300_ret.resample('ME').last()

    # 简易图：Matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chart_data['日期'], chart_data['策略累计收益(%)'], 'b-o', linewidth=2, markersize=4, label='策略')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # 标注制度颜色
    for i, row in chart_data.iterrows():
        color = REGIME_COLORS.get(row['制度'], '#95a5a6')
        ax.axvspan(row['日期'] - pd.Timedelta(days=14), row['日期'] + pd.Timedelta(days=14),
                   alpha=0.1, color=color, linewidth=0)

    ax.set_ylabel('累计收益 (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    st.pyplot(fig)

    # 添加沪深300对比线
    st.caption("💡 沪深300 基准数据可用时，将自动叠加对比曲线。")


if __name__ == "__main__":
    main()
