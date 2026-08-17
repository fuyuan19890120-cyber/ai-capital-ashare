#!/usr/bin/env python3
"""重建 R3 看板：当前持仓 + 实时收益 + 历史持仓"""
import json, sqlite3, pandas as pd, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings = __import__('warnings'); warnings.filterwarnings("ignore")

SIG = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper.json")
DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
DASH = os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/backtest-macd-bb-kdj/dashboard_r3.html")

NAMES = {"510300.SH":"沪深300","510500.SH":"中证500","159915.SZ":"创业板","588000.SH":"科创50",
         "511010.SH":"国债ETF","518880.SH":"黄金ETF","511880.SH":"货币ETF",
         "512880.SH":"证券ETF","512800.SH":"银行ETF","512070.SH":"券商保险","512000.SH":"非银ETF",
         "512010.SH":"医药ETF","512170.SH":"医疗ETF","159992.SZ":"创新药","159928.SZ":"消费ETF",
         "512690.SH":"酒ETF","512480.SH":"半导体","159995.SZ":"芯片ETF","512760.SH":"半导体50",
         "512660.SH":"军工ETF","512710.SH":"军工龙头","515030.SH":"新能车","515790.SH":"光伏ETF",
         "516160.SH":"新能源","512400.SH":"有色金属","515210.SH":"钢铁ETF","159870.SZ":"化工ETF",
         "512200.SH":"房地产","516750.SH":"建材ETF","512980.SH":"传媒ETF","159869.SZ":"游戏ETF",
         "159939.SZ":"信息技术","515000.SH":"科技ETF","512720.SH":"计算机","515050.SH":"5GETF",
         "515880.SH":"通信ETF","159967.SZ":"纳指ETF","562500.SH":"机器人","159819.SZ":"人工智能",
         "159611.SZ":"法国CAC","512580.SH":"碳中和","516670.SH":"畜牧ETF","159865.SZ":"中药ETF"}

G="#3fb950"; R="#f85149"; B="#58a6ff"; GD="#d2991d"

# ── 1. 当前持仓 + 实时收益 ──
sig = json.load(open(SIG))
live_rows = ""; avg = 0

if sig.get("positions"):
    con = sqlite3.connect(DB)
    codes = [p["code"] for p in sig["positions"]]
    ph = ",".join(["?"]*len(codes))
    # 用上月末作为持仓起始日
    from datetime import datetime
    today = datetime.now()
    last_month = today.replace(day=1) - __import__('datetime').timedelta(days=1)
    sd = last_month.strftime("%Y%m%d")
    # 如果数据中没有精确月末日，取最近交易日
    r_check = con.execute("SELECT MAX(date) FROM daily WHERE date <= ?", (sd,)).fetchone()
    if r_check and r_check[0]: sd = r_check[0]
    df_s = pd.read_sql_query(f"SELECT code,close FROM daily WHERE code IN ({ph}) AND date=?", con, params=codes+[sd])
    df_n = pd.read_sql_query(f"SELECT code,date,close FROM daily WHERE code IN ({ph}) ORDER BY date DESC", con, params=codes)
    con.close()
    sp = dict(zip(df_s["code"], df_s["close"]))
    np_ = df_n.groupby("code").first()["close"]
    ld = str(df_n["date"].max())[:10]
    for p in sig["positions"]:
        c=p["code"]; ps=float(sp.get(c,0)); pn=float(np_.get(c,ps))
        ret=(pn/ps-1)*100 if ps>0 else 0; nm = NAMES.get(c, c)
        cls="positive" if ret>=0 else "negative"; sgn="+" if ret>0 else ""
        live_rows += f'<tr><td>{c}</td><td>{nm}</td><td>¥{ps:.2f}</td><td>¥{pn:.2f}</td><td class="{cls}">{sgn}{ret:.1f}%</td></tr>'
        avg += ret
    if sig["positions"]: avg /= len(sig["positions"])

lc = G if avg>=0 else R; ls = "+" if avg>0 else ""

# ── 2. 历史持仓 ──
from backtest_etf_r4 import (load_qmt_data, build_panels, compute_regime_2factor,
    make_monthly_select, make_surge_daily_fn, SECTOR_ETFS, BROAD, DEFENSE,
    SURGE_LOCK_DAYS, SURGE_DD_MELT)

df_raw = load_qmt_data()
close, open_, high, low, vol = build_panels(df_raw)
regime = compute_regime_2factor(close, tanh_mult=10)  # 与回测/实盘一致(×10, 非×5)
monthly_sel = make_monthly_select(close, vol, high, low, use_kdj_defense=True)

all_days = close.index
md = {}
for d in all_days: md.setdefault((d.year,d.month),[]).append(d)
month_ends = sorted([v[-1] for v in md.values()])

# 最近12个月的持仓
hist_rows = ""
for me in month_ends[-12:]:
    rv = float(regime.loc[me]) if me in regime.index else 0.5
    if pd.isna(rv): rv = 0.5

    # 获取该月末的持仓
    try:
        picks = monthly_sel(me, rv, {})
    except:
        continue

    if not picks: continue

    # 计算该批次持仓的至今收益
    picks_str = ""
    total_ret = 0; n = 0
    for code, weight in picks[:5]:
        nm = NAMES.get(code, code)
        # 买入价 = 该月末收盘价
        px_buy = float(close.at[me, code]) if me in close.index and code in close.columns and pd.notna(close.at[me,code]) else 0
        # 现价
        ld_px = float(close.iloc[-1][code]) if code in close.columns and pd.notna(close.iloc[-1][code]) else 0
        ret = (ld_px/px_buy-1)*100 if px_buy>0 else 0
        cls = "positive" if ret>=0 else "negative"; sgn = "+" if ret>0 else ""
        picks_str += f'<span class="{cls}">{nm} {sgn}{ret:.0f}%</span> '
        total_ret += ret; n += 1

    avg_r = total_ret/n if n>0 else 0
    acls = "positive" if avg_r>=0 else "negative"; asgn = "+" if avg_r>0 else ""
    hist_rows += f'<tr><td>{me.date()}</td><td>{picks_str}</td><td class="{acls}">{asgn}{avg_r:.1f}%</td></tr>'

# ── 3. 纸面实盘追踪（从6月底起） ──
PAPER_START = "2026-06-30"
paper_nav = 1_000_000
paper_records = []

# 从起始日开始，每月末调仓
paper_dates = [me for me in month_ends if me >= pd.Timestamp(PAPER_START)]
paper_positions = {}  # {code: shares}

for pi, me in enumerate(paper_dates):
    rv = float(regime.loc[me]) if me in regime.index else 0.5
    if pd.isna(rv): rv = 0.5

    # 获取持仓权重
    try:
        picks = monthly_sel(me, rv, {})
    except:
        picks = []

    if picks and pi > 0:
        # 先估值（用上期持仓）
        tv = paper_nav  # cash component, simplified
        for c, s in paper_positions.items():
            if c in close.columns:
                px = float(close.at[me, c]) if pd.notna(close.at[me, c]) else 0
                tv += s * px

        # 记录上期收益
        if paper_records:
            prev_val = paper_records[-1]["value"]
            ret = tv / prev_val - 1
            paper_records[-1]["return_pct"] = round(ret * 100, 1)
            paper_records[-1]["end_value"] = tv
            paper_records[-1]["end_date"] = str(me.date())

        # 调仓：按权重分配
        total = tv * 0.95
        paper_positions = {}
        for code, weight in picks:
            if code in close.columns:
                px = float(close.at[me, code]) if pd.notna(close.at[me, code]) else 0
                if px > 0:
                    shares = int(total * weight / px / 100) * 100
                    paper_positions[code] = shares

    # 记录期初
    if picks:
        tv_now = paper_nav
        for c, s in paper_positions.items():
            if c in close.columns:
                px = float(close.at[me, c]) if pd.notna(close.at[me, c]) else 0
                tv_now += s * px
        paper_records.append({
            "date": str(me.date()),
            "value": paper_nav if pi == 0 else tv_now,
            "positions": [(c, NAMES.get(c, c), round(float(w)*100)) for c, w in picks[:5]],
            "end_value": None, "end_date": None, "return_pct": None,
        })

# 最后一期的收益（至今）
if paper_records and paper_dates:
    last_me = paper_dates[-1]
    tv = 0
    for c, s in paper_positions.items():
        if c in close.columns:
            px = float(close.iloc[-1][c]) if pd.notna(close.iloc[-1][c]) else 0
            tv += s * px
    if paper_records:
        prev = paper_records[-1]["value"]
        ret = tv / prev - 1 if prev > 0 else 0
        paper_records[-1]["return_pct"] = round(ret * 100, 1)
        paper_records[-1]["end_value"] = tv
        paper_records[-1]["end_date"] = str(close.index[-1].date())

# 累计净值
final_val = paper_records[-1]["end_value"] if paper_records and paper_records[-1].get("end_value") else 1_000_000
cum_nav = final_val / 1_000_000
cum_ret = (cum_nav - 1) * 100

# 纸面实盘HTML
paper_html = ""
for rec in paper_records[-6:]:
    ret_pct = rec.get("return_pct")
    if ret_pct is not None:
        rcls = "positive" if ret_pct >= 0 else "negative"
        rsgn = "+" if ret_pct > 0 else ""
        ret_str = f'<td class="{rcls}">{rsgn}{ret_pct:.1f}%</td>'
    else:
        ret_str = '<td style="color:#8b949e">持有中</td>'

    pos_str = " · ".join(f"{nm} {w}%" for _, nm, w in rec["positions"])
    paper_html += f'<tr><td>{rec["date"]}</td><td style="font-size:12px">{pos_str}</td>{ret_str}</tr>'

# ── 3b. 纸面实盘 ──
TRACKER_FILE = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper_tracker.json")
tracker_html = ""
if os.path.exists(TRACKER_FILE):
    trk = json.load(open(TRACKER_FILE))
    tnav = trk.get("current_nav", 1_000_000)
    tret = trk.get("total_return", 0)
    tcls = "ok" if tret >= 0 else "warn"
    tsgn = "+" if tret > 0 else ""
    tracker_html = f'''<div class="card">
  <h2>纸面实盘 (6/30起)</h2>
  <div class="metric-row"><span class="metric-label">当前净值</span><span class="metric-val">{tnav:,.0f}</span></div>
  <div class="metric-row"><span class="metric-label">累计收益</span><span class="metric-val {tcls}">{tsgn}{tret:.1f}%</span></div>
  <div class="metric-row"><span class="metric-label">初始资金</span><span>100万</span></div>
</div>'''

# ── 4. 构建HTML ──
sd = sig.get("signal_date","--")
r_val = round(sig.get("regime",0.5),3)
rl = sig.get("regime_label","NEUTRAL")
surge = sig.get("surge",False)
surg_d = sig.get("surge_detail",{})
bc = "crisis" if r_val<0.15 else ("riskon" if rl=="RISKON" else "defense")

pos_rows = ""
for p in sig.get("positions",[]):
    nm = NAMES.get(p["code"], p.get("label","--"))
    w = p.get("weight",0.25)*100
    pos_rows += f'<tr><td>{p["code"]}</td><td>{nm}</td><td>{w:.0f}%</td></tr>'

surge_badge = f'<span class="badge surge-on">{surg_d.get("breadth_count",0)}/3</span>' if surge else f'<span class="badge surge-off">{surg_d.get("breadth_count",0)}/3</span>'
bd = surg_d.get("breadth",{})
b300 = f'<span style="color:{G}">✅ 站上SMA50</span>' if bd.get("510300.SH") else f'<span style="color:{R}">❌ 低于SMA50</span>'
b500 = f'<span style="color:{G}">✅ 站上SMA50</span>' if bd.get("510500.SH") else f'<span style="color:{R}">❌ 低于SMA50</span>'
b915 = f'<span style="color:{G}">✅ 站上SMA50</span>' if bd.get("159915.SZ") else f'<span style="color:{R}">❌ 低于SMA50</span>'

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF-R4 纸面实盘</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--green:#3fb950;--red:#f85149;--blue:#58a6ff;--gold:#d2991d}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:20px;max-width:1200px;margin:0 auto}}
.header{{text-align:center;padding:20px 0;border-bottom:1px solid var(--border);margin-bottom:24px}}
.header h1{{font-size:24px;color:var(--blue)}}
.header .sub{{color:#8b949e;font-size:13px;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
.card h2{{font-size:14px;color:#8b949e;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}}
.metric-row{{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(48,54,61,0.5)}}
.metric-row:last-child{{border:none}}
.metric-val{{font-size:20px;font-weight:600}}
.metric-label{{font-size:13px;color:#8b949e}}
.badge{{display:inline-block;padding:4px 12px;border-radius:12px;font-size:13px;font-weight:600}}
.riskon{{background:rgba(63,185,80,0.15);color:var(--green)}}
.defense{{background:rgba(248,81,73,0.15);color:var(--red)}}
.crisis{{background:rgba(139,92,246,0.15);color:#8b5cf6}}
.surge-on{{background:rgba(210,153,29,0.15);color:var(--gold)}}
.surge-off{{background:rgba(139,148,158,0.15);color:#8b949e}}
.ok{{color:var(--green)}}.warn{{color:var(--red)}}
.positive{{color:var(--green)}}.negative{{color:var(--red)}}
table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;padding:8px;border-bottom:1px solid var(--border);font-size:13px}}
th{{color:#8b949e}}
.footer{{text-align:center;padding:20px;color:#484f58;font-size:11px}}
</style>
</head>
<body>
<div class="header">
  <h1>ETF-R4 纸面实盘</h1>
  <div class="sub">Regime择时 x MomR2(70%)+Amihud(30%) x SURGE x KDJ防守 x CRISIS熔断 · 43只ETF · 月度调仓</div>
  <div class="sub">回测2016-2026: 年化34.7% / 夏普1.21 / WFE0.58 / 回撤-35.4% · OOS年化23.0%</div>
</div>
<div class="grid">

<div class="card" style="grid-column:span 2">
  <h2>当前信号 ({sd})</h2>
  <div style="margin-bottom:12px">
    <span class="badge {bc}">{rl} {r_val}</span>
    {surge_badge}
  </div>
  <table><tr><th>代码</th><th>名称</th><th>权重</th></tr>{pos_rows}</table>
</div>

<div class="card">
  <h2>策略指标 (70/30)</h2>
  <div class="metric-row"><span class="metric-label">回测年化</span><span class="metric-val ok">+34.7%</span></div>
  <div class="metric-row"><span class="metric-label">夏普比率</span><span class="metric-val" style="color:var(--blue)">1.21</span></div>
  <div class="metric-row"><span class="metric-label">最大回撤</span><span class="metric-val warn">-35.4%</span></div>
  <div class="metric-row"><span class="metric-label">OOS年化</span><span class="metric-val" style="color:var(--blue)">+23.0%</span></div>
  <div class="metric-row"><span class="metric-label">WFE</span><span class="metric-val" style="color:var(--gold)">0.58</span></div>
</div>

''' + tracker_html + f'''
<div class="card">
  <h2>SURGE 监测</h2>
  <div class="metric-row"><span class="metric-label">状态</span>{surge_badge}</div>
  <div class="metric-row"><span class="metric-label">510300</span>{b300}</div>
  <div class="metric-row"><span class="metric-label">510500</span>{b500}</div>
  <div class="metric-row"><span class="metric-label">159915</span>{b915}</div>
</div>

<div class="card">
  <h2>持仓至今收益 ({ld})</h2>
  <div class="metric-row"><span class="metric-label">上月末以来</span><span class="metric-val" style="color:{lc}">{ls}{avg:.1f}%</span></div>
  <table><tr><th>代码</th><th>名称</th><th>信号价</th><th>现价</th><th>收益</th></tr>{live_rows}</table>
</div>

<div class="card" style="grid-column:span 2">
  <h2>历史持仓 (近12个月)</h2>
  <table><tr><th>月末</th><th>持仓 & 至今收益</th><th>平均</th></tr>{hist_rows}</table>
</div>

</div>
<div class="footer">ETF-R4 · 70/30权重 · 双Agent审查通过 · 2026-08-07</div>
</body>
</html>'''

with open(DASH, 'w') as f:
    f.write(html)
print(f'✅ R3看板已重建 (上月末以来 {avg:+.1f}%)')
