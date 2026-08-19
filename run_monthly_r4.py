#!/usr/bin/env python3
"""
ETF-R4 月末自动运行
每天运行, 仅在月末最后一个交易日执行:
  1. 生成 R4 信号 (paper_trade_r4.py)
  2. 更新看板 (dashboard_r4.html)
  3. 发送飞书通知 (notify_feishu_r4.py)

用法: crontab 每天 15:45 运行 (收盘后, tushare 15:00 刷新后, QMT 15:30 更新后)
"""
import os, sys, json, subprocess, sqlite3
from datetime import datetime, timedelta
import pandas as pd

WORKTREE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_FILE = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper.json")  # tushare 主信号(前复权正确)
QMT_SIGNAL_FILE = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper_qmt.json")  # QMT 交叉参考
TRACKER_FILE = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper_tracker.json")  # 纸面实盘净值
DASHBOARD = os.path.join(WORKTREE, "dashboard_r4.html")
DASHBOARD_R3_PATH = os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/backtest-macd-bb-kdj/dashboard_r3.html")
DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
TUSHARE_DB = os.path.expanduser("~/ai-capital-ashare/data/etf_tushare.db")
TUSHARE_TOKEN = "2d7e92d4cda95afe932389b4bee184fd9fabcdf8851ff79d68fe933b"
WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

def _latest_trade_date():
    """两库中较新的最新交易日(<=今天), 用于判断月末(单库数据不全也能判断)"""
    today = datetime.now().strftime("%Y%m%d")
    latest = None
    for db in (DB, TUSHARE_DB):
        try:
            con = sqlite3.connect(db)
            r = con.execute("SELECT MAX(date) FROM daily WHERE date <= ?", (today,)).fetchone()
            con.close()
            if r and r[0] and (latest is None or r[0] > latest):
                latest = r[0]
        except Exception:
            pass
    return latest

def is_last_trading_day():
    """检查今天是否是本月最后一个交易日(用 tushare 交易日历, 精确判断)"""
    latest = _latest_trade_date()
    if latest is None:
        return False, None

    y, m = latest[:4], latest[4:6]
    try:
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        cal = pro.trade_cal(exchange='SSE', start_date=f'{y}{m}01', end_date=f'{y}{m}31')
        open_days = sorted(cal[cal['is_open'] == 1]['cal_date'].astype(str).tolist())
        last_open = open_days[-1] if open_days else None
        is_last = (latest == last_open)
    except Exception:
        # 交易日历不可用: 退化为「latest 是本月 28 日及以后」的粗略近似
        is_last = int(latest[6:8]) >= 28

    return is_last, pd.to_datetime(latest)

def update_dashboard(signal_path, dashboard_path, qmt_signal_path=None, tracker_path=None):
    """将信号 JSON 嵌入看板 HTML (tushare主 + 可选QMT参考 + 可选纸面实盘, 同步更新)"""
    import re
    with open(signal_path) as f:
        signal = json.load(f)

    with open(dashboard_path) as f:
        html = f.read()

    # 替换 tushare 主信号
    signal_json = json.dumps(signal, ensure_ascii=False)
    new_line = f'var S={signal_json};'
    html = re.sub(r'/\* SIGNAL_DATA_PLACEHOLDER \*/', new_line, html)
    html = re.sub(r'var S=\{.*?\};', new_line, html)

    # 替换 QMT 参考信号 (同步更新)
    if qmt_signal_path and os.path.exists(qmt_signal_path):
        with open(qmt_signal_path) as f:
            qmt = json.load(f)
        qmt_json = json.dumps(qmt, ensure_ascii=False)
        qmt_line = f'var S_QMT={qmt_json};'
        html = re.sub(r'/\* QMT_SIGNAL_PLACEHOLDER \*/', qmt_line, html)
        html = re.sub(r'var S_QMT=\{.*?\};', qmt_line, html)

    # 替换纸面实盘净值 (PAPER, 同步更新)
    if tracker_path and os.path.exists(tracker_path):
        with open(tracker_path) as f:
            tracker = json.load(f)
        daily_nav = tracker.get("daily_nav", [])
        rebal_idx = []
        for r in tracker.get("records", []):
            for i, p in enumerate(daily_nav):
                if p["d"] == r["date"]:
                    rebal_idx.append(i)
                    break
        paper = {
            "start": round(tracker.get("initial_capital", 1e6) / 1e4, 2),
            "nav": [p["v"] for p in daily_nav],
            "regime": [p.get("r", 0.5) for p in daily_nav],
            "start_date": daily_nav[0]["d"] if daily_nav else "",
            "end_date": daily_nav[-1]["d"] if daily_nav else "",
            "rebal_dates": [r["date"] for r in tracker.get("records", [])],
            "rebal_idx": rebal_idx,
        }
        paper_json = json.dumps(paper, ensure_ascii=False)
        paper_line = f'var PAPER={paper_json};'
        html = re.sub(r'/\* PAPER_SIGNAL_PLACEHOLDER \*/', paper_line, html)
        html = re.sub(r'var PAPER=\{.*?\};', paper_line, html)

    with open(dashboard_path, 'w') as f:
        f.write(html)

    return True

def send_feishu_compare(qmt_signal, tushare_signal):
    """飞书发送: 双数据源(QMT/tushare)结果对比"""
    if not WEBHOOK_URL:
        print("⚠️ 未设置 FEISHU_WEBHOOK")
        return False

    import requests
    ETF_NAMES = {
        "510300.SH":"沪深300","510500.SH":"中证500","159915.SZ":"创业板","588000.SH":"科创50",
        "511010.SH":"国债","518880.SH":"黄金","511880.SH":"货币",
        "512880.SH":"证券","512800.SH":"银行","512070.SH":"非银","512000.SH":"券商",
        "512010.SH":"医药","512170.SH":"医疗","159992.SZ":"创新药",
        "159928.SZ":"消费","512690.SH":"酒",
        "512480.SH":"半导体","159995.SZ":"芯片","512760.SH":"半导体",
        "512660.SH":"军工","512710.SH":"国防",
        "515030.SH":"新能车","515790.SH":"光伏","516160.SH":"新能源",
        "512400.SH":"有色","515210.SH":"钢铁","159870.SZ":"化工",
        "512200.SH":"房地产","516750.SH":"建材",
        "512980.SH":"传媒","159869.SZ":"游戏",
        "159939.SZ":"信息技术","515000.SH":"科技","512720.SH":"计算机",
        "515050.SH":"5G","515880.SH":"通信",
        "159967.SZ":"创业50","562500.SH":"机器人",
        "159819.SZ":"人工智能","159611.SZ":"电力","512580.SH":"环保","516670.SH":"畜牧","159865.SZ":"农业",
    }

    def block(sig):
        r = sig['regime']; lbl = sig['regime_label']
        surge = "SURGE触发" if sig.get('surge') else "未触发"
        pos = "\n".join(f"• {ETF_NAMES.get(p['code'], p['code'])}({p['code']}) {p['weight']*100:.0f}%" for p in sig['positions'])
        d = sig.get('signal_date', '?')
        return f"Regime: {r:.3f} ({lbl}) | SURGE: {surge} | 模式: {sig.get('mode','')} | 数据截止 {d}\n持仓:\n{pos}"

    # 时点差异提示: 两源日期不同时标注哪个较新
    qd = qmt_signal.get('signal_date', '')
    td = tushare_signal.get('signal_date', '')
    date_tip = None
    if qd and td and qd != td:
        try:
            diff = (datetime.strptime(qd, "%Y-%m-%d") - datetime.strptime(td, "%Y-%m-%d")).days
            newer = "QMT" if diff > 0 else "tushare"
            date_tip = f"⚠️ 时点不同: {newer}较新 {abs(diff)} 天 (分歧多为时点错位, 非数据质量)"
        except Exception:
            pass

    color = "green" if qmt_signal['regime'] >= 0.5 else "red"
    elements = []
    if date_tip:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": date_tip}})
    elements.extend([
        {"tag": "div", "text": {"tag": "lark_md", "content": "**QMT 数据源**"}},
        {"tag": "div", "text": {"tag": "lark_md", "content": block(qmt_signal)}},
        {"tag": "hr"},
        {"tag": "div", "text": {"tag": "lark_md", "content": "**tushare 数据源**"}},
        {"tag": "div", "text": {"tag": "lark_md", "content": block(tushare_signal)}},
        {"tag": "hr"},
        {"tag": "div", "text": {"tag": "lark_md", "content": "回测: QMT 35.1% / tushare 30.9%\nOOS(2021-2026): QMT 30.5% / tushare 34.3%"}},
        {"tag": "div", "text": {"tag": "lark_md", "content": "**📌 明早 9:30 开盘执行（按金额买入）**"}},
        {"tag": "note", "elements": [{"tag": "plain_text", "content": f"ETF-R4纸面实盘(双源对比) · {datetime.now().strftime('%Y-%m-%d')} · 仅供参考"}]}
    ])

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"ETF-R4 调仓 · {qmt_signal['signal_date']}"},
                "template": color
            },
            "elements": elements
        }
    }

    resp = requests.post(WEBHOOK_URL, json=card)
    if resp.status_code == 200:
        print("✅ 飞书发送成功")
        return True
    else:
        print(f"❌ 飞书失败: {resp.status_code}")
        return False

def send_feishu_single(signal, ok_label, fail_label):
    """单源信号通知(另一个源失败时的降级通知)"""
    if not WEBHOOK_URL:
        return False
    import requests
    pos = "\n".join(f"• {p['code']} {p['weight']*100:.0f}%" for p in signal.get('positions', []))
    card = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"ETF-R4 调仓 · {signal.get('signal_date','')}"}, "template": "orange"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"⚠️ {fail_label} 数据源失败, 仅 {ok_label} 信号\n\n持仓:\n{pos}\n\n📌 明早 9:30 开盘执行"}},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": "ETF-R4 单源信号(降级)"}]}
            ]
        }
    }
    resp = requests.post(WEBHOOK_URL, json=card)
    print(f"  单源飞书: {'✅' if resp.status_code == 200 else '❌'}")
    return resp.status_code == 200


if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ETF-R4 月末检查...")

    is_last, latest_date = is_last_trading_day()
    if not is_last:
        print(f"  非月末(最近交易日: {latest_date.date() if latest_date else '?'}), 跳过")
        sys.exit(0)

    # 信号去重: 本月已生成过信号则跳过(按 YYYY-MM 判断, 而非 signal_date 相等)
    this_month = latest_date.strftime("%Y-%m")
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE) as f:
            existing = json.load(f)
        if str(existing.get("signal_date", "")).startswith(this_month):
            print(f"  本月({this_month})信号已生成, 跳过")
            sys.exit(0)

    print(f"  月末! 最近交易日: {latest_date.date()}")

    # 1. 生成信号 (双数据源: QMT + tushare), 单个失败不阻塞另一个
    script = os.path.join(WORKTREE, "paper_trade_r4.py")
    for label, extra, sigfile in [("tushare", [], SIGNAL_FILE), ("QMT", ["--qmt"], QMT_SIGNAL_FILE)]:
        result = subprocess.run([sys.executable, script] + extra, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ❌ {label} 信号生成失败: {result.stderr[-200:]}")
        else:
            print(f"  ✅ {label} 信号 -> {sigfile}")

    # 2. 更新看板 (两个路径, QMT参考信号同步更新)
    if os.path.exists(SIGNAL_FILE):
        update_dashboard(SIGNAL_FILE, DASHBOARD, QMT_SIGNAL_FILE, TRACKER_FILE)
        print(f"  ✅ 看板已更新: {DASHBOARD}")
        if os.path.exists(DASHBOARD_R3_PATH):
            update_dashboard(SIGNAL_FILE, DASHBOARD_R3_PATH, QMT_SIGNAL_FILE, TRACKER_FILE)
            print(f"  ✅ 看板已更新: {DASHBOARD_R3_PATH}")

    # 2.5 更新因子监控
    factor_script = os.path.join(WORKTREE, "update_factor_dashboard.py")
    result = subprocess.run([sys.executable, factor_script], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if '✅' in line:
                print(f"  {line.strip()}")
    else:
        print(f"  ⚠️ 因子更新: {result.stderr[:100]}")

    # 3. 发送飞书 (双源对比, 单源缺失或陈旧则发降级通知)
    # 判断「本月新鲜」而非「文件存在」: 旧文件(上月日期)残留时不会被当现役源
    this_month = latest_date.strftime("%Y-%m")
    qmt_signal = tushare_signal = None
    if os.path.exists(SIGNAL_FILE):  # tushare 主信号
        with open(SIGNAL_FILE) as f:
            _sig = json.load(f)
        if str(_sig.get("signal_date", "")).startswith(this_month):
            tushare_signal = _sig
    if os.path.exists(QMT_SIGNAL_FILE):  # QMT 交叉参考
        with open(QMT_SIGNAL_FILE) as f:
            _sig = json.load(f)
        if str(_sig.get("signal_date", "")).startswith(this_month):
            qmt_signal = _sig

    if qmt_signal and tushare_signal:
        send_feishu_compare(qmt_signal, tushare_signal)
    elif qmt_signal:
        send_feishu_single(qmt_signal, "QMT", "tushare")
    elif tushare_signal:
        send_feishu_single(tushare_signal, "tushare", "QMT")
    else:
        print("  ❌ 两个数据源信号都生成失败, 无法发送飞书")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ✅ ETF-R4 月末任务完成")
