#!/usr/bin/env python3
"""
ETF-R4 月末自动运行
每天运行, 仅在月末最后一个交易日执行:
  1. 生成 R4 信号 (paper_trade_r4.py)
  2. 更新看板 (dashboard_r4.html)
  3. 发送飞书通知 (notify_feishu_r4.py)

用法: crontab 每天 14:00 运行
"""
import os, sys, json, subprocess, sqlite3
from datetime import datetime, timedelta
import pandas as pd

WORKTREE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_FILE = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper.json")
DASHBOARD = os.path.join(WORKTREE, "dashboard_r4.html")
DASHBOARD_R3_PATH = os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/backtest-macd-bb-kdj/dashboard_r3.html")
DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

def is_last_trading_day():
    """检查今天是否是本月最后一个交易日"""
    con = sqlite3.connect(DB)
    # 获取数据库中最近的交易日
    today = datetime.now().strftime("%Y%m%d")
    dates = pd.read_sql_query(
        f"SELECT DISTINCT date FROM daily WHERE date <= '{today}' ORDER BY date DESC LIMIT 5",
        con)
    con.close()

    if dates.empty:
        return False, None

    dates['date'] = pd.to_datetime(dates['date'])
    latest = dates['date'].iloc[0]  # 最近交易日(今天或最近)
    this_month = latest.month
    this_year = latest.year

    # 检查这个月是否还有其他交易日在 latest 之后
    next_month_start = f"{this_year}{this_month+1:02d}01" if this_month < 12 else f"{this_year+1}0101"
    latest_str = latest.strftime("%Y%m%d")
    con = sqlite3.connect(DB)
    later = pd.read_sql_query(
        f"SELECT COUNT(*) as cnt FROM daily WHERE date > '{latest_str}' AND date < '{next_month_start}'",
        con)
    con.close()

    is_last = later['cnt'].iloc[0] == 0
    return is_last, latest

def update_dashboard(signal_path, dashboard_path):
    """将信号 JSON 嵌入看板 HTML"""
    import re
    with open(signal_path) as f:
        signal = json.load(f)

    with open(dashboard_path) as f:
        html = f.read()

    # 替换 SIGNAL 数据
    signal_json = json.dumps(signal, ensure_ascii=False)
    new_line = f'var S={signal_json};'
    html = re.sub(r'/\* SIGNAL_DATA_PLACEHOLDER \*/', new_line, html)
    html = re.sub(r'var S=\{.*?\};', new_line, html)

    with open(dashboard_path, 'w') as f:
        f.write(html)

    return True

def send_feishu_simple(signal):
    """简化版飞书发送"""
    if not WEBHOOK_URL:
        print("⚠️ 未设置 FEISHU_WEBHOOK")
        return False

    import requests
    r = signal['regime']
    r_label = signal['regime_label']
    color = "purple" if r < 0.15 else ("green" if r_label == "RISKON" else "red")
    surge_text = "🚀 SURGE触发!" if signal.get('surge') else "未触发"

    ETF_NAMES = {
        "510300.SH":"沪深300","510500.SH":"中证500","159915.SZ":"创业板","588000.SH":"科创50",
        "511010.SH":"国债","518880.SH":"黄金","511880.SH":"货币",
        "512880.SH":"证券","512010.SH":"医药","159992.SZ":"创新药","159928.SZ":"消费","512690.SH":"酒",
        "512480.SH":"半导体","159995.SZ":"芯片","512760.SH":"半导体","512660.SH":"军工",
        "515030.SH":"新能车","515790.SH":"光伏","516160.SH":"新能源",
        "515050.SH":"5G","159819.SZ":"人工智能","562500.SH":"机器人",
        "512400.SH":"有色","512200.SH":"房地产","512980.SH":"传媒","159939.SZ":"信息技术",
    }
    pos_lines = []
    for p in signal['positions']:
        code = p['code']
        name = ETF_NAMES.get(code, '--')
        pos_lines.append(f"• {name}({code}) {p['weight']*100:.0f}%")

    content = (
        f"**Regime: {r:.3f} ({r_label})** | SURGE: {surge_text} | 模式: {signal.get('mode','')}\n"
        f"---\n**持仓:**\n" + "\n".join(pos_lines) + "\n---\n"
        f"SURGE广度: {signal['surge_detail']['breadth_count']}/3 (触发条件: ≥2站上SMA50且r≥0.30)\n"
        f"回测(70/30): 年化34.7% / 夏普1.21 / WFE0.58 / 回撤-35.4%\n"
        f"OOS(2021-2026): 年化23.0%"
    )

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"ETF-R4 调仓 · {signal['signal_date']}"},
                "template": color
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": content}},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": f"ETF-R4纸面实盘 · {datetime.now().strftime('%Y-%m-%d')} · 仅供参考"}]}
            ]
        }
    }

    resp = requests.post(WEBHOOK_URL, json=card)
    if resp.status_code == 200:
        print("✅ 飞书发送成功")
        return True
    else:
        print(f"❌ 飞书失败: {resp.status_code}")
        return False

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ETF-R4 月末检查...")

    is_last, latest_date = is_last_trading_day()
    if not is_last:
        print(f"  非月末(最近交易日: {latest_date.date() if latest_date else '?'}), 跳过")
        sys.exit(0)

    # 信号去重保护: 若库里最新日未变(数据冻结/滞后), 不重复生成信号
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE) as f:
            existing = json.load(f)
        if existing.get("signal_date") == latest_date.strftime("%Y-%m-%d"):
            print(f"  信号已生成({latest_date.date()}), 数据未更新, 跳过")
            sys.exit(0)

    print(f"  月末! 最近交易日: {latest_date.date()}")

    # 1. 生成信号
    script = os.path.join(WORKTREE, "paper_trade_r4.py")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ 信号生成失败:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout.strip()[-200:] if len(result.stdout) > 200 else result.stdout.strip())

    # 2. 更新看板 (两个路径)
    if os.path.exists(SIGNAL_FILE):
        update_dashboard(SIGNAL_FILE, DASHBOARD)
        print(f"  ✅ 看板已更新: {DASHBOARD}")
        if os.path.exists(DASHBOARD_R3_PATH):
            update_dashboard(SIGNAL_FILE, DASHBOARD_R3_PATH)
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

    # 3. 发送飞书
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE) as f:
            signal = json.load(f)
        send_feishu_simple(signal)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ✅ ETF-R4 月末任务完成")
