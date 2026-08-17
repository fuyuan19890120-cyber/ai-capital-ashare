#!/usr/bin/env python3
"""
ETF-R4 月末调仓通知 → 飞书消息
每月最后一个交易日收盘后运行:
  1. 运行 paper_trade_r4.py 生成信号
  2. 读取信号 JSON
  3. 发送飞书卡片消息
"""
import json, os, sys, subprocess
from datetime import datetime

SIGNAL_FILE = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper.json")
WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

def generate_and_load():
    """运行信号生成器并加载结果"""
    script = os.path.join(os.path.dirname(__file__), "paper_trade_r4.py")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"信号生成失败: {result.stderr}")
        return None
    print(result.stdout)
    with open(SIGNAL_FILE) as f:
        return json.load(f)

def build_card(s):
    r = s["regime"]
    r_label = s["regime_label"]
    mode = s.get("mode", "")

    # 颜色
    if r < 0.15: color = "purple"
    elif r_label == "RISKON": color = "green"
    else: color = "red"

    surge_text = "🚀 SURGE触发!" if s.get("surge") else "未触发"

    # ETF 名称映射
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

    # 持仓
    pos_lines = []
    for p in s["positions"]:
        code = p['code']
        name = ETF_NAMES.get(code, '--')
        pos_lines.append(f"{name}({code}) {p['weight']*100:.0f}%")

    # CRISIS 警告
    crisis_warning = ""
    if r < 0.15:
        crisis_warning = "\n⚠️ **CRISIS熔断: regime<0.15 → 全防御**"
    elif r < 0.50:
        crisis_warning = "\n🛡️ 防守模式"

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"ETF-R4 调仓通知 · {s['signal_date']}"},
                "template": color
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**Regime: {r:.3f} ({r_label})** | SURGE: {surge_text} | 模式: {mode}{crisis_warning}"}
                },
                {"tag": "hr"},
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": "**持仓:**\n" + "\n".join(f"• {l}" for l in pos_lines)}
                },
                {"tag": "hr"},
                {
                    "tag": "div",
                    "text": {"tag": "lark_md", "content": f"**SURGE监测:** 广度 {s['surge_detail']['breadth_count']}/3 (触发条件: ≥2站上SMA50 且 Regime≥0.30)\n**回测(70/30):** 年化35.1% / 夏普1.22 / WFE 0.58 / 回撤-35.4%\n**OOS(2021-2026):** 年化30.5% / 夏普0.81"}
                },
                {
                    "tag": "note",
                    "elements": [{"tag": "plain_text", "content": f"ETF-R4 纸面实盘 · {datetime.now().strftime('%Y-%m-%d %H:%M')} · 仅供参考, 不构成投资建议"}]
                }
            ]
        }
    }
    return card

def send_feishu(card):
    if not WEBHOOK_URL:
        print("\n⚠️ 未设置 FEISHU_WEBHOOK 环境变量, 跳过发送")
        print("--- 消息预览 ---")
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return False

    import requests
    resp = requests.post(WEBHOOK_URL, json=card)
    if resp.status_code == 200:
        print("✅ 飞书发送成功")
        return True
    else:
        print(f"❌ 飞书发送失败: {resp.status_code} {resp.text}")
        return False

if __name__ == "__main__":
    print(f"╔{'═'*58}╗")
    print(f"║  ETF-R4 月末调仓通知{'':>36}║")
    print(f"║  {datetime.now().strftime('%Y-%m-%d %H:%M'):>58}  ║")
    print(f"╚{'═'*58}╝")
    print()

    signal = generate_and_load()
    if signal is None:
        print("❌ 无法生成信号")
        sys.exit(1)

    card = build_card(signal)
    send_feishu(card)
