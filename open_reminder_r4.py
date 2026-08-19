#!/usr/bin/env python3
"""
开盘前飞书提醒: 若「信号日的下一个交易日」是今天, 提醒执行 ETF-R4 调仓。
并列展示 QMT 和 tushare 两个数据源的持仓, 供参考。

逻辑: 月末最后交易日收盘后生成信号 → 次日开盘执行。
      本脚本工作日 9:15 运行, 检查今天是否是信号日的下一交易日, 是则飞书提醒。

crontab: 15 9 * * 1-5 (工作日 9:15, 开盘前)
"""
import os, sys, json
from datetime import datetime, timedelta

SIGNAL = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper_tushare.json")
QMT_SIGNAL = os.path.expanduser("~/ai-capital-ashare/signals/etf_r4_paper.json")
WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

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


def load_signal(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if not WEBHOOK_URL:
        print("⚠️ 未设置 FEISHU_WEBHOOK, 跳过")
        return

    qmt_sig = load_signal(QMT_SIGNAL)
    tushare_sig = load_signal(SIGNAL)
    if qmt_sig is None and tushare_sig is None:
        return

    # 用较新的 signal_date 判断执行日
    dates = [s.get("signal_date", "") for s in (qmt_sig, tushare_sig) if s]
    signal_date = max(dates) if dates else ""
    if not signal_date:
        return

    today = datetime.now().strftime("%Y%m%d")

    # 执行日 = 信号日的下一个工作日 (跳过周末; 月末最后交易日极少与长假重叠)
    sig_d = datetime.strptime(signal_date, "%Y-%m-%d")
    exec_d = sig_d + timedelta(days=1)
    while exec_d.weekday() >= 5:  # 周六=5 / 周日=6
        exec_d += timedelta(days=1)
    exec_date = exec_d.strftime("%Y%m%d")

    # 今天不是执行日 → 不提醒
    if today != exec_date and "--force" not in sys.argv:
        print(f"[{datetime.now().strftime('%H:%M')}] 非执行日(信号{signal_date}, 执行日{exec_date}), 跳过")
        return

    import requests

    def block(sig, label):
        if sig is None:
            return f"**{label}**\n(无信号)"
        pos = "\n".join(f"• {ETF_NAMES.get(p['code'], p['code'])}({p['code']}) {p['weight']*100:.0f}%"
                        for p in sig.get("positions", []))
        d = sig.get("signal_date", "?")
        return f"**{label}**（数据截止 {d}）\n{pos}"

    content = (f"**📌 今日开盘执行调仓**（信号日 {signal_date}）\n\n"
               f"{block(qmt_sig, 'QMT')}\n\n{block(tushare_sig, 'tushare')}\n\n"
               f"按金额买入，开盘 9:30 执行（双源并列供参考）")

    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": "ETF-R4 调仓执行提醒"},
                "template": "orange"
            },
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": content}},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": "ETF-R4 · 手动下单 · 双源并列 · 仅供参考"}]}
            ]
        }
    }
    resp = requests.post(WEBHOOK_URL, json=card)
    print(f"[{datetime.now().strftime('%H:%M')}] 开盘提醒: {'✅ 已发送' if resp.status_code == 200 else f'❌ {resp.status_code}'}")


if __name__ == "__main__":
    main()
