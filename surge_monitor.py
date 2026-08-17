#!/usr/bin/env python3
"""
日频 SURGE 监控(2026-07-19 上线)

背景: 广度SMA30锁定进场经回测验证(+4.2pp, 复审+扰动敏感性通过), 但实盘管线
此前只在月末检查一次、无锁状态 —— 17 次历史触发只能碰上 2 次。本脚本补上缺口:

每个交易日收盘后运行:
  1. 拉取沪深300指数 + 3只权益ETF(510300/510500/159915)最新日线
  2. 按实盘口径(src/signal_generator._check_breadth_lock 同款公式)计算触发
  3. 锁状态持久化到 signals/surge_state.json:
       {active, trigger_date, expire_date(触发日起第21个交易日), checked_date, detail}
  4. 触发/到期/状态变化时打印显著提示(供 cron 邮件/Actions 日志)

月末调仓管线(run_monthly / signal_generator)读取该文件: active=True 时强制 RISKON。
触发日次日应手动/自动执行进场(与回测口径一致: T收盘信号, T+1开盘成交)。

用法: venv/bin/python surge_monitor.py           # 收盘后跑(建议 15:35 后)
      venv/bin/python surge_monitor.py --dry     # 只打印不写状态
部署: crontab -e 添加
      35 15 * * 1-5 cd ~/ai-capital-ashare && venv/bin/python surge_monitor.py >> signals/surge_monitor.log 2>&1
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(BASE, "signals", "surge_state.json")

# 预注册参数(与回测/实盘口径一致, 勿因单次回测调整)
LOOKBACK = 15        # 广度回看交易日
LOCK_DAYS = 14       # 锁定交易日(与回测 SURGE_LOCK_DAYS 对齐; 21天会跨月漏掉下次SURGE, 回测-2.9pp)
S30_MIN = 0.70       # SMA30 制度分阈值
BASE_MIN = 0.15      # SMA250 制度分底线
EQ_ETFS = {"sh510300": "510300", "sh510500": "510500", "sz159915": "159915"}

# 影子候选(2026-07-19): 20d回看+0.80严确认。扰动敏感性+跨池迁移验证通过,
# 但触发日期与默认版高度重叠(指数级信号), 需等实盘出现"默认触发而影子未触发"
# 的分歧样本才能转正。只记录, 不影响正式信号。
SHADOW = {"lookback": 20, "s30_min": 0.80}

# 行业 ETF 池(36只, 与回测 SECTOR_ETFS 一致), SURGE 触发时用于选 Top-2
SECTOR_ETFS = {
    "512880.SH": "证券ETF", "512800.SH": "银行ETF", "512070.SH": "券商保险",
    "512000.SH": "非银ETF", "512010.SH": "医药ETF", "512170.SH": "医疗ETF",
    "159992.SZ": "创新药", "159928.SZ": "消费ETF", "512690.SH": "酒ETF",
    "512480.SH": "半导体", "159995.SZ": "芯片ETF", "512760.SH": "半导体50",
    "512660.SH": "军工ETF", "512710.SH": "军工龙头", "515030.SH": "新能车",
    "515790.SH": "光伏ETF", "516160.SH": "新能源", "512400.SH": "有色金属",
    "515210.SH": "钢铁ETF", "159870.SZ": "化工ETF", "512200.SH": "房地产",
    "516750.SH": "建材ETF", "512980.SH": "传媒ETF", "159869.SZ": "游戏ETF",
    "159939.SZ": "信息技术", "515000.SH": "科技ETF", "512720.SH": "计算机",
    "515050.SH": "5GETF", "515880.SH": "通信ETF", "159967.SZ": "纳指ETF",
    "562500.SH": "机器人", "159819.SZ": "人工智能", "159611.SZ": "法国CAC",
    "512580.SH": "碳中和", "516670.SH": "畜牧ETF", "159865.SZ": "中药ETF",
}
MOMR2_W = 0.70  # SURGE 选股权重(与 R4 定稿一致)
AMIHUD_W = 0.30


def fetch_daily():
    """指数与ETF日线(近1.5年足够算 SMA250)。ETF: 东财qfq优先(Actions可用), 本机被风控时回退新浪(不复权,
    ETF分红日SMA有~1%级失真, 可接受)。"""
    import akshare as ak
    idx = ak.stock_zh_index_daily(symbol="sh000300")
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"].sort_index()
    etfs = {}
    for sym, code in EQ_ETFS.items():
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                     start_date="20240101", end_date="20500101", adjust="qfq")
            etfs[sym] = df.set_index(pd.to_datetime(df["日期"]))["收盘"].sort_index()
        except Exception:  # noqa: BLE001 - 东财被风控, 回退新浪
            df = ak.fund_etf_hist_sina(symbol=sym)
            df["date"] = pd.to_datetime(df["date"])
            etfs[sym] = df.set_index("date")["close"].sort_index()
    return idx, etfs


def fetch_sector_data():
    """拉行业 ETF 收盘价+成交量(仅 SURGE 触发时调用, 避免平时浪费请求)。"""
    import akshare as ak
    closes, vols = {}, {}
    for code in SECTOR_ETFS:
        sym = ("sh" + code[:6]) if code.endswith(".SH") else ("sz" + code[:6])
        try:
            df = ak.fund_etf_hist_em(symbol=code[:6], period="daily",
                                     start_date="20240101", end_date="20500101", adjust="qfq")
            df = df.set_index(pd.to_datetime(df["日期"])).sort_index()
            closes[code] = df["收盘"]
            vols[code] = df.get("成交量", pd.Series(1.0, index=df.index))
        except Exception:  # noqa: BLE001 - 单只失败跳过, 不影响整体
            continue
    return closes, vols


def select_top2(closes, vols):
    """SURGE 选股: 全量行业池 × MomR²(70%)+Amihud(30%) Z-score → Top-2。
    与回测 make_surge_daily_fn 口径一致(adaptive=False, w=30)。"""
    scores = {}
    for code in closes:
        c = closes[code].dropna()
        if len(c) < 60:
            continue
        # MomR²(30日窗口, log线性回归)
        recent = c.iloc[-31:]
        y = np.log(recent.values.astype(float))
        x = np.arange(len(y), dtype=float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        ann = np.exp(slope * 250) - 1
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        momr2 = ann * r2
        # Amihud(20日)
        v = vols.get(code, pd.Series(1.0, index=c.index)).reindex(c.index)
        c20 = c.iloc[-21:].values.astype(float)
        v20 = v.iloc[-21:].values.astype(float)
        rets = np.abs(np.diff(c20) / c20[:-1])
        denom = v20[1:] * c20[1:]
        amihud = np.mean(rets / (denom + 1e-10)) * 1e8 if len(denom) > 0 else 0
        scores[code] = (momr2, amihud)

    if len(scores) < 2:
        return []

    # Z-score 复合
    codes = list(scores.keys())
    m_arr = np.array([scores[c][0] for c in codes])
    a_arr = np.array([scores[c][1] for c in codes])
    m_z = (m_arr - m_arr.mean()) / (m_arr.std() + 1e-10)
    a_z = (a_arr - a_arr.mean()) / (a_arr.std() + 1e-10)
    composite = MOMR2_W * m_z + AMIHUD_W * a_z
    top2_idx = np.argsort(composite)[-2:][::-1]
    return [(codes[i], SECTOR_ETFS.get(codes[i], codes[i])) for i in top2_idx]


def regime_score(close, fast_window):
    """0.6*tanh趋势 + 0.4*金叉; fast_window=250 为基础分, =30 为快线分(金叉项 sma50>sma_fast)"""
    sma_fast = close.rolling(fast_window).mean()
    sma50 = close.rolling(50).mean()
    dev = (close.iloc[-1] - sma_fast.iloc[-1]) / sma_fast.iloc[-1]
    trend = 0.5 + 0.5 * np.tanh(dev * 10)
    if fast_window == 250:
        golden = 1.0 if sma50.iloc[-1] > sma_fast.iloc[-1] else 0.0
    else:
        golden = 1.0 if sma50.iloc[-1] > sma_fast.iloc[-1] else 0.0  # 实盘口径: sma50 vs sma30
    return 0.6 * trend + 0.4 * golden


def breadth_at(etfs, offset):
    """offset=0 当日, offset=15 为15个交易日前; 3只ETF站上各自SMA50的比例"""
    up = 0
    for sym, close in etfs.items():
        sma50 = close.rolling(50).mean()
        i = -1 - offset
        if len(close) >= 50 + offset and close.iloc[i] > sma50.iloc[i]:
            up += 1
    return up / len(etfs)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"active": False, "trigger_date": None, "expire_date": None}


def send_feishu(title, content, template="red"):
    """飞书卡片推送(SURGE 触发/到期主动提醒)。用标准库 urllib, 无需额外依赖。
    webhook 从环境变量 FEISHU_WEBHOOK 读取, 未设置则静默跳过。"""
    webhook = os.environ.get("FEISHU_WEBHOOK", "")
    if not webhook:
        return
    import urllib.request
    card = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title},
                       "template": template},
            "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": content}},
                         {"tag": "note", "elements": [
                             {"tag": "plain_text", "content": "SURGE 日频监控自动推送"}]}],
        },
    }
    req = urllib.request.Request(webhook, data=json.dumps(card).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"📱 飞书已推送: {title}")
    except Exception as e:  # noqa: BLE001 - 飞书推送失败不影响信号判断
        print(f"⚠️ 飞书推送失败: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    idx, etfs = fetch_daily()
    today = str(idx.index[-1].date())
    cal = idx.index  # 用指数日历近似交易日历

    base = regime_score(idx, 250)
    s30 = regime_score(idx, 30)
    b_now = breadth_at(etfs, 0)

    def evaluate(lookback, s30_min):
        b_prev = breadth_at(etfs, lookback)
        return (base >= BASE_MIN) and (s30 >= s30_min) and (b_now > 2 / 3) and (b_prev < 1 / 3), b_prev

    trig, b_prev = evaluate(LOOKBACK, S30_MIN)
    trig_shadow, b_prev_sh = evaluate(SHADOW["lookback"], SHADOW["s30_min"])

    state = load_state()
    # 到期检查(按指数日历数交易日)
    if state.get("active") and state.get("expire_date") and today >= state["expire_date"]:
        _trigger_date = state.get("trigger_date")
        state = {"active": False, "trigger_date": None, "expire_date": None}
        print(f"[{today}] 🔓 SURGE 锁定到期, 恢复 SMA250 制度判断")
        send_feishu("🔓 SURGE 锁定到期",
                    f"触发 {_trigger_date} 起锁定已到期\n恢复 SMA250 制度判断",
                    "grey")

    if trig and not state.get("active"):
        pos = cal.get_indexer([idx.index[-1]])[0]
        expire = cal[min(pos + LOCK_DAYS - 1, len(cal) - 1)]
        # 若未来日历不足(必然), 到期日按自然日估算: 21交易日≈29自然日
        expire_date = str((idx.index[-1] + pd.Timedelta(days=29)).date()) if pos + LOCK_DAYS - 1 >= len(cal) else str(expire.date())
        state = {"active": True, "trigger_date": today, "expire_date": expire_date}
        print(f"[{today}] 🚨🚨 SURGE 触发! 广度 {b_prev:.0%}→{b_now:.0%}, SMA30分 {s30:.3f}, "
              f"SMA250分 {base:.3f} —— 次日开盘按 RISKON 进场, 锁定至 {expire_date}")

        # 选 Top-2 行业 ETF
        top2 = []
        try:
            closes, vols = fetch_sector_data()
            top2 = select_top2(closes, vols)
        except Exception as e:  # noqa: BLE001 - 选股失败不影响触发判断
            print(f"⚠️ 选股失败: {e}")

        msg = (f"广度 {b_prev:.0%}→{b_now:.0%}（≥2/3站上SMA50）\n"
               f"SMA30分 {s30:.3f} · SMA250分 {base:.3f}\n"
               f"**次日开盘按 RISKON 进场**，锁定至 {expire_date}")
        if top2:
            picks = "\n".join(f"  {i+1}. {name}（{code}）" for i, (code, name) in enumerate(top2))
            msg += f"\n\n**Top-2 行业 ETF（各50%）:**\n{picks}"
        else:
            msg += "\n\n⚠️ 行业 ETF 数据拉取失败，请手动选股"

        send_feishu("🚨 SURGE 触发", msg, "red")
    elif state.get("active"):
        print(f"[{today}] 🔒 SURGE 锁定中(触发 {state['trigger_date']}, 至 {state['expire_date']}), "
              f"制度强制 RISKON | 广度 {b_now:.0%} s30 {s30:.3f} base {base:.3f}")
    else:
        print(f"[{today}] ── 未触发 | 广度 {b_now:.0%}(15日前 {b_prev:.0%}) "
              f"s30 {s30:.3f} base {base:.3f}")

    state["checked_date"] = today
    state["detail"] = {"breadth_now": round(b_now, 3), "breadth_prev": round(b_prev, 3),
                       "s30": round(s30, 3), "base": round(base, 3)}

    # ===== 影子候选(20d/0.80)记录, 不影响正式信号 =====
    sh = state.get("shadow", {"active": False, "trigger_date": None, "expire_date": None})
    if sh.get("active") and sh.get("expire_date") and today >= sh["expire_date"]:
        sh = {"active": False, "trigger_date": None, "expire_date": None}
    if trig_shadow and not sh.get("active"):
        sh = {"active": True, "trigger_date": today,
              "expire_date": str((idx.index[-1] + pd.Timedelta(days=29)).date())}
        print(f"[{today}] 🌓 影子候选(20d/0.80)触发")
    # 分歧样本 = 转正/否决的天然裁决证据, 高亮记录
    if trig != trig_shadow or bool(state.get("active")) != bool(sh.get("active")):
        print(f"[{today}] ⚖️ 分歧样本! 正式(15d/0.70): 触发={trig} 锁={state.get('active')} | "
              f"影子(20d/0.80): 触发={trig_shadow} 锁={sh.get('active')} —— 记录留证, 事后复盘裁决")
    sh["last_trig"] = bool(trig_shadow)
    state["shadow"] = sh

    if not args.dry:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
