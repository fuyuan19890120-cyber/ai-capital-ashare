"""
Obsidian 交易记录模块 v2
每月自动将交易记录和收益率写入 Obsidian
路径：项目/量化/策略1/交易记录/
"""
import os, json
from datetime import datetime

VAULT_DIR = "/Users/fuyuan/Documents/Obsidian Vault/项目/量化/策略1 - AI-Capital"  # 2026-07-19 修复: 原路径指向不存在的目录
TRADE_LOG_DIR = os.path.join(VAULT_DIR, "交易记录")
SUMMARY_FILE = os.path.join(TRADE_LOG_DIR, "收益汇总.md")
TRACKER_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals", "portfolio_tracker.json")


def init():
    """确保目录和汇总文件存在"""
    os.makedirs(TRADE_LOG_DIR, exist_ok=True)
    if not os.path.exists(SUMMARY_FILE):
        content = """---
tags:
  - 量化交易
  - 策略1
  - 收益追踪
created: "{{date}}"
parent: "[[../策略1 - AI-Capital 动量制度检测|策略1]]"
---

# 策略1 收益追踪

> 起始资金：100 万（模拟）
> 起始日期：2026-07-14
> 策略：SMA250 制度择时 + 多因子选股（CSI300+创业板+科创50）

---

## 收益概览

| 日期 | 累计收益 | 年化收益 | 调仓次数 | 备注 |
|------|:--------:|:--------:|:--------:|------|
"""
        content = content.replace("{{date}}", datetime.now().strftime('%Y-%m-%d'))
        with open(SUMMARY_FILE, 'w') as f:
            f.write(content)


def log_monthly_trade(signal_report, prev_positions=None, entry_prices=None, month_end_prices=None):
    """
    记录月度交易到 Obsidian

    signal_report: 信号生成器输出的 report
    prev_positions: 上期持仓 {code: shares}
    entry_prices: 买入价格 {code: price}
    month_end_prices: 月末价格 {code: price}（用于算月度收益）
    """
    init()

    date = signal_report['date']
    regime = signal_report['regime']
    selected = signal_report.get('selected_stocks', [])

    regime_emoji = {'RISKON': '🟢', 'NEUTRAL': '🟡', 'RISKOFF': '🟠', 'CRISIS': '🔴'}
    emoji = regime_emoji.get(regime['regime'], '❓')

    # 获取股票名称
    names = _get_stock_names()

    log_file = os.path.join(TRADE_LOG_DIR, f"交易记录_{date}.md")

    lines = []
    lines.append(f"---")
    lines.append(f'tags: [量化交易, 策略1, 交易记录]')
    lines.append(f'date: {date}')
    lines.append(f'parent: "[[../策略1 - AI-Capital 动量制度检测|策略1]]"')
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"# 交易记录 — {date}")
    lines.append(f"")

    # 市场制度
    lines.append(f"## 市场制度")
    lines.append(f"")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 制度 | {emoji} **{regime['regime']}** (score={regime['score']:.3f}) |")
    lines.append(f"| 沪深300 | {regime['price']} |")
    lines.append(f"| SMA50 | {regime['sma50']} |")
    lines.append(f"| SMA250 | {regime['sma250']} |")
    lines.append(f"| 偏离年线 | {regime['deviation_pct']:+.1f}% |")
    lines.append(f"")

    # 仓位配置
    lines.append(f"## 仓位配置")
    lines.append(f"")
    alloc = signal_report.get('allocation', {})
    n_stocks = len(selected) if selected else 0
    lines.append(f"| 资产类别 | 比例 |")
    lines.append(f"|----------|:----:|")
    for k, v in alloc.items():
        lines.append(f"| {k} | {v} |")
    if n_stocks > 0:
        per_stock = 95.0 / n_stocks if regime['regime'] == 'RISKON' else 60.0 / n_stocks
        lines.append(f"| 每只个股 | {per_stock:.1f}% |")
    lines.append(f"")

    # 精选个股（含名称、得分、价格）
    lines.append(f"## 持仓明细 (Top-{len(selected)})")
    lines.append(f"")

    if entry_prices and month_end_prices:
        # 有月末数据：显示完整月度表现
        lines.append(f"| # | 代码 | 名称 | 得分 | 买入价 | 月末价 | 月度收益 |")
        lines.append(f"|---|------|------|:----:|:------:|:------:|:--------:|")
        for i, s in enumerate(selected, 1):
            code = s['code']
            name = names.get(code, '?')
            score = s['score']
            entry = entry_prices.get(code, '-')
            end = month_end_prices.get(code, '-')
            if isinstance(entry, (int, float)) and isinstance(end, (int, float)) and entry > 0:
                ret = (end / entry - 1) * 100
                ret_str = f"{ret:+.1f}%"
                entry_str = f"{entry:.2f}"
                end_str = f"{end:.2f}"
            else:
                ret_str = '-'
                entry_str = str(entry) if entry else '-'
                end_str = str(end) if end else '-'
            lines.append(f"| {i} | {code} | {name} | {score:.2f} | {entry_str} | {end_str} | {ret_str} |")
    else:
        # 首次记录：只有买入价
        lines.append(f"| # | 代码 | 名称 | 得分 | 买入价 |")
        lines.append(f"|---|------|------|:----:|:------:|")
        for i, s in enumerate(selected, 1):
            code = s['code']
            name = names.get(code, '?')
            score = s['score']
            entry = entry_prices.get(code, '-') if entry_prices else '-'
            entry_str = f"{entry:.2f}" if isinstance(entry, (int, float)) else str(entry)
            lines.append(f"| {i} | {code} | {name} | {score:.2f} | {entry_str} |")
        lines.append(f"")
        lines.append(f"> ⚠️ 月末价待 7 月 31 日更新")

    lines.append(f"")

    # 调仓对比
    if prev_positions:
        prev_codes = set(prev_positions.keys())
        curr_codes = set(s['code'] for s in selected)
        to_buy = curr_codes - prev_codes
        to_sell = prev_codes - curr_codes
        to_hold = curr_codes & prev_codes

        lines.append(f"## 调仓操作")
        lines.append(f"")
        lines.append(f"| 操作 | 数量 | 详情 |")
        lines.append(f"|------|:----:|------|")
        lines.append(f"| ✅ 继续持有 | {len(to_hold)} | {', '.join(sorted(to_hold)) if to_hold else '—'} |")
        if to_sell:
            lines.append(f"| 🔴 卖出 | {len(to_sell)} | {', '.join(sorted(to_sell))} |")
        if to_buy:
            lines.append(f"| 🟢 买入 | {len(to_buy)} | {', '.join(sorted(to_buy))} |")
        if not to_sell and not to_buy:
            lines.append(f"| ✅ 持仓不变 | — | 所有持仓保持不变 |")
    else:
        lines.append(f"## 调仓操作")
        lines.append(f"")
        lines.append(f"- 🆕 首次建仓，买入全部 {len(selected)} 只")

    lines.append(f"")

    # 板块分布
    boards = {}
    for s in selected:
        code = s['code']
        if code.startswith('688'): b = '科创板'
        elif code.startswith(('300', '301')): b = '创业板'
        elif code.startswith('002'): b = '深市主板'
        else: b = '沪市主板'
        boards[b] = boards.get(b, 0) + 1

    lines.append(f"## 板块分布")
    lines.append(f"")
    for b, c in sorted(boards.items(), key=lambda x: -x[1]):
        lines.append(f"- {b}：{c} 只")
    lines.append(f"")

    with open(log_file, 'w') as f:
        f.write('\n'.join(lines))


def update_previous_month_end(prev_date, month_end_prices):
    """为上期交易记录补充月末价格"""
    prev_log = os.path.join(TRADE_LOG_DIR, f"交易记录_{prev_date}.md")
    if not os.path.exists(prev_log):
        return

    with open(prev_log, 'r') as f:
        content = f.read()

    # 简单处理：在文件末尾添加月末更新
    update_section = f"""

---

## 月末更新

| 代码 | 月末价 | 月度收益 |
|------|:------:|:--------:|
"""
    for code, price in sorted(month_end_prices.items()):
        update_section += f"| {code} | {price:.2f} | — |\n"

    if "## 月末更新" not in content:
        content += update_section
        with open(prev_log, 'w') as f:
            f.write(content)


def update_summary():
    """更新收益汇总"""
    init()

    if not os.path.exists(TRACKER_FILE):
        return

    with open(TRACKER_FILE, 'r') as f:
        t = json.load(f)

    total_ret = t.get('total_return', 0) * 100
    ann_ret = t.get('annual_return', 0) * 100
    trades = len(t.get('history', []))

    with open(SUMMARY_FILE, 'r') as f:
        content = f.read()

    today = datetime.now().strftime('%Y-%m-%d')
    # 检查今天的日期是否已经在表中
    if today not in content:
        last_trade = t['history'][-1] if t['history'] else {}
        note = last_trade.get('regime', '—')
        new_row = f"| {today} | {total_ret:+.2f}% | {ann_ret:+.2f}% | {trades} | {note} |"
        if '\n---' in content:
            parts = content.rsplit('\n---', 1)
            content = parts[0] + '\n' + new_row + '\n---' + parts[1]
        else:
            content += '\n' + new_row + '\n'

    with open(SUMMARY_FILE, 'w') as f:
        f.write(content)


def _get_stock_names():
    """获取股票代码→名称映射（从缓存数据中）"""
    names = {}
    # Try to get from existing data
    try:
        import akshare as ak
        for idx in ["000300", "399006", "000688"]:
            try:
                df = ak.index_stock_cons(symbol=idx)
                for _, row in df.iterrows():
                    code = str(row['品种代码']).zfill(6)
                    names[code] = row['品种名称']
            except:
                pass
    except:
        pass
    return names


def full_log(signal_report):
    """完整记录：交易日志 + 更新汇总"""
    # 获取当前价格作为买入价
    entry_prices = _get_current_prices(signal_report)

    # 加载上次持仓做对比
    prev_positions = {}
    prev_signal_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals", "latest.json")
    if os.path.exists(prev_signal_file):
        with open(prev_signal_file, 'r') as f:
            prev = json.load(f)
            for s in prev.get('selected_stocks', []):
                prev_positions[s['code']] = 1

    log_monthly_trade(signal_report, prev_positions, entry_prices)
    update_summary()
    print(f"\n  ✅ 交易记录已写入 Obsidian")
    print(f"     路径：{TRADE_LOG_DIR}/交易记录_{signal_report['date']}.md")


def _get_current_prices(signal_report):
    """获取当前持仓的买入价格"""
    prices = {}
    try:
        import pandas as pd
        CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stocks")
        for s in signal_report.get('selected_stocks', []):
            code = s['code']
            cache_path = os.path.join(CACHE_DIR, f"{code}.csv")
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not df.empty:
                    prices[code] = round(float(df['close'].iloc[-1]), 2)
    except:
        pass
    return prices
