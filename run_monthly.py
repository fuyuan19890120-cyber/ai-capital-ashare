#!/usr/bin/env python3
"""
月度运行脚本
每月最后一个交易日盘后执行：
  python run_monthly.py

输出：
  1. 当前市场制度
  2. 精选个股（Top-15）+ 风控验证
  3. 调仓清单（与上次对比）
  4. Git 账本自动提交
"""
import os, sys, json, csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signal_generator import generate_signals, print_report

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SIGNAL_FILE = os.path.join(PROJECT_ROOT, "signals", "latest.json")
LEDGER_FILE = os.path.join(PROJECT_ROOT, "signals", "ledger.csv")
os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)


def write_ledger(report, prev_report=None):
    """
    写入交易账本（CSV 格式，Git-tracked，不可篡改）。

    字段：date, regime, regime_score, code, action, weight_pct, reason
    """
    date = report['date']
    regime = report['regime']['regime']
    regime_score = report['regime']['score']
    alloc = report.get('allocation', {})

    # 判断操作类型
    curr_codes = set(s['code'] for s in report.get('selected_stocks', []))
    prev_codes = set()
    if prev_report:
        prev_codes = set(s['code'] for s in prev_report.get('selected_stocks', []))

    rows = []
    for s in report.get('selected_stocks', []):
        code = s['code']
        if code not in prev_codes:
            action = 'BUY'
        elif code in curr_codes:
            action = 'HOLD'
        else:
            action = 'BUY'
        rows.append([date, regime, regime_score, code, action, s['score'], ''])

    for code in prev_codes - curr_codes:
        rows.append([date, regime, regime_score, code, 'SELL', 0, '调出'])

    # 防御资产
    if regime == 'CRISIS':
        rows.append([date, regime, regime_score, '国债ETF(sh511010)', 'BUY', 0, 'CRISIS防御'])
        rows.append([date, regime, regime_score, '黄金ETF(sh518880)', 'BUY', 0, 'CRISIS防御'])

    # 追加写入
    file_exists = os.path.exists(LEDGER_FILE)
    with open(LEDGER_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['date', 'regime', 'regime_score', 'code', 'action', 'weight', 'note'])
        writer.writerows(rows)

    return len(rows)


def git_commit():
    """自动 commit 信号和账本到 git（如果在 git 仓库中）"""
    import subprocess
    try:
        git_dir = os.path.join(PROJECT_ROOT, '.git')
        if not os.path.exists(git_dir):
            return False

        subprocess.run(['git', '-C', PROJECT_ROOT, 'add',
                        'signals/latest.json', 'signals/portfolio_tracker.json',
                        'signals/ledger.csv', 'signals/trade_list_*.txt'],
                       capture_output=True)

        result = subprocess.run(['git', '-C', PROJECT_ROOT, 'diff', '--staged', '--quiet'],
                                capture_output=True)
        if result.returncode != 0:
            date_str = datetime.now().strftime('%Y-%m')
            subprocess.run(['git', '-C', PROJECT_ROOT, 'commit',
                            '-m', f'📊 {date_str} 月度信号更新',
                            '--allow-empty'],
                           capture_output=True)
            return True
    except Exception:
        pass
    return False


def main():
    print("=" * 60)
    print("  AI Capital A-Share — 月度信号")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 生成信号
    report = generate_signals()
    print_report(report)

    if report is None:
        return

    # 加载上次信号做对比
    prev_report = None
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, 'r') as f:
            prev_report = json.load(f)

    # 对比调仓
    if prev_report and prev_report.get('selected_stocks'):
        prev_codes = set(s['code'] for s in prev_report['selected_stocks'])
        curr_codes = set(s['code'] for s in report['selected_stocks'])

        to_buy = curr_codes - prev_codes
        to_sell = prev_codes - curr_codes
        to_hold = curr_codes & prev_codes

        print()
        print("  📋 调仓对比（vs 上期）:")
        print(f"     继续持有: {len(to_hold)} 只")
        if to_sell:
            print(f"     🔴 卖出: {', '.join(sorted(to_sell))}")
        if to_buy:
            print(f"     🟢 买入: {', '.join(sorted(to_buy))}")
        if not to_sell and not to_buy:
            print(f"     ✅ 持仓不变")
    else:
        print()
        print("  📋 首次运行，无上期对比")

    # ===== 写入 CSV 账本（新增！）=====
    n_ledger = write_ledger(report, prev_report)
    print(f"\n  📒 账本写入: {n_ledger} 条记录 → signals/ledger.csv")

    # ===== 记录收益 + 写入 Obsidian =====
    from src.return_tracker import update_holdings, print_return_summary
    from src.obsidian_logger import full_log

    update_holdings(report)
    print_return_summary()
    full_log(report)

    # 保存本次信号
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 生成交易清单
    trade_list_path = os.path.join(PROJECT_ROOT, "signals", f"trade_list_{report['date']}.txt")
    with open(trade_list_path, 'w') as f:
        f.write(f"调仓清单 — {report['date']}\n")
        f.write(f"制度: {report['regime']['regime']}\n\n")

        r = report['regime']
        if r['regime'] == 'CRISIS':
            f.write("⚠️ CRISIS 模式：清仓所有个股，切换至债券/黄金/货币\n")
        else:
            f.write("买入以下个股（等权配置）：\n")
            for s in report['selected_stocks']:
                f.write(f"  {s['code']}\n")

    print(f"\n  交易清单已保存: {trade_list_path}")
    print(f"  信号文件已保存: {SIGNAL_FILE}")

    # ===== Git 自动提交（新增！）=====
    committed = git_commit()
    if committed:
        print("  ✅ 已自动提交到 Git")
    else:
        print("  ℹ️  Git 未提交（无变更或非 git 仓库）")

    print("=" * 60)


if __name__ == '__main__':
    main()
