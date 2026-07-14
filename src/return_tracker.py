"""
收益率追踪模块
每次生成信号时记录持仓，下次运行时计算期间收益率
"""
import os, json, sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACKER_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals", "portfolio_tracker.json")


def init_tracker(initial_capital=1000000):
    """初始化追踪器"""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "initial_capital": initial_capital,
            "current_cash": initial_capital,
            "total_invested": 0,
            "total_return": 0,
            "annual_return": 0,
            "start_date": datetime.now().strftime('%Y-%m-%d'),
            "holdings": {},     # {code: {shares, buy_price, buy_date}}
            "history": [],      # [{date, action, details}]
        }
        os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)

    return data


def update_holdings(signal_report, stock_prices=None):
    """
    根据信号更新持仓记录

    signal_report: run_monthly.py 输出的 report dict
    stock_prices: {code: current_price} 可选，用于记录买入价
    """
    tracker = init_tracker()

    date = signal_report['date']
    regime = signal_report['regime']['regime']
    selected = signal_report.get('selected_stocks', [])

    # 获取当前持仓
    old_holdings = set(tracker['holdings'].keys())
    new_holdings = set(s['code'] for s in selected)

    to_sell = old_holdings - new_holdings
    to_buy = new_holdings - old_holdings
    to_hold = old_holdings & new_holdings

    # 估算总资产
    total_value = _estimate_portfolio_value(tracker, stock_prices)

    action_log = {
        "date": date,
        "regime": regime,
        "to_buy": list(to_buy),
        "to_sell": list(to_sell),
        "to_hold": list(to_hold),
        "portfolio_value_before": total_value,
    }

    # 卖出的股票：清算
    for code in to_sell:
        if code in tracker['holdings']:
            h = tracker['holdings'][code]
            # 用当前价格算卖出金额
            sell_price = stock_prices.get(code, 0) if stock_prices else h.get('current_price', h['buy_price'])
            proceeds = h['shares'] * sell_price
            tracker['current_cash'] += proceeds
            tracker['total_invested'] -= h['shares'] * h['buy_price']
            del tracker['holdings'][code]

    # 持有的股票：更新现价
    for code in to_hold:
        if code in tracker['holdings'] and stock_prices and code in stock_prices:
            tracker['holdings'][code]['current_price'] = stock_prices[code]

    # 买入的股票：从现金中扣除
    if to_buy and regime != 'CRISIS':
        # 等权分配可用资金
        n_stocks = len(new_holdings)
        if n_stocks > 0:
            alloc = {
                'RISKON': 0.95,
                'NEUTRAL': 0.60,
                'RISKOFF': 0.30,
                'CRISIS': 0.0,
            }
            equity_pct = alloc.get(regime, 0.60)
            available = tracker['current_cash'] * equity_pct

            per_stock_cash = available / len(to_buy) if to_buy else 0

            for code in to_buy:
                buy_price = stock_prices.get(code, 0) if stock_prices else 0
                if buy_price > 0:
                    shares = int(per_stock_cash / buy_price / 100) * 100
                    cost = shares * buy_price
                    tracker['current_cash'] -= cost
                    tracker['total_invested'] += cost
                    tracker['holdings'][code] = {
                        "shares": shares,
                        "buy_price": buy_price,
                        "buy_date": date,
                        "current_price": buy_price,
                    }

    # 重新计算收益率
    new_value = _estimate_portfolio_value(tracker, stock_prices)
    tracker['total_return'] = new_value / tracker['initial_capital'] - 1

    # 年化
    start = datetime.strptime(tracker['start_date'], '%Y-%m-%d')
    days = (datetime.now() - start).days
    if days > 30:
        tracker['annual_return'] = (1 + tracker['total_return']) ** (365 / days) - 1

    action_log["portfolio_value_after"] = new_value
    action_log["return_pct"] = round(tracker['total_return'] * 100, 2)
    tracker['history'].append(action_log)

    # 保存
    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2, default=str)

    return tracker


def _estimate_portfolio_value(tracker, stock_prices):
    """估算组合总市值"""
    value = tracker['current_cash']
    for code, h in tracker['holdings'].items():
        price = h.get('current_price', h['buy_price'])
        if stock_prices and code in stock_prices:
            price = stock_prices[code]
        value += h['shares'] * price
    return value


def print_return_summary():
    """打印收益率摘要"""
    if not os.path.exists(TRACKER_FILE):
        print("  尚无交易记录")
        return

    with open(TRACKER_FILE, 'r') as f:
        t = json.load(f)

    print()
    print("=" * 50)
    print("  组合收益追踪")
    print("=" * 50)
    print(f"  起始资金: {t['initial_capital']:,.0f} 元")
    print(f"  起始日期: {t['start_date']}")
    print(f"  持仓数量: {len(t['holdings'])} 只")
    print(f"  累计收益: {t['total_return']*100:+.2f}%")
    if t.get('annual_return', 0) != 0:
        print(f"  年化收益: {t['annual_return']*100:+.2f}%")
    print(f"  调仓次数: {len(t['history'])} 次")
    print()

    if t['history']:
        print("  调仓记录:")
        for h in t['history'][-5:]:
            buys = len(h['to_buy'])
            sells = len(h['to_sell'])
            holds = len(h['to_hold'])
            ret = h.get('return_pct', 0)
            print(f"    {h['date']} | {h['regime']:8s} | buy:{buys} sell:{sells} hold:{holds} | 累计:{ret:+.1f}%")


if __name__ == '__main__':
    print_return_summary()
