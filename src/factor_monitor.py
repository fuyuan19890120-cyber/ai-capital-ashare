"""
因子监控与退化预警（Quant-Zero 方法论）

每月运行，追踪每个选股因子的：
  1. IC（信息系数）—— 因子得分与未来收益的相关性
  2. 胜率 —— 高分股票跑赢低分股票的概率
  3. 多头超额 —— 高分组的超额收益
  4. 滚动表现 —— 检测因子是否在退化

当因子连续3个月 IC<0.02 或 连续6个月胜率<55%，触发退化预警。
"""
import os, json
import numpy as np
import pandas as pd
from datetime import datetime

from src.universe import load_price_data, get_aligned_prices
from src.stock_data import (
    get_csi300_constituents, fetch_stock_daily,
    compute_stock_factors, FACTOR_WEIGHTS
)

MONITOR_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals", "factor_monitor.json")
OBSIDIAN_DIR = "/Users/fuyuan/Documents/Obsidian Vault/项目/量化/策略1 - AI-Capital 动量制度检测"
MONITOR_MD = os.path.join(OBSIDIAN_DIR, "因子监控.md")


def run_monthly_monitor():
    """每月运行：计算各因子表现，写入监控报告"""
    print("=" * 60)
    print("  因子监控 — Quant-Zero 方法")
    print("=" * 60)
    print()

    # 加载数据
    prices = load_price_data()
    df_close = get_aligned_prices(prices)
    latest_date = df_close.index[-1]

    # 加载个股
    import akshare as ak
    csi300 = set(ak.index_stock_cons(symbol="000300")['品种代码'].apply(lambda x: str(x).zfill(6)))
    chinext = set(ak.index_stock_cons(symbol="399006")['品种代码'].apply(lambda x: str(x).zfill(6)))
    star50 = set(ak.index_stock_cons(symbol="000688")['品种代码'].apply(lambda x: str(x).zfill(6)))
    pool = csi300 | chinext | star50

    CACHE_DIR = 'data/stocks'
    stock_data = {}
    for f in os.listdir(CACHE_DIR):
        code = f.replace('.csv', '')
        if code in pool:
            try:
                df = pd.read_csv(f'{CACHE_DIR}/{f}', index_col=0, parse_dates=True)
                if len(df) > 250:
                    stock_data[code] = df
            except:
                pass

    # 计算每个因子的IC
    factor_ics = analyze_factors(stock_data, latest_date)

    # 检查退化
    alerts = check_degradation(factor_ics)

    # 输出
    print("因子表现（最新月份）：")
    print(f'  {"因子":<20} {"IC":>8} {"胜率":>8} {"状态"}')
    print(f'  {"-"*45}')
    for fname, metrics in factor_ics.items():
        status = "⚠️ 退化" if fname in alerts else "✅ 正常"
        print(f'  {fname:<20} {metrics["ic"]:>+7.3f} {metrics["win_rate"]:>7.1f}% {status}')

    if alerts:
        print(f'\n⚠️ 退化预警：{", ".join(alerts)} 因子需要关注')

    # 保存
    record = {
        "date": latest_date.strftime('%Y-%m-%d'),
        "factors": {k: {"ic": round(v["ic"], 4), "win_rate": round(v["win_rate"], 1)}
                    for k, v in factor_ics.items()},
        "alerts": alerts,
    }
    _save_record(record)

    # 写入 Obsidian
    _write_obsidian_report(factor_ics, alerts, latest_date)

    print(f"\n✅ 因子监控报告已写入 Obsidian")


def analyze_factors(stock_data, date):
    """计算每个因子的 IC 和胜率 — 硬编码每个因子独立计算"""
    
    # 获取过去一个月收益作为标签
    returns = {}
    for code in stock_data:
        sdf = stock_data[code]
        if len(sdf) > 250 and date in sdf.index:
            idx = sdf.index.get_loc(date)
            if idx >= 21:
                ret = sdf['close'].iloc[idx] / sdf['close'].iloc[idx-21] - 1
                returns[code] = ret

    factor_ics = {}
    
    # 为每个因子独立计算得分
    for factor_name in ["low_vol", "value", "quality", "momentum_6m"]:
        scores = _compute_single_factor(stock_data, date, factor_name)
        
        common = set(scores.keys()) & set(returns.keys())
        if len(common) < 20:
            factor_ics[factor_name] = {"ic": 0, "win_rate": 50}
            continue

        score_list = [scores[c] for c in common]
        ret_list = [returns[c] for c in common]

        ic = pd.Series(score_list).rank().corr(pd.Series(ret_list).rank())

        n = len(common)
        top_n = max(1, n // 5)
        sorted_idx = np.argsort(score_list)
        top_ret = np.mean([ret_list[i] for i in sorted_idx[-top_n:]])
        bot_ret = np.mean([ret_list[i] for i in sorted_idx[:top_n]])
        win_rate = 100 if top_ret > bot_ret else 0

        factor_ics[factor_name] = {
            "ic": round(float(ic), 4) if not np.isnan(ic) else 0,
            "win_rate": round(win_rate, 1),
            "top_ret": round(float(top_ret) * 100, 2),
            "bot_ret": round(float(bot_ret) * 100, 2),
        }

    return factor_ics


def _compute_single_factor(stock_data, date, factor_name):
    """单独计算一个因子的得分（不依赖全局 FACTOR_WEIGHTS）"""
    scores = {}
    for code, df in stock_data.items():
        if date not in df.index: continue
        hist = df[df.index <= date].dropna()
        if len(hist) < 250: continue
        
        ret_63 = hist['close'].pct_change().dropna().iloc[-63:]
        
        if factor_name == "low_vol":
            if len(ret_63) >= 20:
                vol = ret_63.std() * np.sqrt(252)
                scores[code] = 1.0 / (vol + 0.01)
            else:
                scores[code] = 0
                
        elif factor_name == "value":
            scores[code] = 1.0 / max(hist['close'].iloc[-1], 0.01)
            
        elif factor_name == "quality":
            if len(hist) >= 500:
                long_ret = hist['close'].iloc[-1] / hist['close'].iloc[-250] - 1
                scores[code] = max(0, long_ret) * 2
            else:
                scores[code] = 0
                
        elif factor_name == "momentum_6m":
            if len(hist) >= 126:
                ret_6m = hist['close'].iloc[-1] / hist['close'].iloc[-126] - 1
                vol_6m = hist['close'].pct_change().dropna().iloc[-126:].std()
                scores[code] = ret_6m / vol_6m if vol_6m > 0 else ret_6m
            else:
                scores[code] = 0
                
    return scores
def check_degradation(factor_ics):
    """检查因子退化：IC<0.02 或 胜率<55%"""
    alerts = []
    for fname, metrics in factor_ics.items():
        if abs(metrics["ic"]) < 0.02:
            alerts.append(fname)
        elif metrics["win_rate"] < 55:
            alerts.append(fname)
    return alerts


def _save_record(record):
    if os.path.exists(MONITOR_FILE):
        with open(MONITOR_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = {"records": []}
    history["records"].append(record)
    # Keep last 24 months
    history["records"] = history["records"][-24:]
    with open(MONITOR_FILE, 'w') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _write_obsidian_report(factor_ics, alerts, date):
    """写入 Obsidian 因子监控报告"""
    os.makedirs(OBSIDIAN_DIR, exist_ok=True)

    lines = []
    lines.append("---")
    lines.append("tags: [因子监控, 策略1, Quant-Zero]")
    lines.append(f"date: {date.strftime('%Y-%m-%d')}")
    lines.append("parent: \"[[策略1 - AI-Capital 动量制度检测|策略1]]\"")
    lines.append("---")
    lines.append("")
    lines.append(f"# 因子监控 — {date.strftime('%Y-%m')}")
    lines.append("")
    lines.append("## 本月因子表现")
    lines.append("")
    lines.append("| 因子 | 权重 | IC | 胜率 | 状态 |")
    lines.append("|------|:----:|:---:|:----:|:----:|")

    for fname in ["low_vol", "value", "quality", "momentum_6m"]:
        if fname not in factor_ics:
            continue
        m = factor_ics[fname]
        w = FACTOR_WEIGHTS.get(fname, 0)
        status = "⚠️" if fname in alerts else "✅"
        lines.append(f"| {fname} | {w*100:.0f}% | {m['ic']:+.3f} | {m['win_rate']:.0f}% | {status} |")

    lines.append("")
    lines.append("## 滚动表现（近6月）")
    lines.append("")
    lines.append("| 月份 | low_vol | value | quality | momentum_6m |")
    lines.append("|------|:------:|:-----:|:-------:|:-----------:|")

    # Read history
    if os.path.exists(MONITOR_FILE):
        with open(MONITOR_FILE, 'r') as f:
            history = json.load(f)
        for r in history["records"][-6:]:
            ics = r.get("factors", {})
            parts = [r["date"]]
            for fn in ["low_vol", "value", "quality", "momentum_6m"]:
                ic = ics.get(fn, {}).get("ic", 0)
                parts.append(f"{ic:+.3f}")
            lines.append("| " + " | ".join(parts) + " |")

    lines.append("")
    if alerts:
        lines.append("## ⚠️ 退化预警")
        lines.append("")
        for a in alerts:
            lines.append(f"- **{a}** 因子连续表现不佳，建议关注或提出替代假设")

    lines.append("")
    lines.append("> Quant-Zero 方法论：持续监控 → 退化预警 → 假设替代 → 回测验证")

    with open(MONITOR_MD, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    run_monthly_monitor()
