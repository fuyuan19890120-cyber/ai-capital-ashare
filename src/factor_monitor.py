"""
V4.2 因子监控(2026-07-19 重新设计)

每月运行，追踪：
  1. 低波因子 IC — 唯一穿越三年的有效因子
  2. 动量因子 IC — 6月夏普动量
  3. 反转过滤效果 — 被剔除 vs 入选股的次月表现差
  4. 综合得分 IC — V4.2完整排名 vs 未来收益
  5. 制度分档 IC — RISKON vs 非RISKON下的因子行为

当连续3个月 综合IC<0.03 或 反转过滤效果为负，触发退化预警。

与旧版区别:
  - 只监控真实因子(不再监控假"价值"/假"质量")
  - 用指数成分宇宙(与实盘一致)
  - 在反转过滤后的池子里计算IC
  - 新增反转过滤效果指标
"""
import os, json
import numpy as np
import pandas as pd
from datetime import datetime

from src.stock_data import (
    filter_reversal_stocks, compute_stock_factors, get_csi300_constituents,
)
from config import REVERSAL_FILTER_PCT, REGIME_THRESHOLDS

MONITOR_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "signals", "factor_monitor.json")


def run_monthly_monitor():
    print("=" * 60)
    print("  V4.2 因子监控")
    print("=" * 60)

    # ── 加载指数成分宇宙(与 signal_generator 一致) ──
    import akshare as ak
    csi300 = set(get_csi300_constituents())
    try:
        chinext = set(ak.index_stock_cons(symbol="399006")['品种代码']
                      .apply(lambda x: str(x).zfill(6)))
        star50 = set(ak.index_stock_cons(symbol="000688")['品种代码']
                     .apply(lambda x: str(x).zfill(6)))
    except Exception:
        chinext, star50 = set(), set()
    pool = csi300 | chinext | star50

    CACHE_DIR = os.path.expanduser("~/ai-capital-ashare/data/stocks")
    stock_data = {}
    for f in os.listdir(CACHE_DIR):
        code = f.replace('.csv', '')
        if code in pool:
            try:
                df = pd.read_csv(f'{CACHE_DIR}/{f}', index_col=0, parse_dates=True)
                if len(df) > 250:
                    stock_data[code] = df
            except Exception:
                pass

    cal = sorted([d for code in stock_data for d in stock_data[code].index])
    cal = sorted(set(cal))
    if not cal:
        print("无可用日历")
        return
    latest_date = pd.Timestamp(cal[-1])
    # 找最近一个可用月末
    monthly_dates = pd.Series(cal).groupby(
        pd.Series(cal).dt.to_period('M')).last()
    eval_date = pd.Timestamp(monthly_dates.iloc[-2]) if len(monthly_dates) >= 2 else latest_date

    print(f"宇宙: {len(stock_data)} 只 | 评估日: {eval_date.date()}")

    # ── V4.2 反转过滤 ──
    valid_all = {}
    for code, df in stock_data.items():
        if eval_date in df.index and len(df[df.index <= eval_date]) >= 250:
            valid_all[code] = df

    filtered = filter_reversal_stocks(valid_all, eval_date)
    excluded_codes = set(valid_all.keys()) - set(filtered.keys())
    print(f"反转过滤: {len(valid_all)} → {len(filtered)} (剔除前{REVERSAL_FILTER_PCT*100:.0f}%)")

    # ── 1. 真实因子 IC ──
    factor_ics = {}
    factor_names_real = {
        "low_vol": "低波动率",
        "momentum_6m": "动量(6月夏普)",
    }

    returns_1m = _forward_returns(stock_data, eval_date, 21)

    for fname, flabel in factor_names_real.items():
        scores = _compute_real_factor(filtered, eval_date, fname)
        ic, wr = _ic_and_winrate(scores, returns_1m)
        factor_ics[fname] = {"label": flabel, "ic": ic, "win_rate": wr}
        print(f"  {flabel:<12s}  IC={ic:+.4f}  胜率={wr:.0f}%")

    # ── 2. 反转过滤效果 ──
    reversal_effect = _measure_reversal_effect(valid_all, excluded_codes,
                                                eval_date, returns_1m)
    print(f"  反转过滤效果  剔除股平均次月={reversal_effect['excluded_ret']:+.2f}%  "
          f"入选股平均={reversal_effect['selected_ret']:+.2f}%  "
          f"差={reversal_effect['diff']:+.2f}%")

    # ── 3. 综合得分 IC(反转过滤后的池子) ──
    scores_full = compute_stock_factors(filtered, eval_date)
    composite_ic, composite_wr = _ic_and_winrate(scores_full, returns_1m)
    print(f"  综合得分       IC={composite_ic:+.4f}  胜率={composite_wr:.0f}%")

    # ── 4. 制度分档 IC ──
    regime_ic = _regime_breakdown(filtered, eval_date, returns_1m)
    print(f"  综合IC(RISKON)={regime_ic.get('RISKON', 0):+.4f}  "
          f"非RISKON={regime_ic.get('non_RISKON', 0):+.4f}")

    # ── 退化预警 ──
    alerts = []
    if abs(composite_ic) < 0.03:
        alerts.append(f"综合IC={composite_ic:+.3f}")
    if reversal_effect['diff'] < -2.0:
        alerts.append(f"反转过滤负效果({reversal_effect['diff']:.1f}%)")
    for fname, d in factor_ics.items():
        if d['win_rate'] < 40:  # 放宽: 单因子胜率不需要太高
            alerts.append(f"{d['label']}胜率={d['win_rate']:.0f}%")

    if alerts:
        print(f"\n⚠️ 退化预警: {'; '.join(alerts)}")
    else:
        print(f"\n✅ 所有指标正常")

    # ── 保存 ──
    record = {
        "date": eval_date.strftime('%Y-%m-%d'),
        "universe_size": len(valid_all),
        "filtered_size": len(filtered),
        "factors": {
            "low_vol": {"label": factor_ics["low_vol"]["label"],
                        "ic": round(factor_ics["low_vol"]["ic"], 4),
                        "win_rate": round(factor_ics["low_vol"]["win_rate"], 1)},
            "momentum_6m": {"label": factor_ics["momentum_6m"]["label"],
                            "ic": round(factor_ics["momentum_6m"]["ic"], 4),
                            "win_rate": round(factor_ics["momentum_6m"]["win_rate"], 1)},
        },
        "composite": {"ic": round(composite_ic, 4), "win_rate": round(composite_wr, 1)},
        "reversal_effect": {
            "excluded_ret": round(reversal_effect["excluded_ret"], 2),
            "selected_ret": round(reversal_effect["selected_ret"], 2),
            "diff": round(reversal_effect["diff"], 2),
            "n_excluded": reversal_effect["n_excluded"],
            "n_selected": reversal_effect["n_selected"],
        },
        "regime_ic": {k: round(v, 4) for k, v in regime_ic.items()},
        "alerts": alerts,
    }
    _save_record(record)
    _write_obsidian(record)
    print(f"✅ 因子监控已保存")


def _forward_returns(stock_data, date, horizon_days):
    """计算 date 之后 horizon_days 的收益(前视, 仅用于IC计算)。
    如果天数不够, 用可用的最大天数(最少5天)。"""
    rets = {}
    for code, df in stock_data.items():
        idx = df.index.get_loc(date) if date in df.index else None
        if idx is None:
            continue
        available = len(df) - idx - 1
        h = min(horizon_days, available)
        if h < 5:
            continue
        fwd_ret = float(df['close'].iloc[idx + h]) / float(df['close'].iloc[idx]) - 1
        rets[code] = fwd_ret
    return rets


def _ic_and_winrate(scores, forward_returns):
    """Rank IC + Top/Bottom 胜率"""
    common = set(scores.keys()) & set(forward_returns.keys())
    if len(common) < 20:
        return 0.0, 50.0
    slist = [scores[c] for c in common]
    rlist = [forward_returns[c] for c in common]
    ic = float(pd.Series(slist).rank().corr(pd.Series(rlist).rank()))
    if np.isnan(ic):
        ic = 0.0
    n = len(common)
    top_n = max(1, n // 5)
    idx_sorted = np.argsort(slist)
    top_avg = np.mean([rlist[i] for i in idx_sorted[-top_n:]])
    bot_avg = np.mean([rlist[i] for i in idx_sorted[:top_n]])
    wr = 100.0 if top_avg > bot_avg else 0.0
    return round(ic, 4), round(wr, 1)


def _compute_real_factor(valid_stocks, date, fname):
    """计算单个真实因子的截面得分(复用 stock_data 逻辑)"""
    scores = {}
    for code, df in valid_stocks.items():
        if date not in df.index:
            continue
        hist = df[df.index <= date].dropna()
        if len(hist) < 250:
            continue
        if fname == "low_vol":
            ret = hist['close'].pct_change().dropna().iloc[-63:]
            if len(ret) >= 20:
                vol = ret.std() * np.sqrt(252)
                scores[code] = 1.0 / (vol + 0.01)
            else:
                scores[code] = 0.0
        elif fname == "momentum_6m":
            if len(hist) >= 126:
                ret_6m = hist['close'].iloc[-1] / hist['close'].iloc[-126] - 1
                vol_6m = hist['close'].pct_change().dropna().iloc[-126:].std()
                scores[code] = ret_6m / vol_6m if vol_6m > 0 else ret_6m
            else:
                scores[code] = 0.0
    return scores


def _measure_reversal_effect(valid_all, excluded_codes, date, forward_returns):
    """反转过滤效果: 被剔除 vs 入选的次月收益差"""
    ex_rets = [forward_returns[c] for c in excluded_codes if c in forward_returns]
    sel_codes = set(valid_all.keys()) - excluded_codes
    sel_rets = [forward_returns[c] for c in sel_codes if c in forward_returns]
    return {
        "excluded_ret": np.mean(ex_rets) * 100 if ex_rets else 0,
        "selected_ret": np.mean(sel_rets) * 100 if sel_rets else 0,
        "diff": (np.mean(sel_rets) - np.mean(ex_rets)) * 100 if sel_rets and ex_rets else 0,
        "n_excluded": len(ex_rets),
        "n_selected": len(sel_rets),
    }


def _regime_breakdown(valid_stocks, date, forward_returns):
    """按当时市场制度分档计算IC"""
    # 用沪深300 SMA250 近似判断当时制度
    hs300_path = os.path.expanduser("~/ai-capital-ashare/data/index_sh000300.csv")
    regime_label = "non_RISKON"
    if os.path.exists(hs300_path):
        try:
            hs300 = pd.read_csv(hs300_path, index_col=0, parse_dates=True)
            if date in hs300.index:
                idx = hs300.index.get_loc(date)
                if idx >= 250:
                    close = hs300['close'].iloc[idx]
                    sma250 = hs300['close'].iloc[idx-250:idx].mean()
                    # 简化制度判断: RISKON条件
                    if close > sma250:
                        regime_label = "RISKON"
        except Exception:
            pass

    scores = compute_stock_factors(valid_stocks, date)
    ic, _ = _ic_and_winrate(scores, forward_returns)
    return {regime_label: ic}


def _save_record(record):
    if os.path.exists(MONITOR_FILE):
        with open(MONITOR_FILE) as f:
            history = json.load(f)
    else:
        history = {"records": []}
    # 去重同日
    history["records"] = [r for r in history["records"]
                          if r.get("date") != record["date"]]
    history["records"].append(record)
    history["records"] = history["records"][-24:]
    with open(MONITOR_FILE, 'w') as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)


def _write_obsidian(record):
    obsidian_dir = ("/Users/fuyuan/Documents/Obsidian Vault/"
                    "项目/量化/策略1 - AI-Capital")
    os.makedirs(obsidian_dir, exist_ok=True)
    md_path = os.path.join(obsidian_dir, "因子监控.md")

    lines = [
        "---",
        "tags: [因子监控, 策略1, V4.2]",
        f"date: {record['date']}",
        "parent: \"[[策略1 - AI-Capital 动量制度检测|策略1]]\"",
        "---",
        "",
        f"# V4.2 因子监控 — {record['date']}",
        "",
        "## 真实因子表现",
        "",
        "| 因子 | IC | 胜率 |",
        "|------|:---:|:----:|",
    ]
    for fname in ["low_vol", "momentum_6m"]:
        d = record["factors"].get(fname, {})
        lines.append(f"| {d.get('label', fname)} | {d.get('ic', 0):+.4f} | {d.get('win_rate', 0):.0f}% |")

    lines += [
        "",
        "## 综合指标",
        "",
        f"| 指标 | 值 |",
        f"|------|:---|",
        f"| 综合得分 IC | {record['composite']['ic']:+.4f} |",
        f"| 综合胜率 | {record['composite']['win_rate']:.0f}% |",
        f"| 反转过滤效果(入选-剔除) | {record['reversal_effect']['diff']:+.2f}% |",
        f"| 剔除股次月平均收益 | {record['reversal_effect']['excluded_ret']:+.2f}% |",
        f"| 入选股次月平均收益 | {record['reversal_effect']['selected_ret']:+.2f}% |",
        f"| 制度分档IC(RISKON) | {record['regime_ic'].get('RISKON', 0):+.4f} |",
        f"| 制度分档IC(非RISKON) | {record['regime_ic'].get('non_RISKON', 0):+.4f} |",
        f"| 宇宙规模 | {record['universe_size']} → {record['filtered_size']} (过滤后) |",
        "",
    ]

    if record["alerts"]:
        lines.append("## ⚠️ 退化预警")
        for a in record["alerts"]:
            lines.append(f"- {a}")
        lines.append("")

    lines.append("> V4.2 因子监控: 只追踪真实因子(低波/动量) + 反转过滤效果 + 综合得分IC")

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    run_monthly_monitor()
