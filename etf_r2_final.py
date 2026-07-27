#!/usr/bin/env python3
"""
ETF-R2 最终定稿
选股: Mom×R² (30日对数回归动量×R²) 在时间点池上计算
择时: V2/四档双框架 + SURGE (breadth≥2/3, lock=14d, 8%熔断)
权重: 四档=行业70%+成长30%, V2=等权50/50
数据: qmt_qfq.db (QMT前复权)

回测(2015-2026, 严格验证):
  四档+14d: 年化16.21% / 回撤-24.8% / 夏普0.87
  V2+14d:   年化15.02% / 回撤-23.9% / 夏普0.87
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/etf-r1"))

from config import START_DATE
from run_final_backtest import compute_regime
from run_v5_backtest import month_end_dates
from src.stock_backtest import run_stock_backtest

try:
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark
except ModuleNotFoundError:
    import importlib.util
    s = importlib.util.spec_from_file_location("qa", "src/qmt_adapter.py")
    qa = importlib.util.module_from_spec(s); s.loader.exec_module(qa)
    load_qmt_etfs = qa.load_qmt_etfs; load_qmt_benchmark = qa.load_qmt_benchmark

# ── ETF 池 ──────────────────────────────────────────────
BROAD  = ["sh510300", "sh510500", "sz159915", "sh588000"]
DEFENSE = ["sh511010", "sh518880", "sh511880"]

# 37只精选行业ETF（每类选流动性最好的1-2只）
SECTOR_ETFS = [
    # 金融
    "sh512880", "sh512800", "sh512070", "sh512000",
    # 医药
    "sh512010", "sh512170", "sz159992",
    # 消费
    "sz159928", "sh512690",
    # 半导体/芯片
    "sh512480", "sz159995", "sh512760",
    # 军工
    "sh512660", "sh512710",
    # 新能源
    "sh515030", "sh515790", "sh516160",
    # 周期/资源
    "sh512400", "sh515210", "sz159870",
    # 房地产/建筑
    "sh512200", "sh516750",
    # 传媒/游戏
    "sh512980", "sz159869",
    # 科技/IT/云计算
    "sz159939", "sh515000", "sh512720",
    # 通信/5G
    "sh515050", "sh515880",
    # 装备/机器人
    "sz159967", "sh562500",
    # AI
    "sz159819",
    # 电力/环保/交通/农业
    "sz159611", "sh512580", "sh516670", "sz159865",
]

ALL_ETFS = BROAD + DEFENSE + SECTOR_ETFS
CANDIDATES = SECTOR_ETFS + ["sz159915", "sh588000"]  # 可选池=行业+成长

# ── SURGE 原始设计 ─────────────────────────────────────
def build_surge(index_close, etf_close, regime_series, cal, rebal, lock_days=21):
    sma50 = index_close.rolling(50).mean()
    sma30 = index_close.rolling(30).mean()
    dev30 = (index_close - sma30) / sma30
    s30 = 0.6 * (0.5 + 0.5 * np.tanh(dev30 * 10)) + 0.4 * (sma50 > sma30).astype(float)
    breadth = pd.DataFrame({
        e: (etf_close[e] > etf_close[e].rolling(50).mean()).astype(float)
        for e in ["sh510300", "sh510500", "sz159915"]
    }).mean(axis=1)
    s30, breadth = s30.reindex(cal), breadth.reindex(cal)
    base = regime_series.reindex(cal)
    trig = (base >= 0.15) & (s30 >= 0.70) & (breadth >= 2/3)

    rebal_set = set(rebal)
    extra, forced, d2trig = [], {}, {}
    lock_until, last_regime, cur_trig = -1, 'NEUTRAL', None

    def reg(s):
        if s >= .70: return 'RISKON'
        if s >= .50: return 'NEUTRAL'
        if s >= .30: return 'RISKOFF'
        return 'CRISIS'

    for i, d in enumerate(cal):
        if i <= lock_until:
            d2trig[d] = cur_trig
            if d in rebal_set:
                forced[d] = 'RISKON'; last_regime = 'RISKON'
            continue
        if d in rebal_set and d in regime_series.index:
            last_regime = reg(float(regime_series.loc[d]))
        if bool(trig.get(d, False)):
            lock_until = i + lock_days - 1
            cur_trig = d
            d2trig[d] = d
            forced[d] = 'RISKON'
            if last_regime != 'RISKON' and d not in rebal_set:
                extra.append(d)
            last_regime = 'RISKON'

    return pd.DatetimeIndex(extra), forced, d2trig

# ── 因子: Mom×R² (30日) ─────────────────────────────────
def compute_mom_r2(date, dfc, lo, window=30):
    scores = {}
    for code in CANDIDATES:
        if code not in dfc.columns or not lo(code, date):
            continue
        idx = dfc.index.get_loc(date)
        if idx < window + 1:
            continue
        y = np.log(dfc[code].iloc[idx - window:idx].values)
        x = np.arange(len(y))
        if len(y) < 10:
            continue
        slope = np.polyfit(x, y, 1)[0]
        annual_ret = np.exp(slope * 250) - 1
        y_pred = slope * x + np.polyfit(x, y, 1)[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        scores[code] = annual_ret * r2
    return scores

# ── 选股 ────────────────────────────────────────────────
def make_selector(dfc, lo, sector_pct=50):
    """sector_pct: 行业ETF占比 (50=等权/70=七成). SURGE始终50/50."""
    cache = {}

    def select(date, d2t):
        if date in cache:
            return cache[date]
        scores = compute_mom_r2(date, dfc, lo)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        is_surge = d2t.get(date) is not None
        top_s = [e for e, _ in ranked[:2] if lo(e, date)]
        if len(top_s) < 2:
            top_s = [s for s in BROAD if lo(s, date)][:2]

        if is_surge:
            # SURGE: 纯2行业, 各50%
            picks = [(top_s[0], 0.5), (top_s[1], 0.5)]
        else:
            sw = sector_pct / 200  # 每只行业权重
            gp = (100 - sector_pct) / 200  # 每只成长权重
            picks = [(top_s[0], sw), (top_s[1], sw)]
            for g in ["sz159915", "sh588000"]:
                if lo(g, date) and g not in [c for c, _ in picks]:
                    picks.append((g, gp))

        if len(picks) < 2:
            picks = [(s, 1.0/len(BROAD)) for s in BROAD if lo(s, date)][:4]

        cache[date] = picks[:4]
        return picks[:4]

    return select

# ── 主流程 ──────────────────────────────────────────────
def run_one(name, rs, lock_days, sector_pct=50):
    """运行一次回测, 返回 stats dict"""
    extra, forced, d2t = build_surge(idx_close["close"], dfc, rs_orig, cal, me, lock_days=lock_days)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
    selector = make_selector(dfc, lo, sector_pct=sector_pct)

    result = run_stock_backtest(
        dfc, rs, {}, top_n=4, verbose=False,
        execution="next_open", stamp_duty=True, ffill_valuation=True,
        df_open=None, rebalance_dates=rd,
        select_fn=lambda d: selector(d, d2t),
        forced_regime=forced
    )

    v = result["values"]["value"]
    rt = v.pct_change().dropna()
    yrs = len(rt) / 244
    ann = (v.iloc[-1] / 1e6) ** (1 / yrs) - 1
    dd = (v / v.cummax() - 1).min()
    sh = rt.mean() / rt.std() * np.sqrt(244)
    turnover = result["metrics"]["annual_turnover_x"]
    yearly = rt.groupby(rt.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    surge_count = len(set(d2t.values())) - 1

    return {
        "name": name, "ann": round(ann * 100, 2), "mdd": round(dd * 100, 1),
        "sharpe": round(sh, 2), "turnover": round(turnover, 1), "surge": surge_count,
        "yearly": {int(k): round(v, 1) for k, v in yearly.items()}
    }

def main():
    print("=" * 55)
    print("ETF-R2 最终定稿: Mom×R² + V2/四档 双框架")
    print("=" * 55)

    # 加载数据（只做一次）
    print("加载 ETF 数据...")
    etf_data = load_qmt_etfs(ALL_ETFS, verbose=False)
    closes = {}
    for s in ALL_ETFS:
        if s in etf_data:
            closes[s] = etf_data[s]["close"]
    global dfc, dfo, idx_close, rs_orig, cal, me, lc, lo
    dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
    ends = [dfc[s].last_valid_index() for s in BROAD + DEFENSE if s in dfc.columns]
    if ends:
        dfc = dfc.loc[:min(ends)].ffill(limit=5)

    idx_close = load_qmt_benchmark("000300.SH")
    rs_orig = compute_regime(idx_close["close"])
    rs_orig = rs_orig[rs_orig.index >= START_DATE]

    cal = dfc.index[dfc.index >= START_DATE]
    me = month_end_dates(cal)
    lc = dfc.notna().cumsum()
    lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

    # V2 两档
    rs_v2 = rs_orig.copy()
    for d in rs_v2.index:
        rs_v2.loc[d] = 0.85 if rs_v2.loc[d] >= 0.70 else 0.20

    # 运行两个框架
    print("运行回测...")
    s_v2 = run_one("V2两档+14d(50%行业)", rs_v2, lock_days=14, sector_pct=50)
    s_s4 = run_one("四档+14d(70%行业)", rs_orig, lock_days=14, sector_pct=70)

    # 汇总
    all_years = sorted(set(list(s_v2["yearly"].keys()) + list(s_s4["yearly"].keys())))
    print(f"\n{'='*55}")
    print("ETF-R2 最终结果")
    print(f"{'='*55}")
    print(f"  {'':<20} {'V2两档+14d(50%)':>22} {'四档+14d(70%)':>22}")
    print(f"  {'年化':<20} {s_v2['ann']:>17.2f}% {s_s4['ann']:>17.2f}%")
    print(f"  {'最大回撤':<20} {s_v2['mdd']:>17.1f}% {s_s4['mdd']:>17.1f}%")
    print(f"  {'夏普':<20} {s_v2['sharpe']:>18.2f} {s_s4['sharpe']:>18.2f}")
    print(f"  {'SURGE次数':<20} {s_v2['surge']:>18} {s_s4['surge']:>18}")

    print(f"\n逐年收益:")
    print(f"  {'年':<6} {'V2两档':>10} {'四档':>10}")
    for yr in all_years:
        if yr >= 2016:
            print(f"  {yr:<6} {s_v2['yearly'].get(yr,0):>+9.1f}% {s_s4['yearly'].get(yr,0):>+9.1f}%")

    print(f"\n  ETF池: {len(SECTOR_ETFS)}只行业 + {len(BROAD)}宽基 + {len(DEFENSE)}防守")
    print(f"  因子:  Mom×R² (30日窗口)")
    print(f"  数据:  qmt_qfq.db (QMT前复权)")

    # 保存
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtests", "etf_r2_final.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "strategy": "ETF-R2", "factor": "Mom×R² 30d",
            "V2": {"ann": s_v2["ann"], "mdd": s_v2["mdd"], "sharpe": s_v2["sharpe"],
                   "turnover": s_v2["turnover"], "surge": s_v2["surge"], "yearly": s_v2["yearly"]},
            "S4": {"ann": s_s4["ann"], "mdd": s_s4["mdd"], "sharpe": s_s4["sharpe"],
                   "turnover": s_s4["turnover"], "surge": s_s4["surge"], "yearly": s_s4["yearly"]}
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  [saved] {out}")

if __name__ == "__main__":
    main()
