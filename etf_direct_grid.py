#!/usr/bin/env python3
"""
ETF 直接因子网格搜索: Quality × Amihud × Reversal 全部组合
在 V2 两档+SURGE28d 框架下搜索最优权重。
对比: 股票投票版网格搜索结果。
"""
import os, sys, json, warnings, time; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/etf-r1"))

from config import START_DATE
from etf_r1_backtest import build_surge
from run_final_backtest import compute_regime
from run_v5_backtest import month_end_dates
from src.stock_backtest import run_stock_backtest

try:
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark
except ModuleNotFoundError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qmt_adapter", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "qmt_adapter.py"))
    qma = importlib.util.module_from_spec(spec); spec.loader.exec_module(qma)
    load_qmt_etfs = qma.load_qmt_etfs; load_qmt_benchmark = qma.load_qmt_benchmark

BROAD = ["sh510300","sh510500","sz159915","sh588000"]
DEFENSE = ["sh511010","sh518880","sh511880"]
SECTOR_MAP = {
    "sh512880":["证券","资本市场","货币金融"],"sh512800":["货币金融","金融服务"],
    "sh512010":["医药","制药","医疗","中药","生物制品"],
    "sz159928":["食品","饮料","酒","农牧","渔业","养殖","零售","批发","农副"],
    "sh512690":["酒","饮料"],"sh512480":["电子","计算机","通信"],
    "sz159995":["电子","计算机","通信"],
    "sh512660":["航天","航空","船舶","航海","兵装","军工","国防","卫星"],
    "sh515030":["汽车","新能源"],"sh515790":["光伏","电气机械","电源"],
    "sh512400":["金属","有色","黄金","贵金属","黑色金属","采掘","煤炭","石油","矿业","矿物"],
    "sh512200":["房地产","建筑","土木","建材"],
    "sh512980":["传媒","游戏","影视","广告","出版","文化","旅游","教育","广播"],
    "sz159939":["软件","互联网","通信","计算机","IT","信息服务","专业技术"],
    "sh515210":["黑色金属","冶","金属"],
    "sz159870":["化学","化纤","塑料","橡胶","造纸","纺织"],
    "sz159967":["设备","制造","机械","仪器","装备","高端","智能","自动化","机器人"],
    "sz159611":["电力","热力","燃气","水的生产","公用","发电","供电","电网"],
    "sh512580":["环保","环境","生态","水利","节能","清洁","循环","废旧"],
    "sh516670":["运输","交通","物流","铁路","道路","航运","水上","港口","快递","高速","机场"],
    "sh512070":["保险","非银","其他金融"],
    "sz159865":["农业","农牧","林业","畜牧","养殖","种业","种植","饲料"],
    "sh516750":["非金属矿物","水泥","玻璃","陶瓷"],
    "sh512760":["芯片","半导体","集成电路"],"sh515050":["5G","通信设备","信息技术"],
    "sh516160":["新能源","碳中和","清洁能源"],
}
ALL_ETF_CODES = BROAD + DEFENSE + list(SECTOR_MAP.keys())

# ── ETF 级因子 ──────────────────────────────────────────
def compute_etf_factors(etf_data, date):
    """返回原始因子值 {code: {amihud, reversal, quality}}"""
    results = {}
    for code, df in etf_data.items():
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 250:
            continue
        hist = df.iloc[:idx+1]
        closes = hist['close']
        f = {}

        # Quality: 250d return (positive only)
        if len(closes) >= 250:
            f['quality'] = max(0, closes.iloc[-1] / closes.iloc[-250] - 1)
        else:
            f['quality'] = 0

        # Amihud
        sub_63 = hist.iloc[-63:]
        ret_abs = sub_63['close'].pct_change().dropna().abs()
        amt = sub_63['amount'].iloc[-len(ret_abs):]
        valid = amt > 0
        if valid.sum() >= 10:
            f['amihud'] = (ret_abs[valid] / amt[valid]).mean()
        else:
            f['amihud'] = 0

        # Reversal (-Mom×R²)
        sub_34 = hist.iloc[-35:-1]
        if len(sub_34) >= 10:
            y = np.log(sub_34['close'].values); x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            ar = np.exp(slope * 250) - 1
            y_pred = slope * x + intercept
            ss_r = np.sum((y - y_pred)**2)
            ss_t = np.sum((y - y.mean())**2)
            r2 = 1 - ss_r/ss_t if ss_t > 0 else 0
            f['reversal'] = -(ar * r2)
        else:
            f['reversal'] = 0

        results[code] = f
    return results


def combine_scores(raw_factors, w_a, w_r, w_q):
    """加权合并"""
    scored = {}
    for code, f in raw_factors.items():
        total = 0
        if w_q > 0 and f.get('quality', 0) > 0:
            total += np.log1p(f['quality']) * w_q
        if w_a > 0 and f.get('amihud', 0) > 0:
            total += np.log1p(f['amihud'] * 1e10) * w_a
        if w_r > 0:
            total += f.get('reversal', 0) * w_r
        scored[code] = total
    return scored


def select_etfs(scores, lo, date, is_surge, top_n=4):
    candidates = {c: s for c, s in scores.items() if c in SECTOR_MAP and lo(c)}
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    if is_surge:
        picks = [c for c, _ in ranked[:2]]
    else:
        picks = [c for c, _ in ranked[:2]]
        for g in ["sz159915","sh588000"]:
            if lo(g) and g not in picks: picks.append(g)
        for c, _ in ranked:
            if len(picks) >= top_n: break
            if c not in picks and lo(c): picks.append(c)
    if len(picks) < 2:
        for b in BROAD:
            if lo(b) and b not in picks: picks.append(b)
    return picks[:top_n]


# ── 数据加载 ─────────────────────────────────────────────
print("加载数据...")
etf_data_raw = load_qmt_etfs(ALL_ETF_CODES, verbose=False)
closes, opens = {}, {}
for sym in ALL_ETF_CODES:
    if sym in etf_data_raw:
        closes[sym] = etf_data_raw[sym]["close"]; opens[sym] = etf_data_raw[sym]["open"]
dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
ends = [dfc[s].last_valid_index() for s in BROAD+DEFENSE if s in dfc.columns]
if ends: dfc = dfc.loc[:min(ends)].ffill(limit=5)
dfo = pd.DataFrame(opens).sort_index().reindex(dfc.index)
print(f"  ETF panel: {len(dfc.columns)} 只, {len(dfc)} 天")

idx = load_qmt_benchmark("000300.SH")
rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
lc = dfc.notna().cumsum()

# ── 网格搜索 ────────────────────────────────────────────
COMBOS = [
    # 单因子
    (100, 0, 0, "Q100"), (0, 0, 100, "A100"), (0, 100, 0, "R100"),
    # Q 主导
    (30, 0, 70, "Q30-A70"), (50, 0, 50, "Q50-A50"), (70, 0, 30, "Q70-A30"),
    # 三项混合
    (50, 25, 25, "Q50-A25-R25"), (40, 30, 30, "Q40-A30-R30"),
    (60, 20, 20, "Q60-A20-R20"), (33, 33, 34, "Q33-A33-R34"),
    # Q + R
    (70, 30, 0, "Q70-R30"), (50, 50, 0, "Q50-R50"),
    # A 主导
    (10, 0, 90, "Q10-A90"), (20, 0, 80, "Q20-A80"),
    # R 主导
    (10, 90, 0, "Q10-R90"), (30, 70, 0, "Q30-R70"),
]

results = []
t_start = time.time()

for w_q, w_r, w_a, label in COMBOS:
    t0 = time.time()
    print(f"  {label} (Q={w_q} R={w_r} A={w_a}) ...", end=" ", flush=True)

    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=28)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
    rs_mod = rs_raw.copy()
    for d in rs_mod.index: rs_mod.loc[d] = 0.85 if rs_mod.loc[d] >= 0.70 else 0.20

    cache = {}
    def sel(date):
        if date in cache: return cache[date]
        raw = compute_etf_factors(etf_data_raw, date)
        scores = combine_scores(raw, w_a/100., w_r/100., w_q/100.)
        lo = lambda s: (lc.at[date, s] if s in dfc.columns else 0) >= 20
        is_s = d2t.get(date) is not None
        picks = select_etfs(scores, lo, date, is_s)
        cache[date] = picks
        return picks

    rr = run_stock_backtest(
        dfc, rs_mod, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=sel, forced_regime=forced)

    v = rr["values"]["value"]; rt = v.pct_change().dropna()
    yrs = len(rt)/244; ann = (v.iloc[-1]/1e6)**(1/yrs)-1
    dd = (v/v.cummax()-1).min(); sh = rt.mean()/rt.std()*np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]

    elapsed = time.time() - t0
    print(f"年化 {ann*100:.2f}% 回撤 {dd*100:.1f}% 夏普 {sh:.2f} 换手 {t:.1f}x [{elapsed:.0f}s]")

    results.append({"label": label, "w_q": w_q, "w_r": w_r, "w_a": w_a,
                    "ann": round(ann*100,2), "mdd": round(dd*100,1),
                    "sharpe": round(sh,2), "turnover": round(t,1)})

# ── 排名 ─────────────────────────────────────────────────
print(f"\n{'='*80}")
print("ETF 直接因子网格搜索结果（按年化排序）")
print(f"{'='*80}")
print(f"  {'组合':<18} {'Q%':>4} {'R%':>4} {'A%':>4} {'年化':>7} {'回撤':>7} {'夏普':>6} {'换手':>6}")
print(f"  {'-'*60}")
for r in sorted(results, key=lambda x: x['ann'], reverse=True):
    print(f"  {r['label']:<18} {r['w_q']:>4} {r['w_r']:>4} {r['w_a']:>4} "
          f"{r['ann']:>6.2f}% {r['mdd']:>6.1f}% {r['sharpe']:>5.2f} {r['turnover']:>5.1f}x")

# 对比股票投票最佳
print(f"\n  {'─'*60}")
print(f"  对照: 股票投票 Q100          {'':>8}  15.38%  -23.7%  0.88   6.1x")
print(f"  对照: 股票投票 现行因子      {'':>8}  14.89%  -23.5%  0.87   6.7x")

total_t = (time.time() - t_start) / 60
print(f"\n总耗时: {total_t:.1f}min")

out = os.path.join(os.path.dirname(__file__), "backtests", "etf_direct_grid.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"[saved] {out}")
