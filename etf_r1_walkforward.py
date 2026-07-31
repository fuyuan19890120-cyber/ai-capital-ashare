#!/usr/bin/env python3
"""
ETF-R1 Walk-Forward 验证 + 相关性过滤
IS 2015-2021: 网格搜索 lock_days × 框架
OOS 2022-2026: 最优参数样本外评估
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
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark, load_qmt_stocks
except ModuleNotFoundError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qmt_adapter", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "qmt_adapter.py"))
    qma = importlib.util.module_from_spec(spec); spec.loader.exec_module(qma)
    load_qmt_etfs = qma.load_qmt_etfs; load_qmt_benchmark = qma.load_qmt_benchmark; load_qmt_stocks = qma.load_qmt_stocks

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
CORR_THRESHOLD = 0.7
CORR_WINDOW = 60

# ── SURGE 原始版 ─────────────────────────────────────
def build_surge_orig(index_close, etf_close, regime_series, cal, rebal, lock_days=21):
    sma50 = index_close.rolling(50).mean(); sma30 = index_close.rolling(30).mean()
    dev30 = (index_close - sma30) / sma30
    s30 = 0.6 * (0.5 + 0.5 * np.tanh(dev30 * 10)) + 0.4 * (sma50 > sma30).astype(float)
    breadth = pd.DataFrame({e: (etf_close[e] > etf_close[e].rolling(50).mean()).astype(float)
                            for e in ["sh510300","sh510500","sz159915"]}).mean(axis=1)
    s30, breadth = s30.reindex(cal), breadth.reindex(cal)
    base = regime_series.reindex(cal)
    trig = (base >= 0.15) & (s30 >= 0.70) & (breadth >= 2/3)
    rebal_set = set(rebal)
    extra, forced, d2trig = [], {}, {}
    lock_until, last_regime, cur_trig = -1, 'NEUTRAL', None
    def reg(s):
        return 'RISKON' if s >= .70 else 'NEUTRAL' if s >= .50 else 'RISKOFF' if s >= .30 else 'CRISIS'
    for i, d in enumerate(cal):
        if i <= lock_until:
            d2trig[d] = cur_trig
            if d in rebal_set: forced[d] = 'RISKON'; last_regime = 'RISKON'
            continue
        if d in rebal_set and d in regime_series.index:
            last_regime = reg(float(regime_series.loc[d]))
        if bool(trig.get(d, False)):
            lock_until = i + lock_days - 1; cur_trig = d; d2trig[d] = d; forced[d] = 'RISKON'
            if last_regime != 'RISKON' and d not in rebal_set: extra.append(d)
            last_regime = 'RISKON'
    return pd.DatetimeIndex(extra), forced, d2trig

# ── Q100 ─────────────────────────────────────────────
def compute_q100(stock_data, date):
    scores = {}
    for code, df in stock_data.items():
        if date not in df.index: continue
        idx = df.index.get_loc(date)
        if idx < 250: continue
        lr = df['close'].iloc[idx] / df['close'].iloc[idx-250] - 1
        scores[code] = np.log1p(lr) if lr > 0 else 0
    return scores

def select_top_stocks(scores, top_n=30):
    return [c for c,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_n]]

# ── 数据加载 ─────────────────────────────────────────
print("加载数据...")
def load_panels():
    etf_data = load_qmt_etfs(ALL_ETF_CODES, verbose=False)
    closes, opens = {}, {}
    for sym in ALL_ETF_CODES:
        if sym in etf_data: closes[sym] = etf_data[sym]["close"]; opens[sym] = etf_data[sym]["open"]
    dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
    ends = [dfc[s].last_valid_index() for s in BROAD+DEFENSE if s in dfc.columns]
    if ends: dfc = dfc.loc[:min(ends)].ffill(limit=5)
    dfo = pd.DataFrame(opens).sort_index().reindex(dfc.index)
    return dfc, dfo

dfc, dfo = load_panels()
idx = load_qmt_benchmark("000300.SH")
rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me_full = month_end_dates(cal)
lc = dfc.notna().cumsum(); lo = lambda s,d: (lc.at[d,s] if s in dfc.columns else 0) >= 20

ind_map = {}
mf = pd.read_csv(os.path.expanduser("~/ai-capital-ashare/data/industry_map_bs.csv"), dtype=str)
mf = mf[mf["industryClassification"]=="证监会行业分类"]; mf["bare"] = mf["code"].str[3:]
for _,r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"]!="": ind_map[r["bare"]] = str(r["industry"])
sd = load_qmt_stocks(min_history=250)
print(f"  ETF: {len(dfc.columns)}只, 个股: {len(sd)}只")

# ── 带相关性过滤的选股函数 ───────────────────────────
def make_sel_with_corr(enable_filter=True):
    """选股 + 可选相关性过滤"""
    cache = {}
    def sel(date, d2t):
        if (date, id(d2t)) in cache: return cache[(date, id(d2t))]
        valid = {}
        for c,s in sd.items():
            if date in s.index and len(s[s.index <= date]) >= 250: valid[c] = s
        if not valid:
            r = [s for s in BROAD if lo(s,date)]; cache[(date,id(d2t))] = r; return r
        scores = compute_q100(valid,date); top30 = select_top_stocks(scores,30)
        votes = {}
        for code in top30[:30]:
            ind = ind_map.get(code,"")
            for etf,kws in SECTOR_MAP.items():
                if any(k in ind for k in kws): votes[etf] = votes.get(etf,0)+1; break
        ranked = sorted(votes.items(), key=lambda x:x[1], reverse=True)

        # Top-2 行业 ETF，带相关性过滤
        is_s = d2t.get(date) is not None
        top = []
        for e, _ in ranked:
            if not lo(e, date): continue
            if enable_filter and len(top) >= 1:
                # 检查与已选 ETF 的相关性
                prev = top[0]
                if prev in dfc.columns and e in dfc.columns:
                    common = dfc[[prev, e]].dropna().index
                    if len(common) > 0:
                        recent = common[-CORR_WINDOW:]
                        if len(recent) >= 20:
                            corr = dfc.loc[recent, prev].corr(dfc.loc[recent, e])
                            if corr > CORR_THRESHOLD:
                                continue  # 跳过高度相关的
            top.append(e)
            if len(top) >= 2: break

        gb = [s for s in ["sz159915","sh588000"] if lo(s,date)]
        while len(top) < 2 and gb: top.append(gb.pop(0))
        if len(top) < 2: top = [s for s in BROAD if lo(s,date)][:2]

        if is_s: r = top[:2]
        else:
            r = top[:2]
            for g in ["sz159915","sh588000"]:
                if lo(g,date) and g not in r: r.append(g)
            while len(r) < 4:
                for e,_ in ranked:
                    if e not in r and lo(e,date): r.append(e); break
                else: break
        cache[(date,id(d2t))] = r[:4]; return r[:4]
    return sel

def stats(rr):
    v = rr["values"]["value"]; rt = v.pct_change().dropna()
    yrs = len(rt)/244; ann = (v.iloc[-1]/1e6)**(1/yrs)-1
    dd = (v/v.cummax()-1).min(); sh = rt.mean()/rt.std()*np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]
    return {"ann":round(ann*100,2),"mdd":round(dd*100,1),"sharpe":round(sh,2),"turnover":round(t,1)}

def run_with_params(dfc, dfo, rs_raw, cal, me, lock_days, framework, enable_corr):
    """framework: 'V2' | 'S4' (四档)"""
    extra, forced, d2t = build_surge_orig(idx["close"], dfc, rs_raw, cal, me, lock_days=lock_days)
    rd = pd.DatetimeIndex(sorted(set(me)|set(extra)))
    sf = make_sel_with_corr(enable_corr)

    if framework == 'V2':
        rs = rs_raw.copy()
        for d in rs.index: rs.loc[d] = 0.85 if rs.loc[d] >= 0.70 else 0.20
    else:
        rs = rs_raw

    return run_stock_backtest(
        dfc, rs, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)

# ── IS 分割 ──────────────────────────────────────────
is_end = "2021-12-31"
oos_start = "2022-01-01"

# IS 日期范围
cal_is = cal[cal <= is_end]
me_is = [d for d in me_full if d <= pd.Timestamp(is_end)]

# OOS 日期范围
cal_oos = cal[cal >= oos_start]
me_oos = [d for d in me_full if d >= pd.Timestamp(oos_start)]

print(f"\n{'='*70}")
print(f"Walk-Forward 验证: IS 2015-2021 | OOS 2022-2026")
print(f"{'='*70}")

# ── Phase 1: IS 网格搜索 ─────────────────────────────
print(f"\n[Phase 1] IS 网格搜索 (2015-2021)")
print(f"{'='*50}")

lock_options = [21, 28, 42]
framework_options = ['V2', 'S4']
corr_options = [False, True]

is_results = []
for lock in lock_options:
    for fw in framework_options:
        for corr in corr_options:
            label = f"{fw}-lock{lock}{'-corr' if corr else ''}"
            print(f"  {label} ...", end=" ", flush=True)
            rr = run_with_params(dfc, dfo, rs_raw, cal_is, me_is, lock, fw, corr)
            s = stats(rr)
            is_results.append({"label": label, "lock": lock, "fw": fw, "corr": corr,
                               "ann": s['ann'], "mdd": s['mdd'], "sharpe": s['sharpe']})
            print(f"年化 {s['ann']}% 回撤 {s['mdd']}% 夏普 {s['sharpe']}")

# IS 排名
print(f"\n  IS 排名:")
is_sorted = sorted(is_results, key=lambda x: x['ann'], reverse=True)
for i, r in enumerate(is_sorted[:6]):
    print(f"    {i+1}. {r['label']:<20} 年化 {r['ann']:>6.2f}%  回撤 {r['mdd']:>6.1f}%  夏普 {r['sharpe']:.2f}")

best = is_sorted[0]
print(f"\n  最优 IS 参数: {best['label']}")

# ── Phase 2: OOS 评估 ────────────────────────────────
print(f"\n[Phase 2] OOS 评估 (2022-2026, 固定最优IS参数)")
print(f"{'='*50}")

# 用最优 IS 参数跑 OOS
rr_oos = run_with_params(dfc, dfo, rs_raw, cal_oos, me_oos, best['lock'], best['fw'], best['corr'])
s_oos = stats(rr_oos)
print(f"  {best['label']} OOS: 年化 {s_oos['ann']}%  回撤 {s_oos['mdd']}%  夏普 {s_oos['sharpe']}")

# 全样本（用最优参数）
rr_full = run_with_params(dfc, dfo, rs_raw, cal, me_full, best['lock'], best['fw'], best['corr'])
s_full = stats(rr_full)
v = rr_full["values"]["value"]; rt_full = v.pct_change().dropna()
# 逐年
yearly_full = {}
for yr in sorted(set(rt_full.index.year)):
    s_yr = rt_full[rt_full.index.year == yr]
    if len(s_yr) > 10:
        yearly_full[yr] = round(((1+s_yr).prod()-1)*100, 1)

# IS 逐年
v_is = rr_full["values"]["value"]
# 分割 IS/OOS 从全样本价值曲线中提取
# 简化：从 IS 回测结果中取
rr_is = run_with_params(dfc, dfo, rs_raw, cal_is, me_is, best['lock'], best['fw'], best['corr'])
v_is_val = rr_is["values"]["value"]; rt_is = v_is_val.pct_change().dropna()
yearly_is = {}
for yr in sorted(set(rt_is.index.year)):
    s_yr = rt_is[rt_is.index.year == yr]
    if len(s_yr) > 10:
        yearly_is[yr] = round(((1+s_yr).prod()-1)*100, 1)

# OOS 逐年
rt_oos = v.pct_change().dropna(); rt_oos = rt_oos[rt_oos.index >= oos_start]
yearly_oos = {}
for yr in sorted(set(rt_oos.index.year)):
    s_yr = rt_oos[rt_oos.index.year == yr]
    if len(s_yr) > 10:
        yearly_oos[yr] = round(((1+s_yr).prod()-1)*100, 1)

# ── 最终汇总 ─────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Walk-Forward 最终结果")
print(f"{'='*70}")
print(f"\n  最优参数: {best['label']}")
print(f"  锁定 lock_days={best['lock']}, framework={best['fw']}, corr_filter={best['corr']}")
print(f"\n  IS (2015-2021): 年化 {is_sorted[0]['ann']}%  回撤 {is_sorted[0]['mdd']}%  夏普 {is_sorted[0]['sharpe']}")
print(f"  OOS (2022-2026): 年化 {s_oos['ann']}%  回撤 {s_oos['mdd']}%  夏普 {s_oos['sharpe']}")
print(f"  全样本 (2015-2026): 年化 {s_full['ann']}%  回撤 {s_full['mdd']}%  夏普 {s_full['sharpe']}")

# OOS/IS 比率（Walk-Forward Efficiency）
if is_sorted[0]['ann'] > 0:
    wfe = s_oos['ann'] / is_sorted[0]['ann']
    print(f"\n  Walk-Forward Efficiency (OOS/IS): {wfe:.2f}  {'✅ >1' if wfe > 1 else '✅ >0.8' if wfe > 0.8 else '⚠️ <0.8'}")

print(f"\n  逐年:")
print(f"  {'年':<6} {'IS':>8} {'OOS':>8}")
for yr in sorted(set(list(yearly_is.keys()) + list(yearly_oos.keys()))):
    is_v = yearly_is.get(yr, None)
    oos_v = yearly_oos.get(yr, None)
    print(f"  {yr:<6} {f'{is_v}%':>8}" if is_v else f"  {yr:<6} {'':>8}", end="")
    if oos_v: print(f" {f'{oos_v}%':>8}")
    else: print()

# IS 网格搜索完整结果
print(f"\n  IS 完整结果:")
print(f"  {'参数':<22} {'年化':>7} {'回撤':>7} {'夏普':>6}")
for r in sorted(is_results, key=lambda x: x['ann'], reverse=True):
    print(f"  {r['label']:<22} {r['ann']:>6.2f}% {r['mdd']:>6.1f}% {r['sharpe']:>5.2f}")
