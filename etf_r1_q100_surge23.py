#!/usr/bin/env python3
"""
ETF-R1 Q100 最终版: SURGE 原始设计 — breadth ≥ 2/3 即触发（去掉15日跳跃条件）
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

# ── SURGE 原始版: breadth ≥ 2/3 即触发 ───────────────
def build_surge_orig(index_close, etf_close, regime_series, cal, rebal, lock_days=21):
    """原始 SURGE 设计: 三宽基中 ≥2 只站上 SMA50 即触发。无 15 日跳跃条件。"""
    sma50 = index_close.rolling(50).mean(); sma30 = index_close.rolling(30).mean()
    dev30 = (index_close - sma30) / sma30
    s30 = 0.6 * (0.5 + 0.5 * np.tanh(dev30 * 10)) + 0.4 * (sma50 > sma30).astype(float)
    breadth = pd.DataFrame({e: (etf_close[e] > etf_close[e].rolling(50).mean()).astype(float)
                            for e in ["sh510300","sh510500","sz159915"]}).mean(axis=1)
    s30, breadth = s30.reindex(cal), breadth.reindex(cal)
    base = regime_series.reindex(cal)
    # 原始触发: base≥0.15, s30≥0.70, breadth≥2/3 (即 2 或 3 只)
    trig = (base >= 0.15) & (s30 >= 0.70) & (breadth >= 2/3)

    rebal_set = set(rebal)
    extra, forced, d2trig = [], {}, {}
    lock_until, last_regime, cur_trig = -1, 'NEUTRAL', None
    def reg(s):
        return 'RISKON' if s >= .70 else 'NEUTRAL' if s >= .50 else 'RISKOFF' if s >= .30 else 'CRISIS'
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
            cur_trig = d; d2trig[d] = d; forced[d] = 'RISKON'
            if last_regime != 'RISKON' and d not in rebal_set:
                extra.append(d)
            last_regime = 'RISKON'
    return pd.DatetimeIndex(extra), forced, d2trig

# ── Q100 因子 ──────────────────────────────────────────
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

# ── 数据加载 ──────────────────────────────────────────
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
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
lc = dfc.notna().cumsum(); lo = lambda s,d: (lc.at[d,s] if s in dfc.columns else 0) >= 20

ind_map = {}
mf = pd.read_csv(os.path.expanduser("~/ai-capital-ashare/data/industry_map_bs.csv"), dtype=str)
mf = mf[mf["industryClassification"]=="证监会行业分类"]; mf["bare"] = mf["code"].str[3:]
for _,r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"]!="": ind_map[r["bare"]] = str(r["industry"])
sd = load_qmt_stocks(min_history=250)
print(f"  ETF: {len(dfc.columns)}只, 个股: {len(sd)}只")

def make_sel():
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
        top = [e for e,_ in ranked[:2] if lo(e,date)]
        gb = [s for s in ["sz159915","sh588000"] if lo(s,date)]
        while len(top)<2 and gb: top.append(gb.pop(0))
        if len(top)<2: top = [s for s in BROAD if lo(s,date)][:2]
        is_s = d2t.get(date) is not None
        if is_s: r = top[:2]
        else:
            r = top[:2]
            for g in ["sz159915","sh588000"]:
                if lo(g,date) and g not in r: r.append(g)
            while len(r)<4:
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
    yearly = {int(k):round(v,1) for k,v in (rt.groupby(rt.index.year).apply(lambda x: ((1+x).prod()-1)*100)).items()}
    return {"ann":round(ann*100,2),"mdd":round(dd*100,1),"sharpe":round(sh,2),"turnover":round(t,1),"yearly":yearly}

# ── 三变体 ───────────────────────────────────────────
variants = []

# V2 两档+SURGE28d
print("\n[V2 两档+SURGE28d]", flush=True)
extra, forced, d2t = build_surge_orig(idx["close"], dfc, rs_raw, cal, me, lock_days=28)
rd = pd.DatetimeIndex(sorted(set(me)|set(extra)))
rs_v2 = rs_raw.copy()
for d in rs_v2.index: rs_v2.loc[d] = 0.85 if rs_v2.loc[d] >= 0.70 else 0.20
sf = make_sel()
r = run_stock_backtest(dfc, rs_v2, {}, top_n=4, verbose=False, execution="next_open",
    stamp_duty=True, ffill_valuation=True, df_open=dfo,
    rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)
s = stats(r)
# 统计 SURGE 次数
surge_dates = sorted(set(d2t.values()))
surge_dates = [d for d in surge_dates if d is not None]
s['surge_count'] = len(surge_dates)
variants.append(("V2 两档+SURGE28d", s))
print(json.dumps({k:v for k,v in s.items()}, ensure_ascii=False, indent=2))

# S42
print("\n[S42 SURGE独立42d]", flush=True)
extra, forced, d2t = build_surge_orig(idx["close"], dfc, rs_raw, cal, me, lock_days=42)
trig_dates = [d for d,t in d2t.items() if d==t]
rd = pd.DatetimeIndex(sorted(set(me)|set(extra)|set(trig_dates)))
rs_empty = pd.Series(0.20, index=rs_raw.index)
forced_all = {d:"RISKON" for d in d2t if d2t[d] is not None}
sf = make_sel()
r = run_stock_backtest(dfc, rs_empty, {}, top_n=4, verbose=False, execution="next_open",
    stamp_duty=True, ffill_valuation=True, df_open=dfo,
    rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced_all)
s = stats(r)
surge_dates = sorted(set(d2t.values())); surge_dates = [d for d in surge_dates if d is not None]
s['surge_count'] = len(surge_dates)
variants.append(("S42 SURGE独立42d", s))
print(json.dumps({k:v for k,v in s.items()}, ensure_ascii=False, indent=2))

# 四档
print("\n[四档+SURGE21d]", flush=True)
extra, forced, d2t = build_surge_orig(idx["close"], dfc, rs_raw, cal, me, lock_days=21)
rd = pd.DatetimeIndex(sorted(set(me)|set(extra)))
sf = make_sel()
r = run_stock_backtest(dfc, rs_raw, {}, top_n=4, verbose=False, execution="next_open",
    stamp_duty=True, ffill_valuation=True, df_open=dfo,
    rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)
s = stats(r)
surge_dates = sorted(set(d2t.values())); surge_dates = [d for d in surge_dates if d is not None]
s['surge_count'] = len(surge_dates)
variants.append(("四档+SURGE21d", s))
print(json.dumps({k:v for k,v in s.items()}, ensure_ascii=False, indent=2))

# ── 汇总 ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SURGE 原始设计 (breadth ≥ 2/3, 无15日跳跃条件)")
print(f"{'='*70}")

all_years = sorted(set().union(*[set(v[1]['yearly'].keys()) for v in variants]))
print(f"\n{'年':<6}", end="")
for name,_ in variants: print(f"{name:>20}", end="")
print()
for yr in all_years:
    print(f"{yr:<6}", end="")
    for _,s in variants: print(f"{s['yearly'].get(yr,0):>19.1f}%", end="")
    print()

print(f"\n{'指标':<8}", end="")
for name,_ in variants: print(f"{name:>20}", end="")
print(f"\n{'-'*68}")
for metric,label in [('ann','年化'),('mdd','最大回撤'),('sharpe','夏普'),('turnover','换手'),('surge_count','SURGE次数')]:
    print(f"{label:<8}", end="")
    for _,s in variants:
        v = s[metric]
        if metric=='ann': print(f"{v:>19.2f}%", end="")
        elif metric=='mdd': print(f"{v:>19.1f}%", end="")
        elif metric=='surge_count': print(f"{v:>20}", end="")
        else: print(f"{v:>20.2f}", end="")
    print()

# 对照
print(f"\n  对照(旧SURGE): V2 15.50%/-23.7%/0.89 | S42 14.10%/-23.8%/0.93 | 四档 14.58%/-21.0%/0.84")
