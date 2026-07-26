#!/usr/bin/env python3
"""
ETF-R1 最终版: Q100 因子 + 三个变体框架
- V2 两档+SURGE28d (年化最强)
- S42 SURGE独立42d (回撤最浅)
- 四档+SURGE21d (最均衡)
因子: Quality (250日涨幅,≥0), 纯量价, 在QMT个股上计算
数据: qmt_qfq.db (ETF) + qmt_full.db (个股) + industry_map_bs.csv
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/etf-r1"))

from config import START_DATE
from etf_r1_backtest import build_surge
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
ETF_NAMES = {k: k for k in SECTOR_MAP}
ETF_NAMES.update({"sh510300":"沪深300","sh510500":"中证500","sz159915":"创业板50","sh588000":"科创50",
    "sh511010":"国债","sh518880":"黄金","sh511880":"货币","sh512880":"证券","sh512800":"银行",
    "sh512010":"医药","sz159928":"消费","sh512690":"酒","sh512480":"半导体","sz159995":"芯片",
    "sh512660":"军工","sh515030":"新能源车","sh515790":"光伏","sh512400":"有色","sh512200":"房地产",
    "sh512980":"传媒","sz159939":"信息技术","sh515210":"钢铁","sz159870":"化工",
    "sz159967":"装备制造","sz159611":"电力","sh512580":"环保","sh516670":"运输",
    "sh512070":"非银金融","sz159865":"农业","sh516750":"建材","sh512760":"半导体","sh515050":"5G通信",
    "sh516160":"新能源"})

# ── Q100 因子 ────────────────────────────────────────────
def compute_q100(stock_data, date):
    """Quality: 250日涨幅 (≥0), log变换"""
    scores = {}
    for code, df in stock_data.items():
        if date not in df.index: continue
        idx = df.index.get_loc(date)
        if idx < 250: continue
        lr = df['close'].iloc[idx] / df['close'].iloc[idx - 250] - 1
        if lr > 0:
            scores[code] = np.log1p(lr)
        else:
            scores[code] = 0  # 负收益的股票得0分
    return scores


def select_top_stocks(scores, top_n=30):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [code for code, _ in ranked[:top_n]]


# ── 数据加载 ─────────────────────────────────────────────
print("加载数据...")
def load_panels():
    etf_data = load_qmt_etfs(ALL_ETF_CODES, verbose=False)
    closes, opens = {}, {}
    for sym in ALL_ETF_CODES:
        if sym in etf_data:
            closes[sym] = etf_data[sym]["close"]; opens[sym] = etf_data[sym]["open"]
    dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
    ends = [dfc[s].last_valid_index() for s in BROAD+DEFENSE if s in dfc.columns]
    if ends: dfc = dfc.loc[:min(ends)].ffill(limit=5)
    dfo = pd.DataFrame(opens).sort_index().reindex(dfc.index)
    return dfc, dfo

dfc, dfo = load_panels()
idx = load_qmt_benchmark("000300.SH")
rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
lc = dfc.notna().cumsum(); lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

ind_map = {}
mf = pd.read_csv(os.path.expanduser("~/ai-capital-ashare/data/industry_map_bs.csv"), dtype=str)
mf = mf[mf["industryClassification"]=="证监会行业分类"]; mf["bare"] = mf["code"].str[3:]
for _, r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"]!="": ind_map[r["bare"]] = str(r["industry"])
sd = load_qmt_stocks(min_history=250)
print(f"  ETF: {len(dfc.columns)}只, 个股: {len(sd)}只")

# ── 选股函数 ─────────────────────────────────────────────
def make_sel():
    cache = {}
    def sel(date, d2t):
        if (date, id(d2t)) in cache: return cache[(date, id(d2t))]
        valid = {}
        for c, s in sd.items():
            if date in s.index and len(s[s.index <= date]) >= 250: valid[c] = s
        if not valid:
            r = [s for s in BROAD if lo(s, date)]; cache[(date, id(d2t))] = r; return r
        scores = compute_q100(valid, date)
        top30 = select_top_stocks(scores, 30)
        votes = {}
        for code in top30[:30]:
            ind = ind_map.get(code, "")
            for etf, kws in SECTOR_MAP.items():
                if any(k in ind for k in kws): votes[etf] = votes.get(etf,0)+1; break
        ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top = [e for e,_ in ranked[:2] if lo(e, date)]
        gb = [s for s in ["sz159915","sh588000"] if lo(s, date)]
        while len(top) < 2 and gb: top.append(gb.pop(0))
        if len(top) < 2: top = [s for s in BROAD if lo(s, date)][:2]
        is_s = d2t.get(date) is not None
        if is_s: r = top[:2]
        else:
            r = top[:2]
            for g in ["sz159915","sh588000"]:
                if lo(g, date) and g not in r: r.append(g)
            while len(r) < 4:
                for e,_ in ranked:
                    if e not in r and lo(e, date): r.append(e); break
                else: break
        cache[(date, id(d2t))] = r[:4]; return r[:4]
    return sel


def stats(rr):
    v = rr["values"]["value"]; rt = v.pct_change().dropna()
    yrs = len(rt)/244; ann = (v.iloc[-1]/1e6)**(1/yrs)-1
    dd = (v/v.cummax()-1).min(); sh = rt.mean()/rt.std()*np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]
    yearly = {int(k):round(v,1) for k,v in (rt.groupby(rt.index.year).apply(lambda x: ((1+x).prod()-1)*100)).items()}
    # 分时段
    sub = {}
    for a,b in [("2016","2018"),("2019","2021"),("2022","2024"),("2025","2026")]:
        s = rt.loc[a:b]; sub[a+"-"+b[2:]] = round(((1+s).prod()**(244/len(s))-1)*100,1)
    # 选中的 ETF
    return {"ann":round(ann*100,2),"mdd":round(dd*100,1),"sharpe":round(sh,2),
            "turnover":round(t,1),"yearly":yearly,"sub":sub}

# ── 三变体 ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"ETF-R1 Q100 最终版")
print(f"{'='*70}")

variants = []

# 1) V2 两档+SURGE28d
print("\n[V2 两档+SURGE28d]", flush=True)
rs_v2 = rs_raw.copy()
for d in rs_v2.index: rs_v2.loc[d] = 0.85 if rs_v2.loc[d] >= 0.70 else 0.20
extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=28)
rd = pd.DatetimeIndex(sorted(set(me)|set(extra)))
sf = make_sel()
r_v2 = run_stock_backtest(dfc, rs_v2, {}, top_n=4, verbose=False, execution="next_open",
    stamp_duty=True, ffill_valuation=True, df_open=dfo,
    rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)
s_v2 = stats(r_v2)

# ETF 偏好
sel_log = []
for d in rd: sel_log.extend([(str(d.date()), c) for c in sf(d, d2t)])
sel_df = pd.DataFrame(sel_log, columns=['date','etf'])
sel_df['year'] = pd.to_datetime(sel_df['date']).dt.year
etf_prefs = {}
for yr in sorted(sel_df['year'].unique()):
    c = Counter(sel_df[sel_df['year']==yr]['etf'])
    etf_prefs[str(yr)] = [(ETF_NAMES.get(k,k),v) for k,v in c.most_common(4)]

s_v2['etf_prefs'] = etf_prefs
variants.append(("V2 两档+SURGE28d", s_v2))
print(json.dumps({k:v for k,v in s_v2.items() if k!='etf_prefs'}, ensure_ascii=False, indent=2))

# 2) S42
print("\n[S42 SURGE独立42d]", flush=True)
extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=42)
trig_dates = [d for d,t in d2t.items() if d==t]
rd = pd.DatetimeIndex(sorted(set(me)|set(extra)|set(trig_dates)))
rs_empty = pd.Series(0.20, index=rs_raw.index)
forced_all = {d:"RISKON" for d in d2t if d2t[d] is not None}
sf = make_sel()
r_s42 = run_stock_backtest(dfc, rs_empty, {}, top_n=4, verbose=False, execution="next_open",
    stamp_duty=True, ffill_valuation=True, df_open=dfo,
    rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced_all)
s_s42 = stats(r_s42)
variants.append(("S42 SURGE独立42d", s_s42))
print(json.dumps({k:v for k,v in s_s42.items() if k!='etf_prefs'}, ensure_ascii=False, indent=2))

# 3) 四档
print("\n[四档+SURGE21d]", flush=True)
extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=21)
rd = pd.DatetimeIndex(sorted(set(me)|set(extra)))
sf = make_sel()
r_4d = run_stock_backtest(dfc, rs_raw, {}, top_n=4, verbose=False, execution="next_open",
    stamp_duty=True, ffill_valuation=True, df_open=dfo,
    rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)
s_4d = stats(r_4d)
variants.append(("四档+SURGE21d", s_4d))
print(json.dumps({k:v for k,v in s_4d.items() if k!='etf_prefs'}, ensure_ascii=False, indent=2))

# ── 汇总 ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("ETF-R1 Q100 最终汇总")
print(f"{'='*70}")

# 逐年对比
all_years = sorted(set().union(*[set(v[1]['yearly'].keys()) for v in variants]))
print(f"\n{'年':<6}", end="")
for name, _ in variants: print(f"{name:>18}", end="")
print()
for yr in all_years:
    print(f"{yr:<6}", end="")
    for _, s in variants: print(f"{s['yearly'].get(yr, 0):>17.1f}%", end="")
    print()

# 核心指标
print(f"\n{'指标':<10}", end="")
for name, _ in variants: print(f"{name:>18}", end="")
print(f"\n{'-'*65}")
for metric, label in [('ann','年化'),('mdd','最大回撤'),('sharpe','夏普'),('turnover','换手')]:
    print(f"{label:<10}", end="")
    for _, s in variants:
        v = s[metric]
        if metric == 'ann': print(f"{v:>17.2f}%", end="")
        elif metric == 'mdd': print(f"{v:>17.1f}%", end="")
        else: print(f"{v:>18.2f}", end="")
    print()

# 分时段
print(f"\n{'时段':<8}", end="")
for name, _ in variants: print(f"{name:>18}", end="")
print()
for period in ['16-18','19-21','22-24','25-26']:
    print(f"{period:<8}", end="")
    for _, s in variants:
        v = s.get('sub', {}).get(period, 0)
        print(f"{v:>17.1f}%", end="")
    print()

# V2 ETF 偏好
print(f"\n📊 V2 各年 ETF 偏好:")
for yr in sorted(etf_prefs.keys(), key=int):
    items = ", ".join(f"{n}({c})" for n,c in etf_prefs[yr])
    print(f"  {yr}: {items}")

out = os.path.join(os.path.dirname(__file__), "backtests", "etf_r1_q100_final.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,"w") as f:
    json.dump([{"name":n, **s} for n,s in variants], f, ensure_ascii=False, indent=2, default=str)
print(f"\n[saved] {out}")
