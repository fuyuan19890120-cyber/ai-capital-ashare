#!/usr/bin/env python3
"""运行三个变体，输出逐年收益 + 各时段选中的 ETF"""
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
from src.stock_data import compute_stock_factors, select_top_stocks

try:
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark, load_qmt_stocks
except ModuleNotFoundError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qmt_adapter", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "qmt_adapter.py"))
    qma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qma)
    load_qmt_etfs = qma.load_qmt_etfs; load_qmt_benchmark = qma.load_qmt_benchmark; load_qmt_stocks = qma.load_qmt_stocks

BROAD = ["sh510300", "sh510500", "sz159915", "sh588000"]
DEFENSE = ["sh511010", "sh518880", "sh511880"]
SECTOR_MAP = {
    "sh512880": ["证券", "资本市场", "货币金融"], "sh512800": ["货币金融", "金融服务"],
    "sh512010": ["医药", "制药", "医疗", "中药", "生物制品"],
    "sz159928": ["食品", "饮料", "酒", "农牧", "渔业", "养殖", "零售", "批发", "农副"],
    "sh512690": ["酒", "饮料"], "sh512480": ["电子", "计算机", "通信"],
    "sz159995": ["电子", "计算机", "通信"],
    "sh512660": ["航天", "航空", "船舶", "航海", "兵装", "军工", "国防", "卫星"],
    "sh515030": ["汽车", "新能源"], "sh515790": ["光伏", "电气机械", "电源"],
    "sh512400": ["金属", "有色", "黄金", "贵金属", "黑色金属", "采掘", "煤炭", "石油", "矿业", "矿物"],
    "sh512200": ["房地产", "建筑", "土木", "建材"],
    "sh512980": ["传媒", "游戏", "影视", "广告", "出版", "文化", "旅游", "教育", "广播"],
    "sz159939": ["软件", "互联网", "通信", "计算机", "IT", "信息服务", "专业技术"],
    "sh515210": ["黑色金属", "冶", "金属"],
    "sz159870": ["化学", "化纤", "塑料", "橡胶", "造纸", "纺织"],
    "sz159967": ["设备", "制造", "机械", "仪器", "装备", "高端", "智能", "自动化", "机器人"],
    "sz159611": ["电力", "热力", "燃气", "水的生产", "公用", "发电", "供电", "电网"],
    "sh512580": ["环保", "环境", "生态", "水利", "节能", "清洁", "循环", "废旧"],
    "sh516670": ["运输", "交通", "物流", "铁路", "道路", "航运", "水上", "港口", "快递", "高速", "机场"],
    "sh512070": ["保险", "非银", "其他金融"],
    "sz159865": ["农业", "农牧", "林业", "畜牧", "养殖", "种业", "种植", "饲料"],
    "sh516750": ["非金属矿物", "水泥", "玻璃", "陶瓷"],
    "sh512760": ["芯片", "半导体", "集成电路"], "sh515050": ["5G", "通信设备", "信息技术"],
    "sh516160": ["新能源", "碳中和", "清洁能源"],
}
ALL_ETF_CODES = BROAD + DEFENSE + list(SECTOR_MAP.keys())
DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")

# ETF 名称映射（从 fetch_strategy_etfs 推测）
ETF_NAMES = {
    "sh510300":"沪深300", "sh510500":"中证500", "sz159915":"创业板50", "sh588000":"科创50",
    "sh511010":"国债", "sh518880":"黄金", "sh511880":"货币",
    "sh512880":"证券", "sh512800":"银行", "sh512010":"医药", "sz159928":"消费", "sh512690":"酒",
    "sh512480":"半导体", "sz159995":"芯片", "sh512660":"军工", "sh515030":"新能源车",
    "sh515790":"光伏", "sh512400":"有色", "sh512200":"房地产", "sh512980":"传媒",
    "sz159939":"信息技术", "sh515210":"钢铁", "sz159870":"化工",
    "sz159967":"装备制造", "sz159611":"电力", "sh512580":"环保", "sh516670":"运输",
    "sh512070":"非银金融", "sz159865":"农业", "sh516750":"建材", "sh512760":"半导体",
    "sh515050":"5G通信", "sh516160":"新能源",
}

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

def load_index():
    try: return load_qmt_benchmark("000300.SH")
    except:
        etf_data = load_qmt_etfs(["sh510300"], verbose=False)
        return etf_data.get("sh510300")

def load_stocks_qmt():
    return load_qmt_stocks(min_history=250)

# ── 加载数据 ──
print("加载数据...")
dfc, dfo = load_panels()
idx = load_index(); rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
lc = dfc.notna().cumsum(); lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

ind_map = {}
mf = pd.read_csv(os.path.join(DATA_DIR, "industry_map_bs.csv"), dtype=str)
mf = mf[mf["industryClassification"]=="证监会行业分类"]; mf["bare"] = mf["code"].str[3:]
for _, r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"]!="": ind_map[r["bare"]] = str(r["industry"])
sd = load_stocks_qmt()

# 记录选中的 ETF
selection_log = []

def _v2_regime(rs_raw):
    """V2: 两档, score>=0.70→RISKON(0.85), else→CRISIS(0.20)"""
    rs = rs_raw.copy()
    for d in rs.index:
        rs.loc[d] = 0.85 if rs.loc[d] >= 0.70 else 0.20
    return rs


def make_sel():
    cache = {}
    def sel(date, d2t):
        if (date, id(d2t)) in cache: return cache[(date, id(d2t))]
        valid, votes = {}, {}
        for c, s in sd.items():
            if date in s.index and len(s[s.index <= date]) >= 250: valid[c] = s
        if not valid:
            r = [s for s in BROAD if lo(s, date)]; cache[(date, id(d2t))] = r; return r
        scores = compute_stock_factors(valid, date); top30 = select_top_stocks(scores, 30)
        for code in top30[:30]:
            ind = ind_map.get(code, "")
            for etf, kws in SECTOR_MAP.items():
                if any(k in ind for k in kws): votes[etf] = votes.get(etf,0)+1; break
        ranked = sorted(votes.items(), key=lambda x:x[1], reverse=True)
        top = [e for e,_ in ranked[:2] if lo(e, date)]
        gb = [s for s in ["sz159915","sh588000"] if lo(s, date)]
        while len(top) < 2 and gb: top.append(gb.pop(0))
        if len(top) < 2: top = [s for s in BROAD if lo(s, date)][:2]
        is_surge = d2t.get(date) is not None
        if is_surge:
            r = top[:2]
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

def run_variant(name, regime_fn, lock_days):
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=lock_days)
    trig_dates = [d for d,t in d2t.items() if d==t] if lock_days==42 else []
    rd = pd.DatetimeIndex(sorted(set(me)|set(extra)|set(trig_dates)))
    rs = regime_fn()

    # 记录选股日志
    sf = make_sel()
    sel_log = []
    wraps = lambda d: sf(d, d2t)
    # 在回测前先记录所有调仓日的选择
    for d in rd:
        picks = wraps(d)
        is_s = d2t.get(d) is not None
        try:
            regime_val = float(rs.get(d, rs.iloc[0])) if hasattr(rs, 'get') else float(rs.loc[d])
        except Exception:
            regime_val = 0.0
        sel_log.append({"date": str(d.date()), "picks": picks, "surge": is_s, "regime": regime_val})

    rr = run_stock_backtest(
        dfc, rs, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=wraps, forced_regime=forced)
    return rr, sel_log

def stats(rr):
    v = rr["values"]["value"]; rt = v.pct_change().dropna()
    yrs = len(rt)/244; ann = (v.iloc[-1]/1e6)**(1/yrs)-1
    dd = (v/v.cummax()-1).min(); sh = rt.mean()/rt.std()*np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]
    yearly = {int(k):round(v,1) for k,v in (rt.groupby(rt.index.year).apply(lambda x: ((1+x).prod()-1)*100)).items()}
    return {"ann":round(ann*100,2),"mdd":round(dd*100,1),"sharpe":round(sh,2),"turnover":round(t,1),"yearly":yearly}, rt

# ── 运行三个变体 ──
variants = [
    ("V2 两档+SURGE28d",
     (lambda: _v2_regime(rs_raw)),
     28),
    ("S42 SURGE独立42d",
     lambda: pd.Series(0.20, index=rs_raw.index),
     42),
    ("四档+SURGE21d",
     lambda: rs_raw,
     21),
]

for name, regime_fn, lock in variants:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    rr, sel_log = run_variant(name, regime_fn, lock)
    s, rt = stats(rr)
    print(f"年化 {s['ann']}%  回撤 {s['mdd']}%  夏普 {s['sharpe']}  换手 {s['turnover']}x")
    print(f"逐年: {s['yearly']}")

    # 按年汇总选中的 ETF
    print(f"\n📊 各年选中 ETF (按出现次数):")
    sel_df = pd.DataFrame(sel_log)
    sel_df['year'] = pd.to_datetime(sel_df['date']).dt.year
    for yr in sorted(sel_df['year'].unique()):
        picks_yr = []
        for _, row in sel_df[sel_df['year']==yr].iterrows():
            picks_yr.extend(row['picks'])
        counter = Counter(picks_yr)
        surge_count = sel_df[(sel_df['year']==yr) & (sel_df['surge'])].shape[0]
        names_str = ", ".join(f"{ETF_NAMES.get(c,c)}({n})" for c,n in counter.most_common(5))
        tag = f" ⚡{surge_count}次SURGE" if surge_count>0 else ""
        print(f"  {yr}: {names_str}{tag}")

    # 按分时段汇总
    print(f"\n📊 分时段 ETF 偏好:")
    for a,b,label in [("2016","2018","16-18"),("2019","2021","19-21"),("2022","2024","22-24"),("2025","2026","25-26")]:
        sub = sel_df[(sel_df['year']>=int(a)) & (sel_df['year']<=int(b))]
        picks_sub = []
        for _, row in sub.iterrows(): picks_sub.extend(row['picks'])
        counter = Counter(picks_sub)
        names_str = ", ".join(f"{ETF_NAMES.get(c,c)}({n})" for c,n in counter.most_common(5))
        surge_count = sub[sub['surge']].shape[0]
        tag = f" ⚡{surge_count}次SURGE" if surge_count>0 else ""
        print(f"  {label}: {names_str}{tag}")
