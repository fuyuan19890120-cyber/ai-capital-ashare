#!/usr/bin/env python3
"""
ETF-R1 V2 + S42: QMT 数据库版
- V2 主策略: 两档(0.70分界) + SURGE 28d
- S42 卫星策略: SURGE 独立 42d, 非 SURGE 全债
数据源: qmt_qfq.db (ETF前复权) + qmt_full.db (个股)
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/etf-r1"))

from config import START_DATE
from etf_r1_backtest import build_surge  # SECTOR_MAP 用下方扩充版
from run_final_backtest import compute_regime
from run_v5_backtest import month_end_dates
from src.stock_backtest import run_stock_backtest
from src.stock_data import compute_stock_factors, select_top_stocks

BROAD = ["sh510300", "sh510500", "sz159915", "sh588000"]
DEFENSE = ["sh511010", "sh518880", "sh511880"]

# ── 扩充版 SECTOR_MAP（26 只，覆盖率 ~94%）─────────────────
SECTOR_MAP = {
    # 原有 16 只
    "sh512880": ["证券", "资本市场", "货币金融"],
    "sh512800": ["货币金融", "金融服务"],
    "sh512010": ["医药", "制药", "医疗", "中药", "生物制品"],
    "sz159928": ["食品", "饮料", "酒", "农牧", "渔业", "养殖", "零售", "批发", "农副"],
    "sh512690": ["酒", "饮料"],
    "sh512480": ["电子", "计算机", "通信"],
    "sz159995": ["电子", "计算机", "通信"],
    "sh512660": ["航天", "航空", "船舶", "航海", "兵装", "军工", "国防", "卫星"],
    "sh515030": ["汽车", "新能源"],
    "sh515790": ["光伏", "电气机械", "电源"],
    "sh512400": ["金属", "有色", "黄金", "贵金属", "黑色金属", "采掘", "煤炭", "石油", "矿业", "矿物"],
    "sh512200": ["房地产", "建筑", "土木", "建材"],
    "sh512980": ["传媒", "游戏", "影视", "广告", "出版", "文化", "旅游", "教育", "广播"],
    "sz159939": ["软件", "互联网", "通信", "计算机", "IT", "信息服务", "专业技术"],
    "sh515210": ["黑色金属", "冶", "金属"],
    "sz159870": ["化学", "化纤", "塑料", "橡胶", "造纸", "纺织"],
    # 新增 10 只（7只新拉 + 3只已在库）
    "sz159967": ["设备", "制造", "机械", "仪器", "装备", "高端", "智能", "自动化", "机器人"],
    "sz159611": ["电力", "热力", "燃气", "水的生产", "公用", "发电", "供电", "电网"],
    "sh512580": ["环保", "环境", "生态", "水利", "节能", "清洁", "循环", "废旧"],
    "sh516670": ["运输", "交通", "物流", "铁路", "道路", "航运", "水上", "港口", "快递", "高速", "机场"],
    "sh512070": ["保险", "非银", "其他金融"],
    "sz159865": ["农业", "农牧", "林业", "畜牧", "养殖", "种业", "种植", "饲料"],
    "sh516750": ["非金属矿物", "水泥", "玻璃", "陶瓷"],
    "sh512760": ["芯片", "半导体", "集成电路"],
    "sh515050": ["5G", "通信设备", "信息技术"],
    "sh516160": ["新能源", "碳中和", "清洁能源"],
}

ALL_ETF_CODES = BROAD + DEFENSE + list(SECTOR_MAP.keys())
DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")

# ── QMT 数据加载 ──────────────────────────────────────
try:
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark, load_qmt_stocks
except ModuleNotFoundError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qmt_adapter", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "qmt_adapter.py"))
    qma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qma)
    load_qmt_etfs = qma.load_qmt_etfs
    load_qmt_benchmark = qma.load_qmt_benchmark
    load_qmt_stocks = qma.load_qmt_stocks


def load_panels():
    print("从 QMT 加载 ETF panel...")
    etf_data = load_qmt_etfs(ALL_ETF_CODES, verbose=False)
    closes, opens = {}, {}
    for sym in ALL_ETF_CODES:
        if sym in etf_data:
            df = etf_data[sym]
            closes[sym] = df["close"]
            opens[sym] = df["open"]
    dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
    end_dates = []
    for s in BROAD + DEFENSE:
        if s in dfc.columns:
            lvi = dfc[s].last_valid_index()
            if lvi is not None:
                end_dates.append(lvi)
    if end_dates:
        end = min(end_dates)
        dfc = dfc.loc[:end].ffill(limit=5)
    dfo = pd.DataFrame(opens).sort_index().reindex(dfc.index)
    print(f"  ETF panel: {len(dfc.columns)} 只, {len(dfc)} 天")
    return dfc, dfo


def load_index():
    try:
        return load_qmt_benchmark("000300.SH")
    except Exception:
        etf_data = load_qmt_etfs(["sh510300"], verbose=False)
        if "sh510300" in etf_data:
            return etf_data["sh510300"]
        return None


def load_stocks_qmt():
    print("从 QMT 加载个股数据...")
    return load_qmt_stocks(min_history=250)


# ── 主流程 ──────────────────────────────────────────────
print("=" * 55)
print("ETF-R1 QMT 版: V2 两档+SURGE28d | S42 SURGE独立42d")
print("=" * 55)

dfc, dfo = load_panels()
idx = load_index()
rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]
me = month_end_dates(cal)
lc = dfc.notna().cumsum()
lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

# 行业映射
ind_map = {}
mf = pd.read_csv(os.path.join(DATA_DIR, "industry_map_bs.csv"), dtype=str)
mf = mf[mf["industryClassification"] == "证监会行业分类"]
mf["bare"] = mf["code"].str[3:]
for _, r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"] != "":
        ind_map[r["bare"]] = str(r["industry"])

sd = load_stocks_qmt()


def make_sel():
    cache = {}

    def sel(date, d2t):
        if (date, id(d2t)) in cache:
            return cache[(date, id(d2t))]
        valid, votes = {}, {}
        for c, s in sd.items():
            if date in s.index and len(s[s.index <= date]) >= 250:
                valid[c] = s
        if not valid:
            r = [s for s in BROAD if lo(s, date)]
            cache[(date, id(d2t))] = r
            return r
        scores = compute_stock_factors(valid, date)
        top30 = select_top_stocks(scores, 30)
        for code in top30[:30]:
            ind = ind_map.get(code, "")
            for etf, kws in SECTOR_MAP.items():
                if any(k in ind for k in kws):
                    votes[etf] = votes.get(etf, 0) + 1
                    break
        ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top = [e for e, _ in ranked[:2] if lo(e, date)]
        gb = [s for s in ["sz159915", "sh588000"] if lo(s, date)]
        while len(top) < 2 and gb:
            top.append(gb.pop(0))
        if len(top) < 2:
            top = [s for s in BROAD if lo(s, date)][:2]
        is_surge = d2t.get(date) is not None
        if is_surge:
            r = top[:2]
        else:
            r = top[:2]
            for g in ["sz159915", "sh588000"]:
                if lo(g, date) and g not in r:
                    r.append(g)
            while len(r) < 4:
                for e, _ in ranked:
                    if e not in r and lo(e, date):
                        r.append(e)
                        break
                else:
                    break
        cache[(date, id(d2t))] = r[:4]
        return r[:4]

    return sel


def run_v2():
    """主策略: 两档 0.70 + SURGE 28d"""
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=28)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
    rs_mod = rs_raw.copy()
    for d in rs_mod.index:
        rs_mod.loc[d] = 0.85 if rs_mod.loc[d] >= 0.70 else 0.20
    sf = make_sel()
    return run_stock_backtest(
        dfc, rs_mod, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)


def run_s42():
    """卫星策略: SURGE 独立 42d, 非 SURGE 全债"""
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=42)
    trig_dates = [d for d, t in d2t.items() if d == t]
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra) | set(trig_dates)))
    rs_empty = pd.Series(0.20, index=rs_raw.index)
    forced_all = {d: "RISKON" for d in d2t if d2t[d] is not None}
    sf = make_sel()
    return run_stock_backtest(
        dfc, rs_empty, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced_all)


def stats(rr):
    v = rr["values"]["value"]
    rt = v.pct_change().dropna()
    yrs = len(rt) / 244
    ann = (v.iloc[-1] / 1e6) ** (1 / yrs) - 1
    dd = (v / v.cummax() - 1).min()
    sh = rt.mean() / rt.std() * np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]
    yearly = {int(k): round(v, 1) for k, v in
              (rt.groupby(rt.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)).items()}
    return {"ann": round(ann * 100, 2), "mdd": round(dd * 100, 1), "sharpe": round(sh, 2),
            "turnover": round(t, 1), "yearly": yearly}


if __name__ == "__main__":
    print("V2 主策略(两档0.70+SURGE28d):", flush=True)
    r_v2 = run_v2()
    s_v2 = stats(r_v2)
    print(json.dumps(s_v2, ensure_ascii=False, indent=2))

    print("\nS42 卫星策略(SURGE独立42d):", flush=True)
    r_s42 = run_s42()
    s_s42 = stats(r_s42)
    print(json.dumps(s_s42, ensure_ascii=False, indent=2))

    out = os.path.join(os.path.dirname(__file__), "backtests", "etf_v2_s42_qmt.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"V2": s_v2, "S42": s_s42}, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")
