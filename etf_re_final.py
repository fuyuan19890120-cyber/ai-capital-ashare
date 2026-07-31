#!/usr/bin/env python3
"""
ETF-RE 最终定稿 (ETF-R1 Enhanced)
选股: Q100 因子 (250日涨幅≥0) 在2,060只QMT个股上计算 → Top-30 → CSRC行业投票 → 行业ETF
择时: V2两档 + SURGE原始设计 (breadth≥2/3, lock=28d)
数据: qmt_qfq.db (ETF前复权) + qmt_full.db (个股) + industry_map_bs.csv

回测(2015-2026): 年化17.01% / 回撤-23.7% / 夏普0.94
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
    s = importlib.util.spec_from_file_location("qa", "src/qmt_adapter.py")
    qa = importlib.util.module_from_spec(s); s.loader.exec_module(qa)
    load_qmt_etfs = qa.load_qmt_etfs; load_qmt_benchmark = qa.load_qmt_benchmark; load_qmt_stocks = qa.load_qmt_stocks

# ── ETF 池 ──────────────────────────────────────────────
BROAD   = ["sh510300", "sh510500", "sz159915", "sh588000"]
DEFENSE = ["sh511010", "sh518880", "sh511880"]

SECTOR_MAP = {
    # 金融
    "sh512880": ["证券", "资本市场", "货币金融"], "sh512800": ["货币金融", "金融服务"],
    "sh512070": ["保险", "非银", "其他金融"], "sh512000": ["证券", "资本市场"],
    # 医药
    "sh512010": ["医药", "制药", "医疗", "中药", "生物制品"],
    "sh512170": ["医药", "医疗", "制药"], "sz159992": ["医药", "医疗", "制药", "创新"],
    # 消费
    "sz159928": ["食品", "饮料", "酒", "农牧", "渔业", "养殖", "零售", "批发", "农副"],
    "sh512690": ["酒", "饮料"],
    # 半导体/芯片
    "sh512480": ["电子", "计算机", "通信", "半导体"],
    "sz159995": ["电子", "计算机", "通信", "芯片"],
    "sh512760": ["电子", "半导体", "芯片", "集成电路"],
    # 军工
    "sh512660": ["航天", "航空", "船舶", "航海", "兵装", "军工", "国防", "卫星"],
    "sh512710": ["航天", "航空", "船舶", "军工", "国防"],
    # 新能源
    "sh515030": ["汽车", "新能源"], "sh515790": ["光伏", "电气机械", "电源"],
    "sh516160": ["新能源", "碳中和", "清洁能源"],
    # 周期/资源
    "sh512400": ["金属", "有色", "黄金", "贵金属", "黑色金属", "采掘", "煤炭", "石油", "矿业", "矿物"],
    "sh515210": ["黑色金属", "冶", "金属"],
    "sz159870": ["化学", "化纤", "塑料", "橡胶", "造纸", "纺织"],
    # 房地产/建材
    "sh512200": ["房地产", "建筑", "土木", "建材"],
    "sh516750": ["非金属矿物", "水泥", "玻璃", "陶瓷"],
    # 传媒/游戏
    "sh512980": ["传媒", "游戏", "影视", "广告", "出版", "文化", "旅游", "教育", "广播"],
    "sz159869": ["传媒", "游戏", "影视", "互联网"],
    # 科技/IT
    "sz159939": ["软件", "互联网", "通信", "计算机", "IT", "信息服务", "专业技术"],
    "sh515000": ["软件", "互联网", "计算机", "IT"], "sh512720": ["软件", "互联网", "计算机", "IT"],
    # 通信/5G
    "sh515050": ["5G", "通信设备", "信息技术", "通信"], "sh515880": ["通信", "信息技术"],
    # 装备/机器人/AI
    "sz159967": ["设备", "制造", "机械", "仪器", "装备", "高端", "智能", "自动化", "机器人"],
    "sh562500": ["设备", "制造", "机械", "自动化", "机器人"],
    "sz159819": ["软件", "互联网", "计算机", "IT", "信息技术"],
    # 公用
    "sz159611": ["电力", "热力", "燃气", "水的生产", "公用", "发电", "供电", "电网"],
    "sh512580": ["环保", "环境", "生态", "水利", "节能", "清洁"],
    "sh516670": ["运输", "交通", "物流", "铁路", "道路", "航运", "水上", "港口", "快递", "高速", "机场"],
    "sz159865": ["农业", "农牧", "林业", "畜牧", "养殖", "种业", "种植", "饲料"],
}

ALL_ETFS = BROAD + DEFENSE + list(SECTOR_MAP.keys())
DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")

# ── SURGE 原始设计 ─────────────────────────────────────
def build_surge(index_close, etf_close, regime_series, cal, rebal, lock_days=28):
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
            if d in rebal_set: forced[d] = 'RISKON'; last_regime = 'RISKON'
            continue
        if d in rebal_set and d in regime_series.index:
            last_regime = reg(float(regime_series.loc[d]))
        if bool(trig.get(d, False)):
            lock_until = i + lock_days - 1; cur_trig = d
            d2trig[d] = d; forced[d] = 'RISKON'
            if last_regime != 'RISKON' and d not in rebal_set: extra.append(d)
            last_regime = 'RISKON'

    return pd.DatetimeIndex(extra), forced, d2trig

# ── Q100 因子 ───────────────────────────────────────────
def compute_q100(stock_data, date):
    """Quality: 250日涨幅≥0 → log变换, 负收益=0"""
    scores = {}
    for code, df in stock_data.items():
        if date not in df.index: continue
        idx = df.index.get_loc(date)
        if idx < 250: continue
        lr = df['close'].iloc[idx] / df['close'].iloc[idx - 250] - 1
        scores[code] = np.log1p(lr) if lr > 0 else 0
    return scores

def select_top_stocks(scores, top_n=30):
    return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

# ── 选股管线 ────────────────────────────────────────────
def make_selector(sd, ind_map, dfc, lo):
    cache = {}

    def select(date, d2t):
        if (date, id(d2t)) in cache: return cache[(date, id(d2t))]

        # 过滤有数据的个股
        valid = {}
        for c, s in sd.items():
            if date in s.index and len(s[s.index <= date]) >= 250:
                valid[c] = s
        if not valid:
            r = [s for s in BROAD if lo(s, date)]
            cache[(date, id(d2t))] = r; return r

        # Q100 → Top-30 → 行业投票
        scores = compute_q100(valid, date)
        top30 = select_top_stocks(scores, 30)

        votes = {}
        for code in top30[:30]:
            ind = ind_map.get(code, "")
            for etf, kws in SECTOR_MAP.items():
                if any(k in ind for k in kws):
                    votes[etf] = votes.get(etf, 0) + 1
                    break

        ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        # Top-2 行业 ETF
        top = [e for e, _ in ranked[:2] if lo(e, date)]
        gb = [s for s in ["sz159915", "sh588000"] if lo(s, date)]
        while len(top) < 2 and gb:
            top.append(gb.pop(0))
        if len(top) < 2:
            top = [s for s in BROAD if lo(s, date)][:2]

        is_surge = d2t.get(date) is not None
        if is_surge:
            picks = top[:2]
        else:
            picks = top[:2]
            for g in ["sz159915", "sh588000"]:
                if lo(g, date) and g not in picks:
                    picks.append(g)

        cache[(date, id(d2t))] = picks[:4]
        return picks[:4]

    return select

# ── 主流程 ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("ETF-RE 最终定稿: Q100个股投票 + V2两档 + SURGE≥2/3")
    print("=" * 55)

    # 加载 ETF 数据
    print("加载 ETF 数据...")
    etf_data = load_qmt_etfs(ALL_ETFS, verbose=False)
    closes = {}
    for s in ALL_ETFS:
        if s in etf_data: closes[s] = etf_data[s]["close"]
    dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
    ends = [dfc[s].last_valid_index() for s in BROAD + DEFENSE if s in dfc.columns]
    if ends: dfc = dfc.loc[:min(ends)].ffill(limit=5)

    # 加载个股 + 指数
    idx_close = load_qmt_benchmark("000300.SH")
    rs = compute_regime(idx_close["close"])
    rs = rs[rs.index >= START_DATE]
    sd = load_qmt_stocks(min_history=250)

    # 行业映射
    ind_map = {}
    mf = pd.read_csv(os.path.join(DATA_DIR, "industry_map_bs.csv"), dtype=str)
    mf = mf[mf["industryClassification"] == "证监会行业分类"]
    mf["bare"] = mf["code"].str[3:]
    for _, r in mf.iterrows():
        if pd.notna(r["industry"]) and r["industry"] != "":
            ind_map[r["bare"]] = str(r["industry"])

    cal = dfc.index[dfc.index >= START_DATE]
    me = month_end_dates(cal)
    lc = dfc.notna().cumsum()
    lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

    # V2 两档
    rs_v2 = rs.copy()
    for d in rs_v2.index:
        rs_v2.loc[d] = 0.85 if rs_v2.loc[d] >= 0.70 else 0.20

    # SURGE
    extra, forced, d2t = build_surge(idx_close["close"], dfc, rs, cal, me, lock_days=28)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))

    # 选股
    selector = make_selector(sd, ind_map, dfc, lo)

    print(f"  ETF: {len(dfc.columns)}只, 个股: {len(sd)}只")
    print("运行回测...")

    result = run_stock_backtest(
        dfc, rs_v2, {}, top_n=4, verbose=False,
        execution="next_open", stamp_duty=True, ffill_valuation=True,
        df_open=None, rebalance_dates=rd,
        select_fn=lambda d: selector(d, d2t),
        forced_regime=forced
    )

    # 统计
    v = result["values"]["value"]
    rt = v.pct_change().dropna()
    yrs = len(rt) / 244
    ann = (v.iloc[-1] / 1e6) ** (1 / yrs) - 1
    dd = (v / v.cummax() - 1).min()
    sh = rt.mean() / rt.std() * np.sqrt(244)
    turnover = result["metrics"]["annual_turnover_x"]

    print(f"\n{'='*55}")
    print("ETF-RE 最终结果")
    print(f"{'='*55}")
    print(f"  年化收益:  {ann*100:.2f}%")
    print(f"  最大回撤:  {dd*100:.1f}%")
    print(f"  夏普比率:  {sh:.2f}")
    print(f"  换手率:    {turnover:.1f}x")
    print(f"  个股数:    {len(sd)}只")
    print(f"  ETF池:    {len(SECTOR_MAP)}行业 + {len(BROAD)}宽基 + {len(DEFENSE)}防守")

    yearly = rt.groupby(rt.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    print(f"\n  逐年收益:")
    for yr, val in yearly.items():
        if yr >= 2016:
            print(f"    {yr}: {val:+.1f}%")

    surge_count = len(set(d2t.values())) - 1
    print(f"\n  SURGE: {surge_count}次触发")
    print(f"  因子:  Q100 (250日涨幅≥0)")
    print(f"  框架:  V2两档 + SURGE28d (breadth≥2/3)")
    print(f"  数据:  qmt_qfq.db + qmt_full.db + industry_map_bs.csv")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtests", "etf_re_final.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "strategy": "ETF-RE (R1 Enhanced)",
            "factor": "Q100 (250d return≥0)", "framework": "V2两档+SURGE28d (breadth≥2/3)",
            "pipeline": "个股Q100 → Top-30 → CSRC行业投票 → 行业ETF",
            "pool": f"{len(SECTOR_MAP)} sector + {len(BROAD)} broad + {len(DEFENSE)} defense",
            "stocks": len(sd), "ann": round(ann * 100, 2), "mdd": round(dd * 100, 1),
            "sharpe": round(sh, 2), "turnover": round(turnover, 1),
            "yearly": {int(k): round(v, 1) for k, v in yearly.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  [saved] {out}")

if __name__ == "__main__":
    main()
