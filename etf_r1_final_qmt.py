#!/usr/bin/env python3
"""
ETF-R1 最终定稿: QMT 数据库版
- 普通 RISKON: 2行业ETF(各25%)+创业板50(30%)+科创50(20%)
- SURGE 期: 纯2行业ETF(各50%, 100%集中)
- 信号栈: V4.1 SMA250四档 + SURGE三宽基口径(日频/锁21交易日)
- 数据源: qmt_qfq.db (ETF前复权) + qmt_full.db (个股) + industry_map_bs.csv (行业映射)
"""
import os, sys, json, glob, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

# 添加 etf-r1 worktree 路径，复用不改动的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/etf-r1"))

from config import START_DATE
from etf_r1_backtest import build_surge  # SECTOR_MAP 用下方扩充版
from run_final_backtest import compute_regime
from run_v5_backtest import month_end_dates
from src.stock_backtest import run_stock_backtest
from src.stock_data import compute_stock_factors, select_top_stocks
try:
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark, load_qmt_stocks
except ModuleNotFoundError:
    # 如果其他 worktree 的 src shadow 了我们的 src，用 importlib 直接加载
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qmt_adapter", os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "qmt_adapter.py"))
    qma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qma)
    load_qmt_etfs = qma.load_qmt_etfs
    load_qmt_benchmark = qma.load_qmt_benchmark
    load_qmt_stocks = qma.load_qmt_stocks

# ── 常量（复用原版）─────────────────────────────────────
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
    # 新增 10 只
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

# ── QMT 版数据加载 ──────────────────────────────────────
def load_panels():
    """从 QMT 前复权库加载 23 只 ETF 的 close/open 宽表"""
    print("从 QMT 加载 ETF panel...")
    etf_data = load_qmt_etfs(ALL_ETF_CODES, verbose=False)

    closes, opens = {}, {}
    for sym in ALL_ETF_CODES:
        if sym in etf_data:
            df = etf_data[sym]
            closes[sym] = df["close"]
            opens[sym] = df["open"]

    dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
    # 样本终点=宽基+防御共同最后有效日
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
    print(f"  ETF panel: {len(dfc.columns)} 只, {len(dfc)} 天, {dfc.index[0].date()} ~ {dfc.index[-1].date()}")
    return dfc, dfo


def load_index():
    """从 QMT 加载沪深300指数"""
    try:
        hs300 = load_qmt_benchmark("000300.SH")
        return hs300
    except Exception:
        # fallback: 用 510300 ETF close 做代理
        etf_data = load_qmt_etfs(["sh510300"], verbose=False)
        if "sh510300" in etf_data:
            return etf_data["sh510300"]
        return None


def load_stocks_qmt(limit=None):
    """从 QMT 加载个股数据（用于 V4.1 因子投票）"""
    print("从 QMT 加载个股数据...")
    stocks = load_qmt_stocks(min_history=250)
    if limit:
        stocks = dict(list(stocks.items())[:limit])
    return stocks


# ── 主流程 ──────────────────────────────────────────────
print("=" * 55)
print("ETF-R1 QMT 版: V4 投票行业 ETF + SURGE 集中优化")
print("=" * 55)

dfc, dfo = load_panels()
idx = load_index()
rs = compute_regime(idx["close"])

cal = dfc.index[dfc.index >= START_DATE]
me = month_end_dates(cal)
extra, forced, d2t = build_surge(idx["close"], dfc, rs, cal, me)
rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
lc = dfc.notna().cumsum()
lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

# 行业映射（CSV 文件，不在 QMT 中）
ind_map = {}
mf = pd.read_csv(os.path.join(DATA_DIR, "industry_map_bs.csv"), dtype=str)
mf = mf[mf["industryClassification"] == "证监会行业分类"]
mf["bare"] = mf["code"].str[3:]
for _, r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"] != "":
        ind_map[r["bare"]] = str(r["industry"])

# 加载个股
sd = load_stocks_qmt()

# 选股函数
cache = {}

def sel(date):
    if date in cache:
        return cache[date]
    valid, votes = {}, {}
    for c, s in sd.items():
        if date in s.index and len(s[s.index <= date]) >= 250:
            valid[c] = s
    if not valid:
        result = [s for s in BROAD if lo(s, date)]
        cache[date] = result
        return result
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
        result = top[:2]
    else:
        result = top[:2]
        for g in ["sz159915", "sh588000"]:
            if lo(g, date) and g not in result:
                result.append(g)
        while len(result) < 4:
            for e, _ in ranked:
                if e not in result and lo(e, date):
                    result.append(e)
                    break
            else:
                break
    cache[date] = result[:4]
    return result[:4]

# 运行回测
print("运行回测...")
r = run_stock_backtest(dfc, rs, {}, top_n=4, verbose=False, execution="next_open",
                       stamp_duty=True, ffill_valuation=True, df_open=dfo,
                       rebalance_dates=rd, select_fn=sel, forced_regime=forced)
v = r["values"]["value"]
rt = v.pct_change().dropna()
yrs = len(rt) / 244
ann = (v.iloc[-1] / 1e6) ** (1 / yrs) - 1
dd = (v / v.cummax() - 1).min()
sh = rt.mean() / rt.std() * np.sqrt(244)

sub = {}
for a, b in [("2016", "2018"), ("2019", "2021"), ("2022", "2024"), ("2025", "2026")]:
    s = rt.loc[a:b]
    sub[a + "-" + b[2:]] = round(((1 + s).prod() ** (244 / len(s)) - 1) * 100, 1)
yearly = {int(k): round(v, 1) for k, v in
          (rt.groupby(rt.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)).items()}

print(f"\n{'='*55}")
print("ETF-R1 QMT版 最终定稿(V4信号 + SURGE优化)")
print(f"{'='*55}")
print(f"年化 {ann*100:.2f}%  回撤 {dd*100:.1f}%  夏普 {sh:.2f}  换手 {r['metrics']['annual_turnover_x']:.1f}x")
print(f"分时段: {sub}")
print(f"逐年: {yearly}")
print(f"\n普通RISKON: 2行业ETF(各25%)+创业板50(30%)+科创50(20%)")
print(f"SURGE期间: 纯2行业ETF(各50%, 集中优化)")
print(f"数据源: QMT qmt_qfq.db (ETF前复权) + qmt_full.db (个股)")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtests", "etf_r1_final_qmt.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"ann": round(ann*100, 2), "mdd": round(dd*100, 1), "sharpe": round(sh, 2),
               "turnover": round(r["metrics"]["annual_turnover_x"], 1), "sub": sub, "yearly": yearly},
              f, ensure_ascii=False, indent=2)
print(f"[saved] {out}")
