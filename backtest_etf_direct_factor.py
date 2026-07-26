#!/usr/bin/env python3
"""
ETF 直接因子选股回测:
在 ETF K线上直接计算 Quality/Amihud/Reversal 因子，跳过股票投票→行业映射管线。
框架: V2 两档 + SURGE28d（网格搜索验证的最优框架）
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
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
    "sh512880":["证券","资本市场","货币金融"], "sh512800":["货币金融","金融服务"],
    "sh512010":["医药","制药","医疗","中药","生物制品"],
    "sz159928":["食品","饮料","酒","农牧","渔业","养殖","零售","批发","农副"],
    "sh512690":["酒","饮料"], "sh512480":["电子","计算机","通信"],
    "sz159995":["电子","计算机","通信"],
    "sh512660":["航天","航空","船舶","航海","兵装","军工","国防","卫星"],
    "sh515030":["汽车","新能源"], "sh515790":["光伏","电气机械","电源"],
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
    "sh512760":["芯片","半导体","集成电路"], "sh515050":["5G","通信设备","信息技术"],
    "sh516160":["新能源","碳中和","清洁能源"],
}
ALL_ETF_CODES = BROAD + DEFENSE + list(SECTOR_MAP.keys())
ETF_NAMES = {
    "sh510300":"沪深300","sh510500":"中证500","sz159915":"创业板50","sh588000":"科创50",
    "sh511010":"国债","sh518880":"黄金","sh511880":"货币",
    "sh512880":"证券","sh512800":"银行","sh512010":"医药","sz159928":"消费","sh512690":"酒",
    "sh512480":"半导体","sz159995":"芯片","sh512660":"军工","sh515030":"新能源车",
    "sh515790":"光伏","sh512400":"有色","sh512200":"房地产","sh512980":"传媒",
    "sz159939":"信息技术","sh515210":"钢铁","sz159870":"化工",
    "sz159967":"装备制造","sz159611":"电力","sh512580":"环保","sh516670":"运输",
    "sh512070":"非银金融","sz159865":"农业","sh516750":"建材","sh512760":"半导体2",
    "sh515050":"5G通信","sh516160":"新能源",
}

# ── ETF 级因子计算 ──────────────────────────────────────
def compute_etf_factors(etf_data, date):
    """在 ETF K线上直接计算因子得分。
    etf_data: {code: DataFrame(open/high/low/close/volume/amount)}
    返回: {code: score}
    """
    scores = {}
    for code, df in etf_data.items():
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 250:
            continue

        hist = df.iloc[:idx+1]
        closes = hist['close']

        # Quality: 250d return (positive only)
        if len(closes) >= 250:
            lr = closes.iloc[-1] / closes.iloc[-250] - 1
            q_score = max(0, lr)
        else:
            q_score = 0

        # Amihud: mean(|return|/amount)_{63d}
        sub_63 = hist.iloc[-63:]
        ret_abs = sub_63['close'].pct_change().dropna().abs()
        amt = sub_63['amount'].iloc[-len(ret_abs):]
        valid = amt > 0
        if valid.sum() >= 10:
            a_raw = (ret_abs[valid] / amt[valid]).mean()
            a_score = np.log1p(a_raw * 1e10)  # scale and log transform
        else:
            a_score = 0

        # Reversal: -Mom×R² on 34d
        sub_34 = hist.iloc[-35:-1]
        if len(sub_34) >= 10:
            y = np.log(sub_34['close'].values)
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            ar = np.exp(slope * 250) - 1
            y_pred = slope * x + intercept
            ss_r = np.sum((y - y_pred)**2)
            ss_t = np.sum((y - y.mean())**2)
            r2 = 1 - ss_r/ss_t if ss_t > 0 else 0
            r_score = -(ar * r2)  # 反转
        else:
            r_score = 0

        # 综合得分: Q100 (网格搜索最优)
        total = q_score
        # 备选权重也输出，方便对比
        # total = a_score * 1.0
        # total = q_score * 0.7 + a_score * 0.3

        scores[code] = total

    return scores


def select_etfs_direct(scores, listed_ok_fn, date, is_surge, top_n=4):
    """直接从 ETF 中选取得分最高的"""
    # 只保留已上市的 sector ETF
    candidates = {c: s for c, s in scores.items()
                  if c in SECTOR_MAP and listed_ok_fn(c)}
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    if is_surge:
        # SURGE: 纯 top-2 行业 ETF
        picks = [c for c, _ in ranked[:2]]
    else:
        # 普通 RISKON: top-2 行业 + 创业板50 + 科创50
        picks = [c for c, _ in ranked[:2]]
        for g in ["sz159915", "sh588000"]:
            if listed_ok_fn(g) and g not in picks:
                picks.append(g)
        # 补齐到 top_n
        for c, _ in ranked:
            if len(picks) >= top_n: break
            if c not in picks and listed_ok_fn(c):
                picks.append(c)

    # 兜底：至少选够 BROAD
    if len(picks) < 2:
        for b in BROAD:
            if listed_ok_fn(b) and b not in picks:
                picks.append(b)

    return picks[:top_n]


# ── 主流程 ──────────────────────────────────────────────
print("加载 ETF 数据...")
etf_data_raw = load_qmt_etfs(ALL_ETF_CODES, verbose=False)
print(f"  {len(etf_data_raw)} 只 ETF")

# 构建 panel
closes, opens = {}, {}
for sym in ALL_ETF_CODES:
    if sym in etf_data_raw:
        closes[sym] = etf_data_raw[sym]["close"]
        opens[sym] = etf_data_raw[sym]["open"]
dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
ends = [dfc[s].last_valid_index() for s in BROAD+DEFENSE if s in dfc.columns]
if ends: dfc = dfc.loc[:min(ends)].ffill(limit=5)
dfo = pd.DataFrame(opens).sort_index().reindex(dfc.index)
print(f"  panel: {len(dfc.columns)} 只, {len(dfc)} 天")

idx = load_qmt_benchmark("000300.SH")
rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
lc = dfc.notna().cumsum()

# ── 测试不同因子配置 ─────────────────────────────────────
CONFIGS = [
    ("Q100 (纯250d趋势)", lambda s: s),  # 直接用 quality score
    ("A100 (纯Amihud)", None),  # 需要重算
    ("Q70-A30", None),
    ("Q70-R30", None),
]

def run_with_config(label, weight_fn, regime_fn, lock_days):
    """用指定的 ETF 因子选股逻辑回测"""
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=lock_days)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
    rs = regime_fn()

    cache = {}
    def sel(date):
        if date in cache: return cache[date]
        scores = compute_etf_factors(etf_data_raw, date)
        lo = lambda s: (lc.at[date, s] if s in dfc.columns else 0) >= 20
        is_s = d2t.get(date) is not None
        # 应用权重变换
        weighted = {}
        if weight_fn:
            weighted = weight_fn(scores)
        else:
            weighted = scores
        picks = select_etfs_direct(weighted, lo, date, is_s)
        cache[date] = picks
        return picks

    rr = run_stock_backtest(
        dfc, rs, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=sel, forced_regime=forced)
    return rr

# 测试组合（直接用 Q100 跑三个变体框架）
print(f"\n{'='*60}")
print("ETF 直接因子选股回测")
print(f"{'='*60}")

# 简化：只测 Q100 在三个框架下的表现
frameworks = [
    ("V2 两档+SURGE28d",
     lambda rs=rs_raw: (lambda r=rs.copy(): ([r.__setitem__(d, 0.85 if r.loc[d] >= 0.70 else 0.20) for d in r.index], r)[1])(),
     28),
    ("四档+SURGE21d", lambda: rs_raw, 21),
    ("S42 SURGE独立42d",
     lambda: pd.Series(0.20, index=rs_raw.index),
     42),
]

results = []
for fw_name, regime_fn, lock in frameworks:
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=lock)
    trig_dates = [d for d,t in d2t.items() if d==t] if lock==42 else []
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra) | set(trig_dates)))
    rs_regime = regime_fn()
    forced_regime = forced if lock != 42 else {d: "RISKON" for d in d2t if d2t[d] is not None}

    # Q100: 直接用 quality score
    cache = {}
    def sel(date):
        if date in cache: return cache[date]
        scores = compute_etf_factors(etf_data_raw, date)
        lo = lambda s: (lc.at[date, s] if s in dfc.columns else 0) >= 20
        is_s = d2t.get(date) is not None
        picks = select_etfs_direct(scores, lo, date, is_s)
        cache[date] = picks
        return picks

    print(f"\n  {fw_name} ...", end=" ", flush=True)
    rr = run_stock_backtest(
        dfc, rs_regime, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=sel, forced_regime=forced_regime)

    v = rr["values"]["value"]; rt = v.pct_change().dropna()
    yrs = len(rt)/244; ann = (v.iloc[-1]/1e6)**(1/yrs)-1
    dd = (v/v.cummax()-1).min(); sh = rt.mean()/rt.std()*np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]
    yearly = {int(k):round(v,1) for k,v in (rt.groupby(rt.index.year).apply(lambda x: ((1+x).prod()-1)*100)).items()}

    print(f"年化 {ann*100:.2f}% 回撤 {dd*100:.1f}% 夏普 {sh:.2f} 换手 {t:.1f}x")
    print(f"    逐年: {yearly}")

    results.append({"framework": fw_name, "factor": "Q100",
                    "ann": round(ann*100,2), "mdd": round(dd*100,1),
                    "sharpe": round(sh,2), "turnover": round(t,1), "yearly": yearly})

# ── 对比 ─────────────────────────────────────────────────
print(f"\n{'='*80}")
print("对比: ETF直接因子 vs 股票投票映射 (均用Q100, 同框架)")
print(f"{'='*80}")
print(f"  {'方法/框架':<25} {'年化':>7} {'回撤':>7} {'夏普':>6} {'换手':>6}")
print(f"  {'-'*55}")
# 股票投票的参考值 (从网格搜索)
print(f"  {'股票投票 V2+28d (现行)':<25} {'14.89%':>7} {'-23.5%':>7} {'0.87':>6} {'6.7x':>6}")
print(f"  {'股票投票 V2+28d (Q100)':<25} {'15.38%':>7} {'-23.7%':>7} {'0.88':>6} {'6.1x':>6}")
for r in results:
    print(f"  {'ETF直接 V2+28d (Q100)':<25} {r['ann']:>6.2f}% {r['mdd']:>6.1f}% {r['sharpe']:>5.2f} {r['turnover']:>5.1f}x")

out = os.path.join(os.path.dirname(__file__), "backtests", "etf_direct_factor.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n[saved] {out}")
