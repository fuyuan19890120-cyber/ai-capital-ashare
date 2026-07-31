#!/usr/bin/env python3
"""
新因子体系网格搜索: Amihud × Reversal(-MomR²) × Quality
在 V2 两档+SURGE28d 框架下搜索最优权重。
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from collections import defaultdict
import time

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
DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")

# ── 新因子计算 ──────────────────────────────────────────
def compute_new_factors(stock_data, date, weights):
    """weights: (w_amihud, w_reversal, w_quality)"""
    w_a, w_r, w_q = weights
    scores = {}

    for code, df in stock_data.items():
        if date not in df.index:
            continue
        idx = df.index.get_loc(date)
        if idx < 250:
            continue

        hist = df.iloc[:idx+1]
        closes = hist['close']
        factor_vals = {}

        # Amihud: mean(|return|/amount)_{63d}
        sub_63 = hist.iloc[-63:]
        ret_abs = sub_63['close'].pct_change().dropna().abs()
        amt = sub_63['amount'].iloc[-len(ret_abs):]
        valid = amt > 0
        if valid.sum() >= 10:
            factor_vals['amihud'] = (ret_abs[valid] / amt[valid]).mean()
        else:
            factor_vals['amihud'] = 0

        # Reversal: -Mom×R² on 34d
        sub_34 = hist.iloc[-35:-1]  # exclude date itself
        if len(sub_34) >= 10:
            y = np.log(sub_34['close'].values)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            ar = np.exp(slope * 250) - 1
            y_pred = slope * x + np.polyfit(x, y, 1)[1]
            ss_r = np.sum((y - y_pred)**2); ss_t = np.sum((y - y.mean())**2)
            r2 = 1 - ss_r/ss_t if ss_t > 0 else 0
            mom_r2 = ar * r2
            factor_vals['reversal'] = -mom_r2  # 取反号: 短期动量→反转
        else:
            factor_vals['reversal'] = 0

        # Quality: 250d return (positive only)
        if len(closes) >= 250:
            lr = closes.iloc[-1] / closes.iloc[-250] - 1
            factor_vals['quality'] = max(0, lr)
        else:
            factor_vals['quality'] = 0

        # 合并得分（对数变换处理肥尾）
        total = 0
        for f, w in [('amihud', w_a), ('reversal', w_r), ('quality', w_q)]:
            v = factor_vals.get(f, 0)
            if f == 'amihud' and v > 0:
                v = np.log1p(v * 1e10)  # Amihud 值极小，需要放大后 log 变换
            elif f == 'reversal':
                v = v  # 可能负值也可能正值
            elif f == 'quality' and v > 0:
                v = np.log1p(v)
            total += v * w / 100.0

        scores[code] = total

    return scores


def select_top_stocks_new(scores, top_n=30):
    """选取得分最高的 top_n 只股票, 返回代码列表"""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [code for code, _ in ranked[:top_n]]


# ── 数据加载（只做一次）─────────────────────────────────
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

def load_index():
    try: return load_qmt_benchmark("000300.SH")
    except:
        etf_data = load_qmt_etfs(["sh510300"], verbose=False)
        return etf_data.get("sh510300")

dfc, dfo = load_panels()
idx = load_index(); rs_raw = compute_regime(idx["close"])
cal = dfc.index[dfc.index >= START_DATE]; me = month_end_dates(cal)
lc = dfc.notna().cumsum(); lo = lambda s, d: (lc.at[d, s] if s in dfc.columns else 0) >= 20

ind_map = {}
mf = pd.read_csv(os.path.join(DATA_DIR, "industry_map_bs.csv"), dtype=str)
mf = mf[mf["industryClassification"]=="证监会行业分类"]; mf["bare"] = mf["code"].str[3:]
for _, r in mf.iterrows():
    if pd.notna(r["industry"]) and r["industry"]!="": ind_map[r["bare"]] = str(r["industry"])

sd_raw = load_qmt_stocks(min_history=250)
print(f"个股: {len(sd_raw)} 只")

# ── 网格搜索 ────────────────────────────────────────────
# 测试的权重组合
COMBOS = [
    # 单因子基准
    (100, 0, 0, "A100"), (0, 100, 0, "R100"), (0, 0, 100, "Q100"),
    # Amihud 主导
    (70, 20, 10, "A70-R20-Q10"), (70, 10, 20, "A70-R10-Q20"),
    (60, 30, 10, "A60-R30-Q10"), (60, 20, 20, "A60-R20-Q20"),
    (50, 30, 20, "A50-R30-Q20"), (50, 20, 30, "A50-R20-Q30"),
    (40, 30, 30, "A40-R30-Q30"),
    # 反转主导
    (20, 50, 30, "A20-R50-Q30"), (30, 40, 30, "A30-R40-Q30"),
    (30, 50, 20, "A30-R50-Q20"), (20, 60, 20, "A20-R60-Q20"),
    (10, 70, 20, "A10-R70-Q20"),
    # 基准: 现行因子
    None,  # sentinel for "use original factors"
]

results = []

for combo in COMBOS:
    if combo is None:
        label = "ORIGINAL"
        weights = None
        print(f"\n{'='*50}\n[基准] 现行因子 (low_vol+mom_6m+quality)\n{'='*50}")
    else:
        w_a, w_r, w_q, label = combo
        weights = (w_a, w_r, w_q)
        print(f"\n{'='*50}\n{label} (A={w_a} R={w_r} Q={w_q})\n{'='*50}")

    t0 = time.time()

    # 构建 select_fn（根据权重使用新旧因子）
    if weights is None:
        # 使用原始因子
        from src.stock_data import compute_stock_factors, select_top_stocks as orig_top
        def make_sel():
            cache = {}
            def sel(date, d2t):
                if (date, id(d2t)) in cache: return cache[(date, id(d2t))]
                valid = {}
                for c, s in sd_raw.items():
                    if date in s.index and len(s[s.index <= date]) >= 250: valid[c] = s
                if not valid:
                    r = [s for s in BROAD if lo(s, date)]; cache[(date, id(d2t))] = r; return r
                scores = compute_stock_factors(valid, date)
                top30 = orig_top(scores, 30)
                votes = {}
                for code in top30[:30]:
                    ind = ind_map.get(code, "")
                    for etf, kws in SECTOR_MAP.items():
                        if any(k in ind for k in kws): votes[etf] = votes.get(etf,0)+1; break
                ranked = sorted(votes.items(), key=lambda x:x[1], reverse=True)
                top = [e for e,_ in ranked[:2] if lo(e, date)]
                gb = [s for s in ["sz159915","sh588000"] if lo(s, date)]
                while len(top)<2 and gb: top.append(gb.pop(0))
                if len(top)<2: top = [s for s in BROAD if lo(s, date)][:2]
                is_s = d2t.get(date) is not None
                if is_s: r = top[:2]
                else:
                    r = top[:2]
                    for g in ["sz159915","sh588000"]:
                        if lo(g, date) and g not in r: r.append(g)
                    while len(r)<4:
                        for e,_ in ranked:
                            if e not in r and lo(e, date): r.append(e); break
                        else: break
                cache[(date, id(d2t))] = r[:4]; return r[:4]
            return sel
    else:
        def make_sel():
            cache = {}
            def sel(date, d2t):
                if (date, id(d2t)) in cache: return cache[(date, id(d2t))]
                valid = {}
                for c, s in sd_raw.items():
                    if date in s.index and len(s[s.index <= date]) >= 250: valid[c] = s
                if not valid:
                    r = [s for s in BROAD if lo(s, date)]; cache[(date, id(d2t))] = r; return r
                scores = compute_new_factors(valid, date, weights)
                top30 = select_top_stocks_new(scores, 30)
                votes = {}
                for code in top30[:30]:
                    ind = ind_map.get(code, "")
                    for etf, kws in SECTOR_MAP.items():
                        if any(k in ind for k in kws): votes[etf] = votes.get(etf,0)+1; break
                ranked = sorted(votes.items(), key=lambda x:x[1], reverse=True)
                top = [e for e,_ in ranked[:2] if lo(e, date)]
                gb = [s for s in ["sz159915","sh588000"] if lo(s, date)]
                while len(top)<2 and gb: top.append(gb.pop(0))
                if len(top)<2: top = [s for s in BROAD if lo(s, date)][:2]
                is_s = d2t.get(date) is not None
                if is_s: r = top[:2]
                else:
                    r = top[:2]
                    for g in ["sz159915","sh588000"]:
                        if lo(g, date) and g not in r: r.append(g)
                    while len(r)<4:
                        for e,_ in ranked:
                            if e not in r and lo(e, date): r.append(e); break
                        else: break
                cache[(date, id(d2t))] = r[:4]; return r[:4]
            return sel

    # V2 两档 + SURGE28d（之前验证的最佳框架）
    rs_mod = rs_raw.copy()
    for d in rs_mod.index:
        rs_mod.loc[d] = 0.85 if rs_mod.loc[d] >= 0.70 else 0.20
    extra, forced, d2t = build_surge(idx["close"], dfc, rs_raw, cal, me, lock_days=28)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
    sf = make_sel()

    rr = run_stock_backtest(
        dfc, rs_mod, {}, top_n=4, verbose=False, execution="next_open",
        stamp_duty=True, ffill_valuation=True, df_open=dfo,
        rebalance_dates=rd, select_fn=lambda d: sf(d, d2t), forced_regime=forced)

    v = rr["values"]["value"]; rt = v.pct_change().dropna()
    yrs = len(rt)/244; ann = (v.iloc[-1]/1e6)**(1/yrs)-1
    dd = (v/v.cummax()-1).min(); sh = rt.mean()/rt.std()*np.sqrt(244)
    t = rr["metrics"]["annual_turnover_x"]

    elapsed = time.time() - t0
    print(f"  年化 {ann*100:.2f}%  回撤 {dd*100:.1f}%  夏普 {sh:.2f}  换手 {t:.1f}x  [{elapsed:.0f}s]")

    results.append({"label": label, "weights": f"A={weights[0]}_R={weights[1]}_Q={weights[2]}" if weights else "orig",
                    "ann": round(ann*100,2), "mdd": round(dd*100,1), "sharpe": round(sh,2),
                    "turnover": round(t,1), "time": round(elapsed)})

# ── 汇总排序 ─────────────────────────────────────────────
print(f"\n{'='*80}")
print("网格搜索结果（按年化收益排序）")
print(f"{'='*80}")
print(f"  {'组合':<20} {'年化':>7} {'回撤':>7} {'夏普':>6} {'换手':>6}")
print(f"  {'-'*50}")
for r in sorted(results, key=lambda x: x['ann'], reverse=True):
    print(f"  {r['label']:<20} {r['ann']:>6.2f}% {r['mdd']:>6.1f}% {r['sharpe']:>5.2f} {r['turnover']:>5.1f}x")

out = os.path.join(os.path.dirname(__file__), "backtests", "factor_grid_results.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n[saved] {out}")
