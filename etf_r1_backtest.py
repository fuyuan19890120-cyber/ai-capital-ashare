#!/usr/bin/env python3
"""
ETF-R1: V4.1 信号栈驱动 ETF 组合 — 预注册试验(规格见 reports/etf_r1_prereg.md, 运行前锁定)

变体: E0 纯宽基 | E1 高盈利行业 | E2 SURGE窗口动量行业 | E3 组合 | S1/S2 Top-N敏感性
用法: venv/bin/python etf_r1_backtest.py [--variant all]
"""
import os, sys, json, argparse, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from run_final_backtest import load_index, compute_regime
from run_v5_backtest import month_end_dates, metrics_from_values

DATA = os.path.expanduser("~/ai-capital-ashare/data")
BROAD = ["sh510300", "sh510500", "sz159915", "sh588000"]
DEFENSE = ["sh511010", "sh518880", "sh511880"]

# 行业池: ETF → 东财行业关键词(fin_*.csv 行业列子串匹配); 含输家行业(预注册)
SECTOR_MAP = {  # 复审修订版: 对齐东财细分行业标签, 16只全部可命中
    "sh512880": ["证券"], "sh512800": ["银行"],
    "sh512010": ["化学制药", "中药", "生物制品", "医疗", "医药"],
    "sz159928": ["食品", "饮料", "白酒", "酿酒", "农牧", "养殖"],
    "sh512690": ["白酒", "酿酒"], "sh512480": ["半导体"],
    "sz159995": ["半导体", "元件"],
    "sh512660": ["航天", "航空", "船舶", "航海", "兵装", "军工"],
    "sh515030": ["汽车", "乘用车", "商用车", "电池"],
    "sh515790": ["光伏", "电源设备"],
    "sh512400": ["工业金属", "小金属", "贵金属", "能源金属", "有色"],
    "sh512200": ["房地产"],
    "sh512980": ["传媒", "游戏", "影视", "广告", "出版", "文化"],
    "sz159939": ["软件", "互联网", "通信", "计算机", "IT"],
    "sh515210": ["钢"],
    "sz159870": ["化学原料", "化学制品", "化学纤维", "化纤", "塑料", "橡胶"],
}
MIN_LISTED = 120     # 上市满 120 交易日方可入选
MIN_MEMBERS = 10     # 行业盈利分至少 10 只成分股
TOPN = 3

# SURGE 参数(V4.1 正式口径, 勿改)
S_LOOKBACK, S_LOCK, S_S30, S_BASE = 15, 21, 0.70, 0.15


def _fix_splits(close, opn):
    """新浪不复权序列的份额拆分校正: |单日变动|>25% 视为折算(A股ETF真实涨跌上限±20%),
    将跳空前全部历史按跳空比例缩放(后复权式连续化)。open 同步缩放。"""
    c = close.copy(); o = opn.copy()
    r = c.ffill().pct_change()
    for d in r.index[r.abs() > 0.25]:
        i = c.index.get_loc(d)
        prev_idx = c.iloc[:i].last_valid_index()
        if prev_idx is None:
            continue
        f = c.loc[d] / c.loc[prev_idx]
        c.iloc[:i] = c.iloc[:i] * f
        o.iloc[:i] = o.iloc[:i] * f
    return c, o


def load_panels():
    """23 只 ETF 的 close/open 宽表(防御+宽基用旧缓存 qfq, 行业用新浪不复权+拆分校正)"""
    closes, opens = {}, {}
    for sym in BROAD + DEFENSE:
        p = os.path.join(DATA, f"etf_{sym[2:]}.csv")
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        closes[sym], opens[sym] = df["close"], df["open"]
    for p in glob.glob(os.path.join(DATA, "etf_sector", "*.csv")):
        sym = os.path.basename(p)[:-4]
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        closes[sym], opens[sym] = _fix_splits(df["close"], df["open"])
    dfc = pd.DataFrame(closes).sort_index().dropna(how="all")
    # 修复日历并集陷阱(复审修订: 先截断后ffill, 否则截断被ffill抵消成no-op):
    # ①样本终点=宽基+防御共同最后有效日 ②内部缺行 ffill(限5天, 引擎对ETF无停牌ffill保护)
    end = min(dfc[s].last_valid_index() for s in BROAD + DEFENSE)
    dfc = dfc.loc[:end].ffill(limit=5)
    dfo = pd.DataFrame(opens).sort_index().reindex(dfc.index)
    return dfc, dfo


def load_industry_profit():
    """季报 → 行业单季ROE中位数 → {avail_date: {etf: score}}(法定披露期对齐, 无前视)"""
    DL = {"0331": "-05-01", "0630": "-09-01", "0930": "-11-01"}
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "industry_fundamentals", "fin_*.csv"))):
        stat = os.path.basename(f)[4:12]
        df = pd.read_csv(f, dtype={"股票代码": str})
        df = df.rename(columns={"股票代码": "code", "所处行业": "ind", "净资产收益率": "roe"})
        df["roe"] = pd.to_numeric(df["roe"], errors="coerce")
        avail = (str(int(stat[:4]) + 1) + "-05-01") if stat[4:] == "1231" else (stat[:4] + DL[stat[4:]])
        df["code"] = df["code"].str.zfill(6)
        df = df[df["code"].str[:2].isin(["00", "30", "60", "68"])]  # 仅沪深A股, 剔新三板/北交所/B股
        rows.append(pd.DataFrame({"code": df["code"], "ind": df["ind"].astype(str),
                                  "roe": df["roe"], "stat": stat, "avail": avail}))
    q = pd.concat(rows, ignore_index=True).dropna(subset=["roe"])
    q = q.drop_duplicates(["code", "stat"], keep="last").sort_values(["code", "stat"])
    q["year"] = q["stat"].str[:4]
    q["single"] = q.groupby(["code", "year"])["roe"].diff().fillna(q["roe"])  # YTD→单季

    out = {}
    for (stat, avail), g in q.groupby(["stat", "avail"]):
        scores = {}
        for etf, kws in SECTOR_MAP.items():
            m = g[g["ind"].apply(lambda x: any(k in x for k in kws))]
            if len(m) >= MIN_MEMBERS:
                scores[etf] = float(m["single"].median())
        out.setdefault(avail, {})["scores"] = scores
        out[avail]["stat"] = stat
    return dict(sorted(out.items()))


def build_surge(index_close, etf_close, regime_series, cal, rebal):
    """与 run_v5_backtest.build_surge_lock 同一逻辑, 额外返回 锁内日期→触发日 映射"""
    sma30 = index_close.rolling(30).mean(); sma50 = index_close.rolling(50).mean()
    dev30 = (index_close - sma30) / sma30
    s30 = 0.6 * (0.5 + 0.5 * np.tanh(dev30 * 10)) + 0.4 * (sma50 > sma30).astype(float)
    breadth = pd.DataFrame({e: (etf_close[e] > etf_close[e].rolling(50).mean()).astype(float)
                            for e in ["sh510300", "sh510500", "sz159915"]}).mean(axis=1)
    s30, breadth = s30.reindex(cal), breadth.reindex(cal)
    base = regime_series.reindex(cal)
    trig = (base >= S_BASE) & (s30 >= S_S30) & (breadth > 2/3 + 1e-9) & (breadth.shift(S_LOOKBACK) < 1/3)

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
            lock_until = i + S_LOCK - 1
            cur_trig = d
            d2trig[d] = d
            forced[d] = 'RISKON'
            if last_regime != 'RISKON' and d not in rebal_set:
                extra.append(d)
            last_regime = 'RISKON'
    return pd.DatetimeIndex(extra), forced, d2trig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all')
    args = ap.parse_args()

    dfc, dfo = load_panels()
    index_df = load_index()
    regime_series = compute_regime(index_df['close'])
    cal = dfc.index[dfc.index >= START_DATE]
    me = month_end_dates(cal)
    extra, forced, d2trig = build_surge(index_df['close'], dfc, regime_series, cal, me)
    rd = pd.DatetimeIndex(sorted(set(me) | set(extra)))
    profit = load_industry_profit()
    profit_dates = sorted(profit.keys())

    listed_count = dfc.notna().cumsum()

    def listed_ok(sym, date):
        return sym in dfc.columns and listed_count.at[date, sym] >= MIN_LISTED

    def broad_leg(date):
        return [s for s in BROAD if listed_ok(s, date)]

    def profit_leg(date, topn=TOPN):
        ds = str(date.date())
        avail = [a for a in profit_dates if a <= ds]
        if not avail:
            return broad_leg(date)  # 无季报可得时回退宽基(仅样本最初)
        scores = profit[avail[-1]]["scores"]
        cand = {e: v for e, v in scores.items() if listed_ok(e, date)}
        if len(cand) < topn:
            return broad_leg(date)
        return sorted(cand, key=cand.get, reverse=True)[:topn]

    ret63 = dfc.pct_change()
    def momentum_leg(trig_date, topn=TOPN):
        """SURGE 触发日评估一次, 锁内沿用: 63日夏普动量 Top-N"""
        r = dfc.loc[:trig_date].iloc[-64:]
        if len(r) < 64:
            return broad_leg(trig_date)
        mom = r.iloc[-1] / r.iloc[0] - 1
        vol = ret63.loc[:trig_date].iloc[-63:].std()
        score = (mom / vol.replace(0, np.nan)).drop(labels=BROAD + DEFENSE, errors='ignore')
        score = score[[s for s in score.index if listed_ok(s, trig_date)]].dropna()
        if len(score) < topn:
            return broad_leg(trig_date)
        return list(score.sort_values(ascending=False).head(topn).index)

    def make_sel(normal, surge_mode, topn=TOPN):
        def sel(date):
            if date in d2trig and d2trig[date] is not None and surge_mode == 'momentum':
                return momentum_leg(d2trig[date], topn)
            if normal == 'profit':
                return profit_leg(date, topn)
            return broad_leg(date)
        return sel

    VAR = {
        'E0_broad':  make_sel('broad', 'none'),
        'E1_profit': make_sel('profit', 'none'),
        'E2_surgemom': make_sel('broad', 'momentum'),
        'E3_combo': make_sel('profit', 'momentum'),
        'S1_top2': make_sel('profit', 'momentum', topn=2),
        'S2_top4': make_sel('profit', 'momentum', topn=4),
    }

    results = {}
    for name, sel in VAR.items():
        if args.variant != 'all' and name not in args.variant.split(','):
            continue
        print(f"\n===== {name} =====", flush=True)
        r = run_stock_backtest(dfc, regime_series, {}, top_n=TOPN, verbose=False,
                               execution='next_open', stamp_duty=True, ffill_valuation=True,
                               df_open=dfo, rebalance_dates=rd, select_fn=sel,
                               forced_regime=forced)
        m = metrics_from_values(r['values'])
        m['turnover'] = round(r['metrics'].get('annual_turnover_x', np.nan), 1)
        ret = r['values']['value'].pct_change().dropna()
        m['yearly'] = {int(k): round(v, 1) for k, v in
                       (ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)).items()}
        results[name] = m
        print(f"  年化 {m['ann']}% 回撤 {m['mdd']}% 夏普 {m['sharpe']} 换手 {m['turnover']}x")
        print(f"  分时段 {m['sub']}")

    out = os.path.join(os.path.dirname(__file__), 'backtests', 'etf_r1_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        with open(out) as f:
            old = json.load(f)
        old.update(results); results = old
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == '__main__':
    main()
