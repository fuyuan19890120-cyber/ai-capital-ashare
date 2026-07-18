#!/usr/bin/env python3
"""
P4: 无幸存者偏差重跑(investment_data qlib 数据, 含全部退市股 + CSI300 时点成分)

一次性解决审计 S1/S2 中无法用旧数据修复的两个问题:
  - 幸存者偏差: 6119 只含退市股(旧缓存 1154 只零退市)
  - 成分股前视: csi300.txt 自带时点成分区间(旧版用"今天的成分"回测 2015)

预注册宇宙定义(近似 V4 的"CSI300+创业板+科创50"口径):
  调仓日 d 的候选 = CSI300 时点成分(covering d) ∪ 创业板全体(300/301) ∪ 科创板全体(688/689)
  且 存续中(instrument区间含d) 且 历史≥250根bar。北交所剔除。
  注: 创业板指/科创50 历史成分不可免费获得, 用全板块近似(池更大, 选股强度略稀释, 如实声明)。

价格: qlib close/open = 原始价×factor(前复权到2000基准), 收益率/因子无污染;
      手数取整基于重定基价格, 对百万级资金失真可忽略。
变体: p4_v4fixed(对照 11.72%) / p4_v4fixed_surge(对照 15.89%)
用法: venv/bin/python run_p4_backtest.py [--variant p4_v4fixed,p4_v4fixed_surge]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src.stock_data import compute_stock_factors, select_top_stocks
from run_final_backtest import load_etf_prices, load_index
from run_v5_backtest import month_end_dates, metrics_from_values, build_surge_lock, build_ratchet

QLIB = os.path.expanduser("~/ai-capital-ashare/data/qlib_cn/qlib_bin")


# ============================================================
# qlib bin 读取(不依赖 pyqlib)
# ============================================================

def load_calendar():
    return pd.DatetimeIndex([l.strip() for l in open(os.path.join(QLIB, "calendars", "day.txt"))])


def read_field(sym_dir, field, cal):
    a = np.fromfile(os.path.join(sym_dir, f"{field}.day.bin"), dtype="<f4")
    start = int(a[0])
    v = a[1:]
    return pd.Series(v, index=cal[start:start + len(v)], dtype="float64")


def load_instruments(fname):
    """返回 {qlib_sym: [(start,end),...]}(csi300 有多段)"""
    out = {}
    with open(os.path.join(QLIB, "instruments", fname)) as f:
        for line in f:
            sym, s, e = line.strip().split("\t")
            out.setdefault(sym, []).append((pd.Timestamp(s), pd.Timestamp(e)))
    return out


def build_p4_stock_data(cal):
    """加载宇宙内全部股票的 open/close; 返回 (stock_data{bare_code: df}, membership 函数)"""
    all_ins = load_instruments("all.txt")
    csi300 = load_instruments("csi300.txt")

    def is_board(sym):
        return sym[2:5] in ("300", "301") and sym.startswith("SZ") or \
               sym[2:5] in ("688", "689") and sym.startswith("SH")

    want = set(csi300) | {s for s in all_ins if is_board(s)}
    want = {s for s in want if not s.startswith("BJ")}

    stock_data, meta = {}, {}
    miss = 0
    for sym in sorted(want):
        d = os.path.join(QLIB, "features", sym.lower())
        if not os.path.isdir(d):
            miss += 1
            continue
        try:
            close = read_field(d, "close", cal)
            opn = read_field(d, "open", cal)
        except Exception:
            miss += 1
            continue
        bare = sym[2:]
        df = pd.DataFrame({"open": opn, "close": close}).dropna(how="all")
        if len(df) < 60:
            continue
        stock_data[bare] = df
        meta[bare] = {
            "alive": all_ins.get(sym, []),
            "csi300": csi300.get(sym, []),
            "board": is_board(sym),
        }
    print(f"[universe] 候选 {len(want)} 只, 载入 {len(stock_data)} 只(缺数据 {miss}); "
          f"其中曾属CSI300 {sum(1 for m in meta.values() if m['csi300'])} 只, "
          f"板块股 {sum(1 for m in meta.values() if m['board'])} 只")

    def member_at(code, d):
        m = meta[code]
        alive = any(s <= d <= e for s, e in m["alive"])
        if not alive:
            return False
        if any(s <= d <= e for s, e in m["csi300"]):
            return True
        return m["board"]

    return stock_data, member_at


def make_select_fn(stock_data, member_at, top_n=15):
    """P4 版选股: 时点成分过滤 + 原版 V4 因子(与 v4fixed 完全同一因子代码)"""
    def select_fn(date):
        valid = {}
        for code, sdf in stock_data.items():
            if date in sdf.index and member_at(code, date):
                valid[code] = sdf
        if not valid:
            return []
        scores = compute_stock_factors(valid, date)  # 内部自带 ≥250bar 检查
        return select_top_stocks(scores, top_n)
    return select_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="p4_v4fixed,p4_v4fixed_surge")
    args = ap.parse_args()

    def compute_regime_w(close, slow=250):
        """基础制度分, slow=慢线窗口(默认250, OAT: 120/180/200)"""
        sma50 = close.rolling(50).mean()
        sma_s = close.rolling(slow).mean()
        scores = pd.Series(index=close.index, dtype=float)
        for i in range(max(252, slow + 2), len(close)):
            dev = (close.iloc[i] - sma_s.iloc[i]) / sma_s.iloc[i]
            trend = 0.5 + 0.5 * np.tanh(dev * 10)
            golden = 1.0 if sma50.iloc[i] > sma_s.iloc[i] else 0.0
            scores.iloc[i] = 0.6 * trend + 0.4 * golden
        return scores.dropna()

    cal_q = load_calendar()
    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    regime_series = compute_regime_w(index_df["close"])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = month_end_dates(cal)

    stock_data, member_at = build_p4_stock_data(cal_q)
    sel = make_select_fn(stock_data, member_at)
    # 归因变体宇宙: oldpool=旧1141只幸存池(验证数据源一致性); csi300=纯时点CSI300
    old_codes = {f[:-4] for f in os.listdir(os.path.expanduser("~/ai-capital-ashare/data/stocks"))
                 if f.endswith(".csv")}
    sel_oldpool = make_select_fn({c: d for c, d in stock_data.items() if c in old_codes},
                                 lambda code, d: True)
    csi300_meta = load_instruments("csi300.txt")
    def member_csi300(code, d):
        segs = csi300_meta.get("SH" + code, []) or csi300_meta.get("SZ" + code, [])
        return any(s <= d <= e for s, e in segs)
    sel_csi300 = make_select_fn(stock_data, member_csi300)
    SELECTORS = {"p4_v4fixed": sel, "p4_v4fixed_surge": sel,
                 "p4_oldpool": sel_oldpool, "p4_csi300": sel_csi300,
                 "p4_csi300_surge": sel_csi300, "p4_csi300_combo": sel_csi300,
                 "p4_csi300_ratchet": sel_csi300, "p4_csi300_surge_ratchet": sel_csi300,
                 "p4_csi300_sma120": sel_csi300, "p4_csi300_sma180": sel_csi300,
                 "p4_csi300_sma200": sel_csi300}
    # _combo = SURGE 候选参数(回看20d + s30阈值0.80), 在可信宇宙上做伪样本外验证
    SURGE_KW = {"p4_csi300_combo": {"lookback": 20, "s30_min": 0.80}}
    # 宇宙规模抽样
    for y in ["2016-06-30", "2020-06-30", "2025-06-30"]:
        d = cal[cal.get_indexer([pd.Timestamp(y)], method="nearest")[0]]
        n = sum(1 for c in stock_data if d in stock_data[c].index and member_at(c, d))
        print(f"  {d.date()} 在市候选: {n} 只")

    results = {}
    for name in args.variant.split(","):
        rs_use = regime_series
        if "_sma" in name:  # 基础窗口 OAT: p4_csi300_sma120/180/200
            w = int(name.split("_sma")[1])
            rs_use = compute_regime_w(index_df["close"], slow=w)
        surge = "_surge" in name or name.endswith("_combo")
        rd = me_dates
        forced = None
        if surge:
            extra, forced = build_surge_lock(index_df["close"], df_close, rs_use, cal, rd,
                                             **SURGE_KW.get(name, {}))
            rd = pd.DatetimeIndex(sorted(set(rd) | set(extra)))
        dg = build_ratchet(index_df["close"], rs_use, cal, rd) if "_ratchet" in name else None
        print(f"\n===== {name} =====", flush=True)
        r = run_stock_backtest(df_close, rs_use, stock_data, top_n=15, verbose=False,
                               execution="next_open", stamp_duty=True, ffill_valuation=True,
                               df_open=df_open, rebalance_dates=rd,
                               select_fn=SELECTORS.get(name, sel),
                               forced_regime=forced, downgrade_exec=dg)
        m = metrics_from_values(r["values"])
        m["turnover"] = round(r["metrics"].get("annual_turnover_x", np.nan), 1)
        ret = r["values"]["value"].pct_change().dropna()
        m["yearly"] = {int(k): round(v, 1) for k, v in
                       (ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)).items()}
        results[name] = m
        print(f"  年化 {m['ann']}% 回撤 {m['mdd']}% 夏普 {m['sharpe']} 换手 {m['turnover']}x")
        print(f"  分时段 {m['sub']}")
        print(f"  逐年 {m['yearly']}")

    out = os.path.join(os.path.dirname(__file__), "backtests", "p4_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):  # 增量合并, 分批跑变体不互相覆盖
        with open(out) as f:
            old = json.load(f)
        old.update(results)
        results = old
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
