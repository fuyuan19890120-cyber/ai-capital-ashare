# -*- coding: utf-8 -*-
"""
V5 因子模块(2026-07-18 路线图 P1+P2)

预注册因子与权重(来源: reports/v4_improvement_roadmap.md, 参数借文献不调参):
  低波250        25%  std(日收益,250) 低优
  流动性桶       25%  PMO=sum(turn,20)/sum(turn,250) 低优 + Amihud=mean(|ret|/amount,20) 高优, 桶内等权
  质量桶         25%  ROE水平 高优 + ΔROE(yoy) 高优 + ROE近8季std 低优, 桶内等权(缺项取已有均值)
  改造动量       15%  夏普动量 ret(T-120→T-20)/std(窗口日收益) 高优; MAX20 最高五分位者动量记中性0.5
  真价值EP       10%  1/peTTM 高优; peTTM<=0 或缺失记 0
硬过滤(选股前):
  isST==1 剔除; 近20日涨幅位于截面前25% 剔除(A股反转, 多头端作剔除项)
季报时点对齐: 东财季报缓存无披露日, 按法定披露期限保守对齐:
  Q1(0331)→当年05-01 / 中报(0630)→09-01 / Q3(0930)→11-01 / 年报(1231)→次年05-01
  (法定截止日 ≥ 实际披露日, 只会晚不会早, 无前视)
"""
import os
import glob

import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
AUX_DIR = os.path.join(DATA_DIR, "stock_aux")

V5_WEIGHTS = {"low_vol": 0.25, "liquidity": 0.25, "quality": 0.25, "momentum": 0.15, "ep": 0.10}
REVERSAL_EXCLUDE_PCT = 0.25   # 剔除近月涨幅前 25%
MAX20_NEUTRAL_PCT = 0.20      # MAX20 最高 20% 的动量记中性


# ============================================================
# 数据加载
# ============================================================

def load_aux_panels(codes, calendar):
    """从 data/stock_aux 构建 turn/peTTM/amount/isST 日频面板(对齐交易日历)"""
    turn, pe, amt, isst = {}, {}, {}, {}
    for code in codes:
        p = os.path.join(AUX_DIR, f"{code}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        for col in ("turn", "peTTM", "amount"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        turn[code] = df["turn"]
        pe[code] = df["peTTM"]
        amt[code] = df["amount"]
        isst[code] = pd.to_numeric(df["isST"], errors="coerce")
    def panel(d):
        return pd.DataFrame(d).reindex(calendar)
    return {
        "turn": panel(turn),
        "peTTM": panel(pe),
        "amount": panel(amt),
        "isST": panel(isst).ffill(),  # ST 状态跨停牌沿用
    }


def _deadline(stat_date):
    """法定披露截止日(保守对齐)"""
    y, md = stat_date[:4], stat_date[4:]
    return {"0331": f"{y}-05-01", "0630": f"{y}-09-01",
            "0930": f"{y}-11-01", "1231": f"{int(y) + 1}-05-01"}[md]


def load_quarterly_roe():
    """
    读取东财季报缓存 → 长表 [code, stat, avail, roe(YTD), single(单季)]
    ⚠️ 东财"净资产收益率"是年初至今累计(YTD): 同年内差分得单季值(复审 C2 修复);
    某年首个可得季度若非 Q1, 其 single 含多季累计, 噪声有限且方向无偏。
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "industry_fundamentals", "fin_*.csv"))):
        stat = os.path.basename(f)[4:12]
        df = pd.read_csv(f, dtype={"股票代码": str})
        df = df.rename(columns={"股票代码": "code", "净资产收益率": "roe"})
        df["roe"] = pd.to_numeric(df["roe"], errors="coerce")
        df = df.dropna(subset=["roe"])
        rows.append(pd.DataFrame({
            "code": df["code"].str.zfill(6), "stat": stat,
            "avail": pd.Timestamp(_deadline(stat)), "roe": df["roe"]}))
    q = pd.concat(rows, ignore_index=True)
    q = q.drop_duplicates(["code", "stat"], keep="last").sort_values(["code", "stat"])
    q["year"] = q["stat"].str[:4]
    q["single"] = q.groupby(["code", "year"])["roe"].diff()
    q["single"] = q["single"].fillna(q["roe"])  # 每年首个可得季度: YTD 即单季(Q1)
    return q


def quality_scores_at(qroe, date):
    """
    date 时点可得的质量桶原始值: {code: (ROE_TTM, ΔROE同季yoy, 8季单季std)}
    只用 avail_date <= date 的记录; 全部基于单季化 ROE(复审 C2 修复)。
    """
    avail = qroe[qroe["avail"] <= date]
    out = {}
    for code, g in avail.groupby("code"):
        g = g.sort_values("stat")
        if g.empty:
            continue
        singles = g["single"]
        roe_ttm = singles.iloc[-min(4, len(g)):].sum()
        last = g.iloc[-1]
        prev_stat = str(int(last["stat"][:4]) - 1) + last["stat"][4:]
        prev = g[g["stat"] == prev_stat]
        d_roe = (last["single"] - prev["single"].iloc[0]) if len(prev) else np.nan
        stab = singles.iloc[-8:].std() if len(g) >= 8 else np.nan
        out[code] = (roe_ttm, d_roe, stab)
    return out


# ============================================================
# 因子计算(截面, 单个调仓日)
# ============================================================

def _pct_rank(s, good_high):
    """截面百分位得分 0~1(1=最优)。good_high=True: 值越大得分越高; False: 值越小得分越高。
    pandas rank(ascending=True) 给最小值最低分, 故 good_high 直接对应 ascending。"""
    return s.rank(pct=True, ascending=good_high)


def build_select_fn(stock_data, calendar, aux, qroe, top_n=15,
                    low_vol_window=250, reversal_pct=REVERSAL_EXCLUDE_PCT,
                    cap_neutral=False, weights=None):
    """
    返回 select_fn(date) -> [code,...], 供回测引擎调用。
    所有面板一次性预计算, 每个调仓日只做截面切片。
    """
    w = weights or V5_WEIGHTS
    close = pd.DataFrame({c: sdf["close"] for c, sdf in stock_data.items()}).reindex(calendar)
    ret = close.pct_change()

    vol = ret.rolling(low_vol_window, min_periods=int(low_vol_window * 0.8)).std()
    ret20 = close / close.shift(20) - 1.0
    max20 = ret.rolling(20).max()
    # 夏普动量: T-120→T-20 收益 / 窗口日收益std
    mom_ret = close.shift(20) / close.shift(120) - 1.0
    mom_vol = ret.shift(20).rolling(100, min_periods=80).std()
    sharpe_mom = mom_ret / mom_vol.replace(0, np.nan)

    turn, amount, pe, isst = aux["turn"], aux["amount"], aux["peTTM"], aux["isST"]
    pmo = turn.rolling(20, min_periods=15).sum() / turn.rolling(250, min_periods=200).sum()
    # Amihud: 按预注册剔除涨跌停日(|ret|>=9.5% 近似, 复审 C3 修复)
    ret_nolimit = ret.where(ret.abs() < 0.095)
    amihud = (ret_nolimit.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    # 流通市值代理(市值中性化敏感性用): 成交额/换手率
    float_cap = amount / (turn.replace(0, np.nan) / 100.0)

    bars_count = close.notna().cumsum()

    def select_fn(date):
        if date not in close.index:
            return []
        # 可交易候选: 当日有bar 且 历史≥250根
        alive = close.loc[date].notna() & (bars_count.loc[date] >= 250)
        idx = alive[alive].index
        if len(idx) < top_n:
            return []
        # 硬过滤1: ST
        st_row = isst.loc[date].reindex(idx)
        idx = idx[(st_row != 1).fillna(True).values]
        # 硬过滤2: 近月涨幅前 reversal_pct 剔除(反转)
        r20 = ret20.loc[date].reindex(idx)
        cut = r20.quantile(1 - reversal_pct)
        idx = idx[(r20 < cut).fillna(False).values]
        if len(idx) < top_n:
            return []

        r_lowvol = _pct_rank(vol.loc[date].reindex(idx), good_high=False)      # 低波优
        r_pmo = _pct_rank(pmo.loc[date].reindex(idx), good_high=False)         # 低换手优
        r_amihud = _pct_rank(amihud.loc[date].reindex(idx), good_high=True)    # 高非流动性优
        r_liq = pd.concat([r_pmo, r_amihud], axis=1).mean(axis=1)

        qs = quality_scores_at(qroe, date)
        qdf = pd.DataFrame.from_dict(qs, orient="index",
                                     columns=["roe", "droe", "stab"]).reindex(idx)
        r_q = pd.concat([
            _pct_rank(qdf["roe"], good_high=True),
            _pct_rank(qdf["droe"], good_high=True),
            _pct_rank(qdf["stab"], good_high=False),
        ], axis=1).mean(axis=1)  # 缺项取已有均值

        r_mom = _pct_rank(sharpe_mom.loc[date].reindex(idx), good_high=True)
        m20 = max20.loc[date].reindex(idx)
        lottery = m20 >= m20.quantile(1 - MAX20_NEUTRAL_PCT)
        r_mom[lottery.fillna(False)] = 0.5  # 彩票股动量记中性

        pe_row = pe.loc[date].reindex(idx)
        ep = 1.0 / pe_row.where(pe_row > 0)
        r_ep = _pct_rank(ep, good_high=True).fillna(0.0)  # 亏损/缺失记 0

        comp = (w["low_vol"] * r_lowvol.fillna(0.5)
                + w["liquidity"] * r_liq.fillna(0.5)
                + w["quality"] * r_q.fillna(0.5)
                + w["momentum"] * r_mom.fillna(0.5)
                + w["ep"] * r_ep)

        if cap_neutral:  # 敏感性: 对 log流通市值截面回归取残差
            lc = np.log(float_cap.loc[date].reindex(idx))
            m = pd.concat([comp.rename("y"), lc.rename("x")], axis=1).dropna()
            if len(m) > 50:
                beta = np.polyfit(m["x"], m["y"], 1)
                resid = comp - (beta[0] * lc + beta[1])
                comp = resid.fillna(comp - comp.mean())

        return list(comp.sort_values(ascending=False).head(top_n).index)

    return select_fn
