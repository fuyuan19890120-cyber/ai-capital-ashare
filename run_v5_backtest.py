#!/usr/bin/env python3
"""
V5 回测: 预注册试验矩阵(见 reports/v4_improvement_roadmap.md)

变体(主试验):
  v4fixed      修复后 V4 基线(对照)
  v5           V5 因子重构, 月末调仓
  v5_tranche   V5 + 错峰双腿(15日/月末各半仓, NAV 均值)
  v5_ratchet   V5 + 周频只降不升回撤棘轮
  v5_full      V5 + 棘轮 + 条件式波动率目标
敏感性(V5 核心上单点扰动): top20 / rev30 / lowvol120 / capneutral

用法: venv/bin/python run_v5_backtest.py [--variant all]
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import START_DATE
from src.stock_backtest import run_stock_backtest
from src import factors_v5
from run_final_backtest import load_etf_prices, load_index, load_stocks, compute_regime

RATCHET_DD = 0.15        # 距52周高点回撤阈值
RATCHET_ENVELOPE = 0.97  # 年线包络带
CV_QUANTILE = 0.80       # 条件波动率目标: 最高五分位启用
LEVELS = ['RISKON', 'NEUTRAL', 'RISKOFF', 'CRISIS']
EQ_FRAC = {'RISKON': 0.95, 'NEUTRAL': 0.60, 'RISKOFF': 0.30, 'CRISIS': 0.0}


def regime_of(score):
    if score >= 0.70: return 'RISKON'
    if score >= 0.50: return 'NEUTRAL'
    if score >= 0.30: return 'RISKOFF'
    return 'CRISIS'


def month_end_dates(cal):
    return pd.DatetimeIndex(pd.Series(cal).groupby(pd.Series(cal).dt.to_period('M')).last())


def mid_month_dates(cal):
    """每月15日(含)前最后一个交易日"""
    s = pd.Series(cal)
    return pd.DatetimeIndex(s[s.dt.day <= 15].groupby(s[s.dt.day <= 15].dt.to_period('M')).last())


def build_ratchet(index_close, regime_series, cal, rebal_dates):
    """周频只降不升棘轮 → {执行日(触发次日): 目标权益占比}"""
    ma250 = index_close.rolling(250).mean()
    high52 = index_close.rolling(250, min_periods=120).max()
    cal_ser = pd.Series(range(len(cal)), index=cal)
    rebal_set = set(rebal_dates)
    exec_map = {}
    cur_level_idx = 0
    last_rebal_level = 0
    # 周五(或周内最后交易日)集合
    week_last = pd.Series(cal).groupby(pd.Series(cal).dt.to_period('W')).last()
    week_last = set(pd.DatetimeIndex(week_last))
    for d in cal:
        if d in rebal_set and d in regime_series.index:
            last_rebal_level = LEVELS.index(regime_of(float(regime_series.loc[d])))
            cur_level_idx = last_rebal_level
            continue
        if d in week_last and d in index_close.index and pd.notna(ma250.get(d, np.nan)):
            p = float(index_close.loc[d])
            trig = (p / float(high52.loc[d]) - 1 <= -RATCHET_DD) or (p < float(ma250.loc[d]) * RATCHET_ENVELOPE)
            if trig and cur_level_idx < len(LEVELS) - 1:
                cur_level_idx += 1
                i = cal_ser.get(d)
                if i is not None and i + 1 < len(cal):
                    exec_map[cal[i + 1]] = EQ_FRAC[LEVELS[cur_level_idx]]
    return exec_map


def build_cond_vol_scale(index_close, rebal_dates):
    """条件式波动率目标 → {调仓日: 权益系数}; 仅上月已实现波动处历史最高五分位时启用"""
    ret = index_close.pct_change()
    rv = ret.rolling(21).std() * np.sqrt(244)  # 上月已实现波动(年化)
    out = {}
    for d in rebal_dates:
        if d not in rv.index or pd.isna(rv.loc[d]):
            continue
        hist = rv.loc[:d].dropna()
        if len(hist) < 750:  # 至少3年历史再启用
            continue
        cur = float(rv.loc[d])
        if cur >= float(hist.quantile(CV_QUANTILE)):
            sigma_lt = float(hist.mean())
            out[d] = min(1.0, sigma_lt / cur)
    return out


SURGE_LOCK_DAYS = 21  # 锁定交易日数(修复了旧ETF回测里按月递减≈21个月的bug)


def build_surge_lock(index_close, etf_close, regime_series, cal, rebal_dates):
    """
    广度SMA30锁定进场(口径移植自实盘 src/signal_generator.py:_check_breadth_lock):
      触发(三条同时): ①3只权益ETF站上各自SMA50比例 15个交易日前<33% 且 当日>67%
                      ②SMA30制度分(金叉项用 sma50>sma30)≥0.70  ③SMA250制度分≥0.15
      触发后: 强制 RISKON 锁定 21 个交易日(锁内月末调仓也强制 RISKON), 期满恢复 SMA250。
      加速进场: 若触发时上一调仓档位非 RISKON, 触发日作为额外调仓日(T收盘信号, T+1开盘进场)。
      锁定期内忽略新触发(不展期, 与实盘无状态实现一致)。
    返回 (额外调仓日 DatetimeIndex, {调仓日: 'RISKON'})
    """
    eq_etfs = ['sh510300', 'sh510500', 'sz159915']
    sma30 = index_close.rolling(30).mean()
    sma50 = index_close.rolling(50).mean()
    dev30 = (index_close - sma30) / sma30
    s30 = 0.6 * (0.5 + 0.5 * np.tanh(dev30 * 10)) + 0.4 * (sma50 > sma30).astype(float)
    breadth = pd.DataFrame({
        e: (etf_close[e] > etf_close[e].rolling(50).mean()).astype(float)
        for e in eq_etfs if e in etf_close.columns}).mean(axis=1)
    s30 = s30.reindex(cal)
    breadth = breadth.reindex(cal)
    base = regime_series.reindex(cal)
    trig = (base >= 0.15) & (s30 >= 0.70) & (breadth > 2 / 3) & (breadth.shift(15) < 1 / 3)

    rebal_set = set(rebal_dates)
    extra, forced = [], {}
    lock_until = -1
    last_regime = 'NEUTRAL'
    for i, d in enumerate(cal):
        if i <= lock_until:
            if d in rebal_set:
                forced[d] = 'RISKON'
                last_regime = 'RISKON'
            continue
        if d in rebal_set and d in regime_series.index:
            last_regime = regime_of(float(regime_series.loc[d]))
        if bool(trig.get(d, False)):
            lock_until = i + SURGE_LOCK_DAYS - 1
            forced[d] = 'RISKON'
            if last_regime != 'RISKON' and d not in rebal_set:
                extra.append(d)
            last_regime = 'RISKON'
    return pd.DatetimeIndex(extra), forced


def metrics_from_values(dfv):
    r = dfv['value'].pct_change().dropna()
    nav = (1 + r).cumprod()
    yrs = len(r) / 244
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    mdd = (nav / nav.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(244) if r.std() > 0 else 0
    sub = {}
    for name, (a, b) in {'2016-18': ('2016', '2018'), '2019-21': ('2019', '2021'),
                         '2022-24': ('2022', '2024'), '2025-26': ('2025', '2026')}.items():
        seg = r.loc[a:b]
        if len(seg) > 60:
            sub[name] = round(((1 + seg).prod() ** (244 / len(seg)) - 1) * 100, 1)
    return {'ann': round(ann * 100, 2), 'mdd': round(mdd * 100, 1),
            'sharpe': round(sharpe, 2), 'sub': sub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all')
    args = ap.parse_args()

    df_close, df_open = load_etf_prices(True)
    index_df = load_index()
    stock_data = load_stocks(None)
    regime_series = compute_regime(index_df['close'])
    cal = df_close.index[df_close.index >= START_DATE]
    me_dates = month_end_dates(cal)
    mm_dates = mid_month_dates(cal)

    print(f"个股 {len(stock_data)} 只, 日历 {cal[0].date()}~{cal[-1].date()}")

    codes = list(stock_data)
    aux = factors_v5.load_aux_panels(codes, cal)
    n_aux = aux['turn'].notna().any().sum()
    print(f"辅助面板覆盖 {n_aux}/{len(codes)} 只")
    qroe = factors_v5.load_quarterly_roe()
    qroe = qroe[qroe['code'].isin(set(codes))]
    print(f"季报 ROE 记录 {len(qroe)} 条({qroe['code'].nunique()} 只)")

    def sf(**kw):
        return factors_v5.build_select_fn(stock_data, cal, aux, qroe, **kw)

    def run(select_fn=None, rebal=None, ratchet=False, condvol=False, top_n=15, surge=False):
        rd = rebal if rebal is not None else me_dates
        forced = None
        if surge:
            extra, forced = build_surge_lock(index_df['close'], df_close, regime_series, cal, rd)
            rd = pd.DatetimeIndex(sorted(set(rd) | set(extra)))
        dg = build_ratchet(index_df['close'], regime_series, cal, rd) if ratchet else None
        es = build_cond_vol_scale(index_df['close'], rd) if condvol else None
        r = run_stock_backtest(df_close, regime_series, stock_data, top_n=top_n, verbose=False,
                               execution='next_open', stamp_duty=True, ffill_valuation=True,
                               df_open=df_open, rebalance_dates=rd, select_fn=select_fn,
                               equity_scale=es, downgrade_exec=dg, forced_regime=forced)
        return r

    variants = {}
    def register(name, fn):
        if args.variant == 'all' or name in args.variant.split(','):
            variants[name] = fn

    register('v4fixed', lambda: run())
    # 预注册主口径: 市值中性化(roadmap"至少市值"; 复审 C1 纠正——此前误把中性版降级为敏感性)
    register('v5', lambda: run(select_fn=sf(cap_neutral=True)))
    register('v5_ratchet', lambda: run(select_fn=sf(cap_neutral=True), ratchet=True))
    register('v5_full', lambda: run(select_fn=sf(cap_neutral=True), ratchet=True, condvol=True))
    # 声明偏离变体: 不做市值中性(含小市值敞口, 单列供对照)
    register('v5_nocap', lambda: run(select_fn=sf(cap_neutral=False)))
    # 敏感性(在主口径上单点扰动)
    register('sens_top20', lambda: run(select_fn=sf(cap_neutral=True, top_n=20), top_n=20))
    register('sens_rev30', lambda: run(select_fn=sf(cap_neutral=True, reversal_pct=0.30)))
    register('sens_lowvol120', lambda: run(select_fn=sf(cap_neutral=True, low_vol_window=120)))
    # 广度SMA30锁定进场(用户既有设计, 移植实盘口径, 预注册两组)
    register('v4fixed_surge', lambda: run(surge=True))
    register('v5_surge', lambda: run(select_fn=sf(cap_neutral=True), surge=True))

    results = {}
    for name, fn in variants.items():
        print(f"\n===== {name} =====", flush=True)
        r = fn()
        m = metrics_from_values(r['values'])
        m['turnover'] = round(r['metrics'].get('annual_turnover_x', np.nan), 1)
        m['skipped'] = r['metrics'].get('skipped_trades')
        results[name] = m
        print(f"  年化 {m['ann']}% 回撤 {m['mdd']}% 夏普 {m['sharpe']} 换手 {m['turnover']}x")
        print(f"  分时段 {m['sub']}")
        if name == 'v5' and args.variant == 'all':
            # tranche: 两个独立半仓组合 = NAV 均值(复审 C6: 不能平均日收益, 那等于免费日频再平衡)
            r_mm = run(select_fn=sf(cap_neutral=True), rebal=mm_dates)
            nav_a = r['values']['value'] / r['values']['value'].iloc[0]
            nav_b = r_mm['values']['value'] / r_mm['values']['value'].iloc[0]
            mix = pd.DataFrame({'value': (nav_a + nav_b) / 2 * 1e6})
            mt = metrics_from_values(mix)
            mt['turnover'] = round((r['metrics']['annual_turnover_x'] + r_mm['metrics']['annual_turnover_x']) / 2, 1)
            results['v5_tranche'] = mt
            print(f"  [v5_tranche] 年化 {mt['ann']}% 回撤 {mt['mdd']}% 夏普 {mt['sharpe']}")
            print(f"  [v5_tranche] 分时段 {mt['sub']}")

    suffix = '' if args.variant == 'all' else '_' + args.variant.replace(',', '_')[:40]
    out = os.path.join(os.path.dirname(__file__), 'backtests', f'v5_results{suffix}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out}")


if __name__ == '__main__':
    main()
