#!/usr/bin/env python3
"""
Amihud + Mom×R² 因子验证
在 QMT 个股数据上计算截面表现，对比现有因子的预测力。

指标:
- Rank IC (Spearman): 因子值与未来收益的秩相关系数
- 多空收益差: Top 20% vs Bottom 20% 的未来收益差
- t 统计量: IC 的统计显著性
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from collections import defaultdict

def spearman_r(x, y):
    """手动实现 Spearman rank correlation (避免 scipy 依赖)"""
    n = len(x)
    if n < 3:
        return 0, 1
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    # 处理 ties: 平均秩
    for arr, ranks in [(x, rx), (y, ry)]:
        unique = np.unique(arr)
        for u in unique:
            mask = arr == u
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
    r = np.corrcoef(rx, ry)[0, 1]
    # t-statistic
    if abs(r) >= 1.0:
        t = np.inf if r > 0 else -np.inf
    else:
        t = r * np.sqrt((n - 2) / (1 - r * r))
    return r, t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 数据加载 ─────────────────────────────────────────────
import sqlite3
DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_full.db")

def load_stocks():
    """加载 QMT 个股数据"""
    conn = sqlite3.connect(DB)
    all_codes = [row[0] for row in conn.execute(
        "SELECT DISTINCT code FROM daily WHERE close > 0 AND volume > 0"
    ).fetchall()]
    # 只保留 A 股
    stocks = [c for c in all_codes if c.split(".")[0].startswith(
        ("000","001","002","003","300","301","600","601","603","605","688"))]
    print(f"加载 {len(stocks)} 只个股...")

    result = {}
    for code in stocks:
        bare = code.split(".")[0]
        sql = f"SELECT date, open, close, volume, amount FROM daily WHERE code='{code}' ORDER BY date"
        df = pd.read_sql(sql, conn, index_col='date', parse_dates=['date'])
        if len(df) >= 500:
            result[bare] = df
    conn.close()
    print(f"  {len(result)} 只满足≥500天")
    return result

# ── 因子计算 ─────────────────────────────────────────────

def compute_amihud(df, date, window=63):
    """Amihud 非流动性: mean(|return|/amount) over window"""
    if date not in df.index:
        return None
    idx = df.index.get_loc(date)
    if idx < window:
        return None
    sub = df.iloc[idx-window+1:idx+1]
    ret = sub['close'].pct_change().dropna().abs()
    amt = sub['amount'].iloc[-len(ret):]
    # 避免 amount=0
    valid = amt > 0
    if valid.sum() < 10:
        return None
    ratios = ret[valid] / amt[valid]
    return ratios.mean()

def compute_mom_r2(df, date, window=34):
    """Mom×R²: log-linear regression slope * 250 × R²"""
    if date not in df.index:
        return None
    idx = df.index.get_loc(date)
    if idx < window:
        return None
    sub = df.iloc[idx-window:idx]  # exclude date itself
    y = np.log(sub['close'].values)
    x = np.arange(len(y))
    if len(y) < 10:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    annual_ret = np.exp(slope * 250) - 1
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return annual_ret * r2

def compute_current_factors(df, date):
    """现有因子体系"""
    if date not in df.index:
        return {}
    idx = df.index.get_loc(date)
    if idx < 250:
        return {}
    closes = df['close']

    res = {}

    # low_vol: 1/63d vol
    ret_63 = closes.pct_change().dropna().iloc[-63:]
    if len(ret_63) >= 20:
        vol = ret_63.std() * np.sqrt(252)
        res['low_vol'] = 1.0 / (vol + 0.01)

    # momentum_6m: 6m return / 6m vol
    if len(closes) >= 126:
        ret_6m = closes.iloc[-1] / closes.iloc[-126] - 1
        vol_6m = closes.pct_change().dropna().iloc[-126:].std()
        if vol_6m and vol_6m > 0:
            res['mom_6m'] = ret_6m / vol_6m

    # quality: 250d return (floor 0)
    if len(closes) >= 250:
        long_ret = closes.iloc[-1] / closes.iloc[-250] - 1
        res['quality'] = max(0, long_ret) * 2

    return res

# ── 未来收益 ─────────────────────────────────────────────

def forward_return(df, date, horizon=20):
    """horizon 天后的收益率"""
    if date not in df.index:
        return None
    idx = df.index.get_loc(date)
    if idx + horizon >= len(df):
        return None
    return df['close'].iloc[idx + horizon] / df['close'].iloc[idx] - 1

# ── 验证主循环 ───────────────────────────────────────────

def validate(sd, years_back=5, step_months=3):
    """在每月末截面上计算因子的 Rank IC 和多空收益"""
    # 取所有股票的日期并集
    all_dates = sorted(set().union(*[set(s.index) for s in sd.values()]))
    all_dates = [d for d in all_dates if d.year >= 2021]  # 最近5年

    # 每月末
    test_dates = []
    for d in all_dates:
        if d.day >= 25:  # 月末附近
            test_dates.append(d)
    # 去重：每月只取最后一个
    monthly = []
    last_m = None
    for d in test_dates:
        m = (d.year, d.month)
        if m != last_m:
            monthly.append(d)
            last_m = m
    # 每隔 step_months 取一个
    test_dates = monthly[::step_months]

    print(f"测试日期: {len(test_dates)} 个 (每月取1, 间隔{step_months}月)")

    results = {name: {'ic': [], 'long_ret': [], 'short_ret': [], 'spread': []}
               for name in ['amihud', 'mom_r2', 'low_vol', 'mom_6m', 'quality']}

    for i, date in enumerate(test_dates):
        # 计算所有股票的因子值
        factor_vals = {name: {} for name in results}
        fwd_rets = {}

        for code, df in sd.items():
            fr = forward_return(df, date, horizon=20)
            if fr is None:
                continue
            fwd_rets[code] = fr

            # Amihud
            av = compute_amihud(df, date)
            if av is not None: factor_vals['amihud'][code] = av

            # Mom×R²
            mv = compute_mom_r2(df, date)
            if mv is not None: factor_vals['mom_r2'][code] = mv

            # Current factors
            cf = compute_current_factors(df, date)
            if 'low_vol' in cf: factor_vals['low_vol'][code] = cf['low_vol']
            if 'mom_6m' in cf: factor_vals['mom_6m'][code] = cf['mom_6m']
            if 'quality' in cf: factor_vals['quality'][code] = cf['quality']

        if len(fwd_rets) < 50:
            continue

        # 计算每个因子的 Rank IC 和多空收益
        for name in results:
            fv = factor_vals[name]
            if len(fv) < 30:
                continue

            # 交集: 有因子值也有未来收益的股票
            common = set(fv.keys()) & set(fwd_rets.keys())
            codes = list(common)
            if len(codes) < 30:
                continue

            f_vec = np.array([fv[c] for c in codes])
            r_vec = np.array([fwd_rets[c] for c in codes])

            # Rank IC (Spearman)
            ic, _ = spearman_r(f_vec, r_vec)
            if np.isfinite(ic):
                results[name]['ic'].append(ic)

            # 多空收益 (等分5组)
            n = len(codes)
            sorted_idx = np.argsort(f_vec)
            # 对于 amihud，高值=高非流动性=预期高收益 (正向)
            # 对于 low_vol，高值=低波动=预期高收益 (正向)
            # 对于 mom_r2/mom_6m/quality，高值=强趋势=预期高收益 (正向)
            # 所以都是高分组做多、低分组做空
            q_size = max(1, n // 5)
            long_idx = sorted_idx[-q_size:]
            short_idx = sorted_idx[:q_size]
            long_ret = np.mean(r_vec[long_idx])
            short_ret = np.mean(r_vec[short_idx])
            results[name]['long_ret'].append(long_ret)
            results[name]['short_ret'].append(short_ret)
            results[name]['spread'].append(long_ret - short_ret)

        if (i+1) % 5 == 0:
            print(f"  [{i+1}/{len(test_dates)}] {date.date()}  samples: {len(fwd_rets)} stocks")

    return results

# ── 报告 ─────────────────────────────────────────────────

def report(results):
    print(f"\n{'='*70}")
    print(f"因子验证报告 (2021-2026, 月频截面, 20日未来收益)")
    print(f"{'='*70}")

    for name in ['amihud', 'mom_r2', 'low_vol', 'mom_6m', 'quality']:
        r = results[name]
        n = len(r['ic'])
        if n < 5:
            print(f"\n  {name}: 样本不足 ({n})")
            continue

        ic_arr = np.array(r['ic'])
        spread_arr = np.array(r['spread'])

        mean_ic = np.mean(ic_arr)
        ic_t = mean_ic / (ic_arr.std() / np.sqrt(n)) if ic_arr.std() > 0 else 0
        ic_pos = np.mean(ic_arr > 0)
        mean_spread = np.mean(spread_arr) * 100
        spread_pos = np.mean(spread_arr > 0)

        print(f"\n  {'='*50}")
        print(f"  {name}")
        print(f"  {'='*50}")
        print(f"  测试月数: {n}")
        print(f"  Rank IC 均值: {mean_ic:.4f}  (t={ic_t:.2f})")
        print(f"  IC > 0 比例: {ic_pos:.1%}")
        print(f"  多空收益差: {mean_spread:.2f}% (20日)")
        print(f"  多空为正比例: {spread_pos:.1%}")

    # 汇总对比表
    print(f"\n{'='*70}")
    print(f"对比汇总")
    print(f"{'='*70}")
    print(f"  {'因子':<12} {'IC均值':>8} {'t值':>6} {'IC>0%':>7} {'多空差':>8} {'多空>0%':>8}")
    print(f"  {'-'*50}")
    for name in ['amihud', 'mom_r2', 'low_vol', 'mom_6m', 'quality']:
        r = results[name]
        n = len(r['ic'])
        if n < 5:
            continue
        ic_arr = np.array(r['ic'])
        spread_arr = np.array(r['spread'])
        mean_ic = np.mean(ic_arr)
        ic_t = mean_ic / (ic_arr.std() / np.sqrt(n)) if ic_arr.std() > 0 else 0
        print(f"  {name:<12} {mean_ic:>8.4f} {ic_t:>6.2f} {np.mean(ic_arr>0):>7.1%} "
              f"{np.mean(spread_arr)*100:>8.2f}% {np.mean(spread_arr>0):>8.1%}")

if __name__ == '__main__':
    sd = load_stocks()
    results = validate(sd, years_back=5, step_months=3)
    report(results)

    out = os.path.join(os.path.dirname(__file__), "backtests", "factor_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    clean = {k: {kk: [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in vv]
                 for kk, vv in v.items()}
             for k, v in results.items()}
    with open(out, "w") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")
