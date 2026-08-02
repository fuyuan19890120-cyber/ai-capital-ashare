#!/usr/bin/env python3
"""
ETF-R4 策略回测

R4 改进 (vs R3):
  1. SURGE 实时化: 每日监控 breadth, 触发即调仓 (R3仅在月末检查)
  2. KDJ K>80 中途出场: 防守期日内出场 (R3拿满全月)
  3. Regime 升级: 8因子版 + tanh乘数10→5 扩大NEUTRAL

策略核心 (继承R3):
  Regime >= 0.50 或 SURGE → 进攻: MomR²(85%)+Amihud(15%) Z-score复合
  Regime < 0.50            → 防守: KDJ优先 → Amihud补位 → 分级底仓

对比实验:
  (A) R3 baseline:  2因子regime(tanh×10), 月末SURGE, 无KDJ出场, 无每日信号
  (B) R4a SURGE实时: 同上 + SURGE每日检查
  (C) R4b +KDJ出场:  同上 + KDJ K>80中途出场
  (D) R4c +8因子:   8因子regime + SURGE每日 + KDJ出场
  (E) R4 完整版:    2因子regime(tanh×5) + SURGE每日 + KDJ出场
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, sqlite3
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.etf_backtest_engine import run_etf_backtest, ETF_COST

# ═══════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════
DB_PATH = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
START_DATE = "2016-01-01"
END_DATE = "2026-07-31"
INITIAL_CAPITAL = 1_000_000
COST_RATE = 0.0006
MOMR2_MAX_WIN = 50; MOMR2_MIN_WIN = 15; MOMR2_MID_WIN = 15
MOMR2_VL = 0.15; MOMR2_VH = 0.40
AMIHUD_W = 0.15; MOMR2_W = 0.85
SECTOR_PCT = 0.50

SURGE_LOCK_DAYS = 14
SURGE_DD_MELT = 0.08
BREADTH_ETFS = ["510300.SH", "510500.SH", "159915.SZ"]
KDJ_K_OVERBOUGHT = 80

# ETF池 (43只 = R3同款)
BROAD = ["510300.SH", "510500.SH", "159915.SZ", "588000.SH"]
DEFENSE = ["511010.SH", "518880.SH", "511880.SH"]
SECTOR_ETFS = [
    "512880.SH","512800.SH","512070.SH","512000.SH","512010.SH","512170.SH","159992.SZ",
    "159928.SZ","512690.SH","512480.SH","159995.SZ","512760.SH","512660.SH","512710.SH",
    "515030.SH","515790.SH","516160.SH","512400.SH","515210.SH","159870.SZ",
    "512200.SH","516750.SH","512980.SH","159869.SZ","159939.SZ","515000.SH","512720.SH",
    "515050.SH","515880.SH","159967.SZ","562500.SH","159819.SZ","159611.SZ","512580.SH",
    "516670.SH","159865.SZ",
]
HIGH_BETA = {"512880.SH","512760.SH","159995.SZ","512660.SH","515050.SH",
             "515790.SH","516160.SH","159915.SZ"}
ALL_ETFS = BROAD + DEFENSE + SECTOR_ETFS


# ═══════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════
def load_qmt_data():
    con = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?'] * len(ALL_ETFS))
    df = pd.read_sql_query(
        f"SELECT code, date, open, high, low, close, volume FROM daily "
        f"WHERE code IN ({placeholders}) AND date >= '20150101' ORDER BY code, date",
        con, params=ALL_ETFS)
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def build_panels(df):
    close_panel = df.pivot(index='date', columns='code', values='close').ffill(limit=5)
    open_panel = df.pivot(index='date', columns='code', values='open').ffill(limit=5)
    high_panel = df.pivot(index='date', columns='code', values='high').ffill(limit=5)
    low_panel = df.pivot(index='date', columns='code', values='low').ffill(limit=5)
    vol_panel = df.pivot(index='date', columns='code', values='volume').ffill(limit=5)
    return close_panel, open_panel, high_panel, low_panel, vol_panel


# ═══════════════════════════════════════════════
# Regime
# ═══════════════════════════════════════════════
def compute_regime_2factor(close_panel, tanh_mult=5.0):
    """2因子 Regime: tanh(dev×multiplier) + SMA50金叉"""
    idx_col = "510300.SH"
    if idx_col not in close_panel.columns:
        idx_col = close_panel.columns[0]
    idx_close = close_panel[idx_col].dropna()
    sma50 = idx_close.rolling(50).mean()
    sma250 = idx_close.rolling(250).mean()
    dev = (idx_close - sma250) / sma250
    trend = 0.5 + 0.5 * np.tanh(dev * tanh_mult)
    gc = (sma50 > sma250).astype(float)
    regime = 0.6 * trend + 0.4 * gc
    return regime.dropna()


def compute_regime_8factor(close_panel, vol_panel=None):
    """8因子 Regime 向量化版 (weights from config.py REGIME_WEIGHTS)"""
    idx_col = "510300.SH"
    if idx_col not in close_panel.columns:
        idx_col = close_panel.columns[0]
    idx = close_panel[idx_col].dropna()
    common_idx = close_panel.index.intersection(idx.index)
    idx = idx.loc[common_idx]

    scores = pd.DataFrame(index=idx.index)
    scores['total'] = 0.0

    # F1: Price vs SMA200 (0.10)
    sma200 = idx.rolling(200).mean()
    scores['f1'] = (idx > sma200).astype(float)
    scores['total'] += 0.10 * scores['f1']

    # F2: SMA200 slope (0.08)
    sma200_change = sma200.pct_change(21)
    scores['f2'] = 0.5
    scores.loc[sma200_change > 0.005, 'f2'] = 1.0
    scores.loc[sma200_change < -0.005, 'f2'] = 0.0
    scores['total'] += 0.08 * scores['f2']

    # F3: Market Breadth (0.15)
    breadth_sum = pd.Series(0.0, index=idx.index)
    count = 0
    for code in close_panel.columns:
        c = close_panel[code]
        valid = c.dropna()
        if len(valid) >= 50:
            above = (valid > valid.rolling(50).mean()).astype(float)
            above = above.reindex(idx.index).fillna(0.5)
            breadth_sum = breadth_sum + above
            count += 1
    breadth_pct = breadth_sum / max(count, 1)
    scores['f3'] = 0.5
    scores.loc[breadth_pct > 0.5, 'f3'] = 1.0
    scores.loc[breadth_pct < 0.25, 'f3'] = 0.0
    scores['total'] += 0.15 * scores['f3']

    # F4: Volatility (0.15)
    rets = idx.pct_change()
    vol_21 = rets.rolling(21).std() * np.sqrt(252)
    vol_median_3y = vol_21.rolling(756).median()
    scores['f4'] = 0.5
    scores.loc[vol_21 < vol_median_3y, 'f4'] = 1.0
    scores.loc[vol_21 > vol_median_3y * 1.3, 'f4'] = 0.0
    scores['total'] += 0.15 * scores['f4'].fillna(0.5)

    # F5: Max DD 21d (0.15)
    dd_21 = idx.rolling(21).apply(lambda x: (x / x.expanding().max() - 1).min())
    scores['f5'] = 0.5
    scores.loc[dd_21 > -0.03, 'f5'] = 1.0
    scores.loc[dd_21 < -0.08, 'f5'] = 0.0
    scores['total'] += 0.15 * scores['f5'].fillna(0.5)

    # F6: Correlation (0.05) — skip for speed
    scores['f6'] = 0.5
    scores['total'] += 0.05 * scores['f6']

    # F7: Volume Momentum (0.15)
    if vol_panel is not None and idx_col in vol_panel.columns:
        vol = vol_panel[idx_col]
        vol_20 = vol.rolling(20).mean()
        vol_60 = vol.rolling(60).mean()
        ratio = vol_20 / vol_60
        scores['f7'] = 0.5
        scores.loc[ratio > 1.15, 'f7'] = 1.0
        scores.loc[ratio < 0.85, 'f7'] = 0.0
    else:
        scores['f7'] = 0.5
    scores['total'] += 0.15 * scores['f7'].fillna(0.5)

    # F8: Deviation from 60MA (0.17)
    ma60 = idx.rolling(60).mean()
    dev_60 = (idx - ma60) / ma60
    scores['f8'] = 0.5
    scores.loc[(dev_60 >= 0.05) & (dev_60 <= 0.15), 'f8'] = 1.0
    scores.loc[(dev_60 > 0.20) | (dev_60 < -0.10), 'f8'] = 0.0
    scores['total'] += 0.17 * scores['f8'].fillna(0.5)

    return scores['total'].dropna()


# ═══════════════════════════════════════════════
# 因子
# ═══════════════════════════════════════════════
def compute_momr2(close_panel, date, adaptive=False):
    """Mom×R² 因子"""
    idx_loc = close_panel.index.get_loc(date)
    scores = {}

    if adaptive and "510300.SH" in close_panel.columns:
        idx_c = close_panel["510300.SH"]
        rets = idx_c.pct_change().dropna()
        mkt_vol = 0.20
        if idx_loc < len(rets) and idx_loc >= 60:
            mkt_vol = rets.rolling(60).std().iloc[idx_loc] * np.sqrt(252)
        if np.isnan(mkt_vol): mkt_vol = 0.20
        w = int(np.clip(MOMR2_MAX_WIN - (mkt_vol - MOMR2_VL) / (MOMR2_VH - MOMR2_VL) * (MOMR2_MAX_WIN - MOMR2_MID_WIN),
                        MOMR2_MID_WIN, MOMR2_MAX_WIN))
    else:
        w = 30

    for code in close_panel.columns:
        c = close_panel[code].iloc[:idx_loc+1].dropna()
        if len(c) < w + 10:
            continue
        try:
            recent = c.iloc[-(w+1):]
            y = np.log(recent.values.astype(float))
            x = np.arange(len(y), dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            ann = np.exp(slope * 250) - 1
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            if ss_tot > 0:
                scores[code] = ann * (1 - ss_res / ss_tot)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return scores


def compute_amihud(close_panel, vol_panel, date):
    """Amihud 非流动性指标"""
    idx_loc = close_panel.index.get_loc(date)
    scores = {}
    for code in close_panel.columns:
        if code not in vol_panel.columns or idx_loc < 21:
            continue
        c = close_panel[code].iloc[idx_loc-20:idx_loc]
        v = vol_panel[code].iloc[idx_loc-20:idx_loc]
        if len(c) < 20:
            continue
        c_vals = c.values.astype(float)
        v_vals = v.values.astype(float)
        rets = np.diff(c_vals) / c_vals[:-1]
        valid = (v_vals[1:] > 0) & (c_vals[1:] > 0) & np.isfinite(rets)
        if valid.sum() >= 15:
            ratios = np.abs(rets[valid]) / (v_vals[1:][valid] * c_vals[1:][valid])
            scores[code] = np.mean(ratios) * 1e8
    return scores


def compute_kdj(close_panel, high_panel, low_panel, date):
    """计算所有ETF的KDJ指标"""
    idx_loc = close_panel.index.get_loc(date)
    result = {}
    for code in close_panel.columns:
        c = close_panel[code].iloc[:idx_loc+1]
        h = high_panel[code].iloc[:idx_loc+1]
        l = low_panel[code].iloc[:idx_loc+1]
        n = len(c)
        if n < 30:
            continue
        c_v = c.values.astype(float)
        h_v = h.values.astype(float)
        l_v = l.values.astype(float)
        rsv = np.zeros(n); k = np.ones(n)*50; d = np.ones(n)*50; j = np.zeros(n)
        for i in range(8, n):
            hh = h_v[i-8:i+1].max(); ll = l_v[i-8:i+1].min()
            rsv[i] = (c_v[i] - ll) / (hh - ll) * 100 if hh > ll else 50
        for i in range(1, n):
            k[i] = 2/3*k[i-1] + 1/3*rsv[i]
            d[i] = 2/3*d[i-1] + 1/3*k[i]
            j[i] = 3*k[i] - 2*d[i]

        golden = bool(k[-2] <= d[-2] and k[-1] > d[-1])
        dead = bool(k[-2] >= d[-2] and k[-1] < d[-1])

        s = pd.Series(c_v)
        ma3 = s.rolling(3).mean(); ma6 = s.rolling(6).mean()
        ma12 = s.rolling(12).mean(); ma24 = s.rolling(24).mean()
        bbi = (ma3.iloc[-1] + ma6.iloc[-1] + ma12.iloc[-1] + ma24.iloc[-1]) / 4
        ma20 = s.rolling(20).mean()

        result[code] = {
            "K": float(k[-1]), "D": float(d[-1]), "J": float(j[-1]),
            "golden": golden, "dead": dead,
            "j_oversold": j[-2] < 20,
            "above_bbi": c_v[-1] > bbi,
            "below_ma20": c_v[-1] < ma20.iloc[-1] if pd.notna(ma20.iloc[-1]) else False,
            "kdj_ent": golden and (j[-2] < 20) and (c_v[-1] > bbi),
            "kdj_ext": (k[-1] > KDJ_K_OVERBOUGHT) or dead or (c_v[-1] < ma20.iloc[-1] if pd.notna(ma20.iloc[-1]) else False),
        }
    return result


# ═══════════════════════════════════════════════
# 信号函数
# ═══════════════════════════════════════════════
def make_surge_daily_fn(close_panel, vol_panel):
    """SURGE 检查 — 对齐原始 R3: MomR²+Amihud 选全量行业池 Top-2, 50/50"""
    def check_surge(date, positions, context):
        idx_loc = close_panel.index.get_loc(date)
        if idx_loc < 50:
            return None
        breadth = 0
        for etf_code in BREADTH_ETFS:
            if etf_code in close_panel.columns:
                c = close_panel[etf_code].iloc[:idx_loc+1].dropna()
                if len(c) >= 50:
                    sma50 = c.rolling(50).mean()
                    if c.iloc[-1] > sma50.iloc[-1]:
                        breadth += 1
        regime_val = context.get("regime_val", 0.5)
        if breadth >= 2 and regime_val >= 0.15:
            # 对齐 R3: 全量行业池 × MomR²(85%)+Amihud(15%) Z-score → Top-2, 各50%
            mr2 = compute_momr2(close_panel, date, adaptive=False)
            amihud = compute_amihud(close_panel, vol_panel, date)
            def zscore(s):
                if not s: return {}
                vals = list(s.values()); mu, std = np.mean(vals), np.std(vals)
                if std == 0: return {c: 0 for c in s}
                return {c: (v - mu) / std for c, v in s.items()}
            z1 = zscore(mr2); z2 = zscore(amihud)
            all_codes = set(z1.keys()) | set(z2.keys())
            composite = {c: MOMR2_W * z1.get(c, 0) + AMIHUD_W * z2.get(c, 0) for c in all_codes}
            # 只在行业ETF中选
            sector_ranked = [(c, s) for c, s in sorted(composite.items(), key=lambda x: x[1], reverse=True)
                           if c in SECTOR_ETFS]
            top2 = sector_ranked[:2]
            if len(top2) >= 2:
                return {top2[0][0]: 0.50, top2[1][0]: 0.50}
        return None
    return check_surge


def make_kdj_exit_fn(close_panel, high_panel, low_panel):
    """KDJ 中途出场检查 (只计算持仓ETF, 不是全部43只)"""
    def check_kdj_exit(date, positions, context):
        regime_val = context.get("regime_val", 0.5)
        if regime_val >= 0.50 or not positions:
            return None
        # 只计算持仓ETF的KDJ
        exits = {}
        for code in positions:
            if code not in close_panel.columns:
                continue
            idx_loc = close_panel.index.get_loc(date)
            c = close_panel[code].iloc[:idx_loc+1]
            h = high_panel[code].iloc[:idx_loc+1]
            l = low_panel[code].iloc[:idx_loc+1]
            n = len(c)
            if n < 30:
                continue
            c_v = c.values.astype(float); h_v = h.values.astype(float); l_v = l.values.astype(float)
            rsv = np.zeros(n); k = np.ones(n)*50; d = np.ones(n)*50
            for i in range(8, n):
                hh = h_v[i-8:i+1].max(); ll = l_v[i-8:i+1].min()
                rsv[i] = (c_v[i] - ll) / (hh - ll) * 100 if hh > ll else 50
            for i in range(1, n):
                k[i] = 2/3*k[i-1] + 1/3*rsv[i]
                d[i] = 2/3*d[i-1] + 1/3*k[i]
            dead = bool(k[-2] >= d[-2] and k[-1] < d[-1])
            s = pd.Series(c_v)
            ma20 = s.rolling(20).mean().iloc[-1]
            kdj_ext = (k[-1] > KDJ_K_OVERBOUGHT) or dead or (c_v[-1] < ma20 if pd.notna(ma20) else False)
            if kdj_ext:
                exits[code] = 'exit'
        return exits if exits else None
    return check_kdj_exit


def make_monthly_select(close_panel, vol_panel, high_panel, low_panel, use_kdj_defense=True):
    """月末选股"""
    def monthly_select(date, regime_val, context):
        mr2 = compute_momr2(close_panel, date, adaptive=True)
        amihud = compute_amihud(close_panel, vol_panel, date)

        def zscore(s):
            if not s: return {}
            vals = list(s.values()); mu, std = np.mean(vals), np.std(vals)
            if std == 0: return {c: 0 for c in s}
            return {c: (v - mu) / std for c, v in s.items()}

        z1 = zscore(mr2); z2 = zscore(amihud)
        all_codes = set(z1.keys()) | set(z2.keys())
        composite = {c: MOMR2_W * z1.get(c, 0) + AMIHUD_W * z2.get(c, 0) for c in all_codes}
        ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)

        if regime_val >= 0.50:
            # 进攻模式
            sector_ranked = [(c, s) for c, s in ranked if c in SECTOR_ETFS]
            top2 = sector_ranked[:2]
            result = [(c, SECTOR_PCT / 2) for c, _ in top2]
            growth_codes = [c for c in ["159915.SZ", "588000.SH"] if c in close_panel.columns]
            for g in growth_codes[:2]:
                result.append((g, (1.0 - SECTOR_PCT) / 2))
            return result
        else:
            if not use_kdj_defense:
                defense_ok = [c for c in DEFENSE if c in close_panel.columns]
                if not defense_ok: defense_ok = ["511010.SH", "511880.SH"]
                w = 1.0 / len(defense_ok)
                return [(c, w) for c in defense_ok]

            kdj = compute_kdj(close_panel, high_panel, low_panel, date)
            kdj_codes = [c for c in kdj if kdj[c]["kdj_ent"] and
                        c in SECTOR_ETFS + ["159915.SZ", "588000.SH", "510300.SH", "510500.SH"]]

            if len(kdj_codes) >= 2:
                n = min(len(kdj_codes), 4); w = 1.0 / n
                return [(c, w) for c in kdj_codes[:n]]
            elif len(kdj_codes) == 1:
                for c, _ in ranked:
                    if c not in kdj_codes and c in SECTOR_ETFS:
                        kdj_codes.append(c)
                        if len(kdj_codes) >= 4: break
                n = len(kdj_codes); w = 1.0 / max(n, 1)
                return [(c, w) for c in kdj_codes[:4]]
            else:
                if regime_val >= 0.40: floor = 0.30
                elif regime_val >= 0.30: floor = 0.20
                elif regime_val >= 0.15: floor = 0.10
                else: floor = 0.0

                result = []
                if floor > 0:
                    broad_ok = [c for c in BROAD if c in close_panel.columns]
                    if len(broad_ok) >= 2:
                        result.append((broad_ok[0], floor / 2))
                        result.append((broad_ok[1], floor / 2))
                    elif broad_ok:
                        result.append((broad_ok[0], floor))

                defense_ok = [c for c in DEFENSE if c in close_panel.columns]
                remain = 1.0 - floor
                n_def = max(len(defense_ok), 1)
                for d in defense_ok[:3]:
                    result.append((d, remain / min(n_def, 3)))
                return result
    return monthly_select


# ═══════════════════════════════════════════════
# 统计
# ═══════════════════════════════════════════════
def compute_metrics(result, label=""):
    m = result.get("metrics", {})
    if not m:
        print(f"  {label}: NO METRICS (error: {result.get('error', 'unknown')})")
        return None
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    for k in ["annual_return", "sharpe_ratio", "max_drawdown", "calmar_ratio",
              "annual_volatility", "annual_turnover", "total_return"]:
        v = m.get(k, 0)
        if k in ("annual_return", "max_drawdown", "annual_volatility", "total_return"):
            print(f"  {k}: {v:+.1f}%")
        else:
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    ar = m.get("annual_returns", {})
    if ar:
        print(f"  年度收益:")
        for yr in sorted(ar.keys()):
            v = ar[yr]
            flag = "🟢" if v > 0 else "🔴"
            print(f"    {yr}: {flag} {v:+.1f}%")
    return m


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ETF-R4 策略回测")
    print(f"  区间: {START_DATE} ~ {END_DATE}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n加载 QMT 前复权数据...")
    df_raw = load_qmt_data()
    close_panel, open_panel, high_panel, low_panel, vol_panel = build_panels(df_raw)
    print(f"  数据: {len(close_panel)} 天 × {len(close_panel.columns)} 只ETF")
    print(f"  范围: {close_panel.index[0].date()} ~ {close_panel.index[-1].date()}")

    # ── Regime ──
    print("\n计算 Regime 序列...")
    regime_2f_10x = compute_regime_2factor(close_panel, tanh_mult=10)
    regime_2f_5x = compute_regime_2factor(close_panel, tanh_mult=5)
    regime_8f = compute_regime_8factor(close_panel, vol_panel)

    for name, r in [("2f×10", regime_2f_10x), ("2f×5", regime_2f_5x), ("8f", regime_8f)]:
        pct_risk = (r >= 0.50).sum() / len(r) * 100 if len(r) > 0 else 0
        print(f"  Regime {name}: {len(r)} days, RISKON={pct_risk:.0f}%")

    # ── 信号 ──
    surge_fn = make_surge_daily_fn(close_panel, vol_panel)
    kdj_exit_fn = make_kdj_exit_fn(close_panel, high_panel, low_panel)
    monthly_sel = make_monthly_select(close_panel, vol_panel, high_panel, low_panel, use_kdj_defense=True)
    monthly_sel_no_kdj = make_monthly_select(close_panel, vol_panel, high_panel, low_panel, use_kdj_defense=False)

    base_ctx = {"close_panel": close_panel, "open_panel": open_panel,
                "high_panel": high_panel, "low_panel": low_panel, "vol_panel": vol_panel}

    results = {}

    # ═══ (A) R3 baseline ═══
    print("\n>>> (A) R3 baseline (2f×10, no daily signals, no KDJ exit)...")
    r3 = run_etf_backtest(
        df_close=close_panel, df_open=open_panel,
        regime_series=regime_2f_10x,
        monthly_select_fn=monthly_sel_no_kdj,
        daily_signals=None,
        surge_config={"enabled": False},  # R3: 无 SURGE
        start_date=START_DATE, end_date=END_DATE,
        verbose=False, context=base_ctx,
    )
    results['R3_baseline'] = compute_metrics(r3, "(A) R3 baseline (2f×10)")

    # ═══ (B) R4a: SURGE daily ═══
    print("\n>>> (B) R4a: +SURGE daily...")
    surge_cfg = {"enabled": True, "lock_days": SURGE_LOCK_DAYS, "dd_melt": SURGE_DD_MELT}
    r4a = run_etf_backtest(
        df_close=close_panel, df_open=open_panel,
        regime_series=regime_2f_10x,
        monthly_select_fn=monthly_sel_no_kdj,
        daily_signals={"surge": surge_fn},
        surge_config=surge_cfg,
        start_date=START_DATE, end_date=END_DATE,
        verbose=False, context=base_ctx,
    )
    results['R4a_SURGE'] = compute_metrics(r4a, "(B) R4a +SURGE daily")

    # ═══ (C) R4b: +KDJ exit ═══
    print("\n>>> (C) R4b: +KDJ mid-month exit...")
    r4b = run_etf_backtest(
        df_close=close_panel, df_open=open_panel,
        regime_series=regime_2f_10x,
        monthly_select_fn=monthly_sel,
        daily_signals={"surge": surge_fn, "kdj_exit": kdj_exit_fn},
        surge_config=surge_cfg,
        start_date=START_DATE, end_date=END_DATE,
        verbose=False, context=base_ctx,
    )
    results['R4b_KDJ'] = compute_metrics(r4b, "(C) R4b +KDJ exit")

    # ═══ (D) R4c: 8f regime ═══
    print("\n>>> (D) R4c: 8-factor regime...")
    r4c = run_etf_backtest(
        df_close=close_panel, df_open=open_panel,
        regime_series=regime_8f,
        monthly_select_fn=monthly_sel,
        daily_signals={"surge": surge_fn, "kdj_exit": kdj_exit_fn},
        surge_config=surge_cfg,
        start_date=START_DATE, end_date=END_DATE,
        verbose=False, context=base_ctx,
    )
    results['R4c_8factor'] = compute_metrics(r4c, "(D) R4c 8f regime")

    # ═══ (E) R4 full: 2f×5 ═══
    print("\n>>> (E) R4 full: 2f×5 + SURGE + KDJ...")
    r4e = run_etf_backtest(
        df_close=close_panel, df_open=open_panel,
        regime_series=regime_2f_5x,
        monthly_select_fn=monthly_sel,
        daily_signals={"surge": surge_fn, "kdj_exit": kdj_exit_fn},
        surge_config=surge_cfg,
        start_date=START_DATE, end_date=END_DATE,
        verbose=False, context=base_ctx,
    )
    results['R4_full'] = compute_metrics(r4e, "(E) R4 full (2f×5)")

    # ═══ 汇总 ═══
    print(f"\n{'=' * 70}")
    print(f"  📊 策略对比汇总")
    print(f"{'=' * 70}")
    header = f"  {'指标':<16s}"
    for name in results:
        header += f" {name:>11s}"
    print(header)
    print(f"  {'─' * 70}")
    for key, label in [("annual_return", "年化收益%"), ("sharpe_ratio", "夏普比率"),
                        ("max_drawdown", "最大回撤%"), ("calmar_ratio", "卡玛比率"),
                        ("annual_volatility", "年化波动%")]:
        line = f"  {label:<16s}"
        for name in results:
            v = (results[name] or {}).get(key, 0) if results[name] else 0
            if key in ("annual_return", "max_drawdown", "annual_volatility"):
                line += f" {v:+10.1f}"
            else:
                line += f" {v:10.2f}"
        print(line)

    # ── SURGE 事件 ──
    for name, res in [("R3", r3), ("R4_full", r4e)]:
        se = res.get("surge_events", [])
        if se:
            total_pnl = sum(e.get("pnl_pct", 0) or 0 for e in se)
            print(f"\n  🚀 {name} SURGE: {len(se)} events, total PnL {total_pnl:+.1f}%")

    # ── 保存 ──
    out_dir = os.path.join(os.path.dirname(__file__), "backtests")
    os.makedirs(out_dir, exist_ok=True)
    vals = pd.DataFrame({
        "R3_baseline": r3["values"]["value"],
        "R4_full": r4e["values"]["value"],
    })
    vals.to_csv(os.path.join(out_dir, "etf_r4_comparison.csv"))
    print(f"\n💾 净值对比: backtests/etf_r4_comparison.csv")
    print(f"\n{'=' * 60}")
    print(f"  ✅ ETF-R4 回测完成")
    print(f"{'=' * 60}")
