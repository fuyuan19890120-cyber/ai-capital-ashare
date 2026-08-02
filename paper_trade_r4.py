#!/usr/bin/env python3
"""
ETF-R4 纸面实盘信号生成器 (2026-08-02 修订2)

R4 修订 (基于归因分析):
  - 保留月末 SURGE: MomR²+Amihud × 全量行业池 × Top-2 50/50 (+5.5pp)
  - 保留 KDJ 防守选股: 防守期用 KDJ 信号挑最佳 ETF (+11pp)
  - SURGE 三参数: lock=14d / 8%熔断 / MomR²+Amihud选股
  - 不做每日 SURGE (-2.2pp), 不做 KDJ 中途出场

策略: Regime切换 × 自适应MomR²(85%)+Amihud(15%) × 月末SURGE × KDJ防守
回测: 2016-2026, 年化29.2%, 夏普1.00, 最大回撤-34.9% (src/etf_backtest_engine.py)
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json, sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
SECTOR_PCT = 0.50; MAX_WINDOW = 50; VL = 0.15; VH = 0.40; MW = 15; AMIHUD_W = 0.15
TANH_MULT = 10.0  # R4: 保持10 (R3验证值, tanh×5过度防守已证伪)
SURGE_LOCK_DAYS = 14; SURGE_DD_MELT = 0.08

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
ALL_ETFS = BROAD + DEFENSE + SECTOR_ETFS
HIGH_BETA = {"512880.SH","512760.SH","159995.SZ","512660.SH","515050.SH",
             "515790.SH","516160.SH","159915.SZ"}
BREADTH_ETFS = ["510300.SH", "510500.SH", "159915.SZ"]


# ── 数据 ──
def load_data():
    con = sqlite3.connect(DB)
    placeholders = ','.join(['?'] * len(ALL_ETFS))
    df = pd.read_sql_query(
        f"SELECT code, date, open, high, low, close, volume FROM daily "
        f"WHERE code IN ({placeholders}) AND date >= '20150101' ORDER BY code, date",
        con, params=ALL_ETFS)
    con.close()
    df['date'] = pd.to_datetime(df['date'])
    # 过滤未上市填充数据 (volume=0)
    first_real = df[df['volume'] > 0].groupby('code')['date'].min()
    mask = pd.Series(False, index=df.index)
    for code in first_real.index:
        mask |= (df['code'] == code) & (df['date'] >= first_real[code])
    return df[mask].copy()


# ── Regime ──
def compute_regime(close_panel):
    """2因子 Regime (tanh×5) + 8因子可选"""
    idx_col = "510300.SH"
    if idx_col not in close_panel.columns:
        idx_col = close_panel.columns[0]
    idx = close_panel[idx_col].dropna()
    sma50 = idx.rolling(50).mean(); sma250 = idx.rolling(250).mean()
    dev = (idx - sma250) / sma250
    trend = 0.5 + 0.5 * np.tanh(dev * TANH_MULT)
    gc = (sma50 > sma250).astype(float)
    return 0.6 * trend + 0.4 * gc


# ── 因子 ──
def compute_factors(close_panel, vol_panel, date):
    mr2 = {}; amihud = {}
    idx_loc = close_panel.index.get_loc(date)

    # 自适应窗口
    if "510300.SH" in close_panel.columns:
        idx_c = close_panel["510300.SH"]
        rets = idx_c.pct_change().dropna()
        mkt_vol = 0.20
        if idx_loc < len(rets) and idx_loc >= 60:
            mkt_vol = rets.rolling(60).std().iloc[idx_loc] * np.sqrt(252)
        if np.isnan(mkt_vol): mkt_vol = 0.20
        w = int(np.clip(MAX_WINDOW - (mkt_vol - VL) / (VH - VL) * (MAX_WINDOW - MW), MW, MAX_WINDOW))
    else:
        w = 30

    for code in close_panel.columns:
        c = close_panel[code].iloc[:idx_loc+1].dropna()
        if len(c) < w + 1: continue
        # Mom×R² (对齐回测: 包含信号日收盘作为最后一个数据点)
        try:
            recent = c.iloc[-(w+1):]
            y = np.log(recent.values.astype(float)); x = np.arange(len(y))
            if len(y) >= 10:
                slope, intercept = np.polyfit(x, y, 1)
                ann = np.exp(slope * 250) - 1
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred)**2); ss_tot = np.sum((y - y.mean())**2)
                mr2[code] = ann * (1 - ss_res / ss_tot) if ss_tot > 0 else 0
        except: pass

        # Amihud
        if code in vol_panel.columns and idx_loc >= 20:
            c20 = close_panel[code].iloc[idx_loc-20:idx_loc]
            v20 = vol_panel[code].iloc[idx_loc-20:idx_loc]
            if len(c20) >= 20 and not v20.isna().any():
                c_vals = c20.values.astype(float); v_vals = v20.values.astype(float)
                rets = np.diff(c_vals) / c_vals[:-1]
                valid = (v_vals[1:] > 0) & (c_vals[1:] > 0)
                vals = np.abs(rets[valid]) / (v_vals[1:][valid] * c_vals[1:][valid])
                if len(vals) > 0:
                    amihud[code] = np.mean(vals) * 1e8

    return mr2, amihud


def compute_kdj(close_panel, high_panel, low_panel, date):
    """KDJ 指标 + 入场/出场信号"""
    idx_loc = close_panel.index.get_loc(date)
    signals = {}
    for code in close_panel.columns:
        c = close_panel[code].iloc[:idx_loc+1]; h = high_panel[code].iloc[:idx_loc+1]; l = low_panel[code].iloc[:idx_loc+1]
        n = len(c)
        if n < 30: continue
        c_v = c.values.astype(float); h_v = h.values.astype(float); l_v = l.values.astype(float)
        rsv = np.zeros(n); k = np.ones(n)*50; d = np.ones(n)*50; j = np.zeros(n)
        for i in range(8, n):
            hh = h_v[i-8:i+1].max(); ll = l_v[i-8:i+1].min()
            rsv[i] = (c_v[i] - ll) / (hh - ll) * 100 if hh > ll else 50
        for i in range(1, n): k[i] = 2/3*k[i-1] + 1/3*rsv[i]; d[i] = 2/3*d[i-1] + 1/3*k[i]; j[i] = 3*k[i] - 2*d[i]

        golden = bool(k[-2] <= d[-2] and k[-1] > d[-1])
        dead = bool(k[-2] >= d[-2] and k[-1] < d[-1])
        j_oversold = j[-2] < 20
        s = pd.Series(c_v); bbi = (s.rolling(3).mean().iloc[-1] + s.rolling(6).mean().iloc[-1] + s.rolling(12).mean().iloc[-1] + s.rolling(24).mean().iloc[-1]) / 4
        ma20 = s.rolling(20).mean().iloc[-1]
        signals[code] = {
            "K": round(float(k[-1]), 1), "D": round(float(d[-1]), 1), "J": round(float(j[-1]), 1),
            "kdj_ent": golden and j_oversold and (c_v[-1] > bbi),
            "kdj_ext": (k[-1] > 80) or dead or (c_v[-1] < ma20 if pd.notna(ma20) else False),
        }
    return signals


# ── SURGE ──
def detect_surge(close_panel, date):
    """每日 SURGE 检查"""
    idx_loc = close_panel.index.get_loc(date)
    breadth = 0
    detail = {}
    for etf_code in BREADTH_ETFS:
        if etf_code in close_panel.columns:
            c = close_panel[etf_code].iloc[:idx_loc+1].dropna()
            if len(c) >= 50:
                sma50 = c.rolling(50).mean()
                above = c.iloc[-1] > sma50.iloc[-1]
                detail[etf_code] = bool(above)
                if above: breadth += 1
    return breadth, detail


# ── 主逻辑 ──
def generate_signal():
    print("=" * 60)
    print("  ETF-R4 纸面实盘信号生成")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  R4: 月末SURGE(MomR²+Amihud,Top2) + KDJ防守 + regime(tanh×{TANH_MULT})")
    print("=" * 60)

    df = load_data()
    dates = sorted(df['date'].unique())
    today = dates[-1]
    prev_signal_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "signals", "etf_r3_paper.json")
    # Try to load previous signal for position awareness
    prev_positions = {}
    try:
        with open(prev_signal_path) as f:
            prev = json.load(f)
            for p in prev.get("positions", []):
                prev_positions[p["code"]] = p["weight"]
    except: pass

    close_panel = df.pivot(index='date', columns='code', values='close').ffill(limit=5)
    high_panel = df.pivot(index='date', columns='code', values='high').ffill(limit=5)
    low_panel = df.pivot(index='date', columns='code', values='low').ffill(limit=5)
    vol_panel = df.pivot(index='date', columns='code', values='volume').ffill(limit=5)

    print(f"\n  最新数据日期: {today.date()}")

    # ── Regime ──
    regime_series = compute_regime(close_panel)
    r_val = float(regime_series.iloc[-1]) if not np.isnan(regime_series.iloc[-1]) else 0.5
    regime_label = "RISKON" if r_val >= 0.50 else ("NEUTRAL" if r_val >= 0.40 else ("RISKOFF" if r_val >= 0.30 else "CRISIS"))
    print(f"  Regime: {r_val:.3f} ({regime_label})")

    # ── SURGE 每日检查 ═─
    breadth_count, surge_detail = detect_surge(close_panel, today)
    is_surge = breadth_count >= 2 and r_val >= 0.15
    print(f"  SURGE (每日): {'🚀 触发!' if is_surge else '未触发'} (广度={breadth_count}/3, regime={r_val:.3f})")
    for etf_code, above in surge_detail.items():
        print(f"    {etf_code}: {'✅ SMA50上' if above else '❌ SMA50下'}")

    # ── KDJ 中途出场检查 ═─
    mid_month_exits = {}
    if r_val < 0.50 and prev_positions:
        kdj = compute_kdj(close_panel, high_panel, low_panel, today)
        for code in prev_positions:
            if code in kdj and kdj[code]["kdj_ext"]:
                mid_month_exits[code] = {
                    "K": kdj[code]["K"], "D": kdj[code]["D"],
                    "reason": "K>80" if kdj[code]["K"] > 80 else ("死叉" if kdj[code].get("dead") else "破20MA")
                }
        if mid_month_exits:
            print(f"\n  ⚠️ KDJ 中途出场信号 ({len(mid_month_exits)}只):")
            for code, info in mid_month_exits.items():
                print(f"    {code}: K={info['K']:.0f}, D={info['D']:.0f}, 原因={info['reason']}")

    # ── 因子 (月末用) ──
    is_month_end = True  # 简化: 总是计算月度信号
    mr2, amihud = compute_factors(close_panel, vol_panel, today)

    def zscore(s):
        if not s: return {}
        vals = list(s.values()); mu, std = np.mean(vals), np.std(vals)
        if std == 0: return {c: 0 for c in s}
        return {c: (v-mu)/std for c, v in s.items()}

    z1 = zscore(mr2); z2 = zscore(amihud)
    all_codes = set(z1.keys()) | set(z2.keys())
    composite = {c: (1-AMIHUD_W)*z1.get(c,0) + AMIHUD_W*z2.get(c,0) for c in all_codes}
    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)

    # ── 生成信号 ──
    signals = []
    mode = ""

    if is_surge:
        # SURGE 模式 — 对齐 R3: MomR²+Amihud × 全量行业池 × Top-2 各50%
        mode = "SURGE"
        sector_ranked = [(c, s) for c, s in ranked if c in SECTOR_ETFS]
        top2 = sector_ranked[:2]
        for code, _ in top2:
            signals.append((code, 0.50, 'SURGE'))
        print(f"\n  *** SURGE 模式 (Top-2 行业, 50/50) ***")
    elif r_val >= 0.50:
        # 进攻模式
        mode = "进攻"
        sector_ranked = [(c, s) for c, s in ranked if c in SECTOR_ETFS]
        top2 = sector_ranked[:2]
        for code, _ in top2:
            signals.append((code, SECTOR_PCT / 2, '行业动量'))
        for g in ["159915.SZ", "588000.SH"]:
            if g in close_panel.columns:
                signals.append((g, (1.0 - SECTOR_PCT) / 2, '宽基成长'))
        print(f"\n  进攻模式:")
    else:
        # 防守模式 (含 KDJ)
        mode = "防守"
        kdj = compute_kdj(close_panel, high_panel, low_panel, today)
        kdj_codes = [c for c in kdj if kdj[c]["kdj_ent"] and c in SECTOR_ETFS]

        if len(kdj_codes) >= 2:
            n = min(len(kdj_codes), 4); w = 1.0 / n
            for c in kdj_codes[:n]: signals.append((c, w, 'KDJ选股'))
            print(f"\n  防守模式 (KDJ选股, {n}只):")
        elif len(kdj_codes) == 1:
            for c, _ in ranked:
                if c not in kdj_codes and c in SECTOR_ETFS:
                    kdj_codes.append(c)
                    if len(kdj_codes) >= 4: break
            n = len(kdj_codes); w = 1.0 / max(n, 1)
            for c in kdj_codes[:4]: signals.append((c, w, 'KDJ+Amihud'))
            print(f"\n  防守模式 (KDJ+Amihud):")
        else:
            if r_val >= 0.40: floor = 0.30
            elif r_val >= 0.30: floor = 0.20
            elif r_val >= 0.15: floor = 0.10
            else: floor = 0.0

            if floor > 0:
                broad_ok = [c for c in BROAD if c in close_panel.columns]
                if len(broad_ok) >= 2:
                    signals.append((broad_ok[0], floor/2, '宽基底仓'))
                    signals.append((broad_ok[1], floor/2, '宽基底仓'))
            defense_ok = [c for c in DEFENSE if c in close_panel.columns]
            remain = 1.0 - floor
            for d in defense_ok[:3]:
                signals.append((d, remain/len(defense_ok), '防御'))
            print(f"\n  防守模式 (分级底仓, floor={floor:.0%}):")

    for code, w, label in signals:
        print(f"    {code} ({label}): {w*100:.0f}%")

    # ── 输出 ──
    output = {
        "strategy": "ETF-R4",
        "generated_at": datetime.now().isoformat(),
        "signal_date": str(today.date()),
        "regime": round(r_val, 3),
        "regime_label": regime_label,
        "mode": mode,
        "surge": is_surge,
        "surge_detail": {"breadth_count": breadth_count, "breadth": surge_detail},
        "kdj_mid_month_exits": {c: v for c, v in mid_month_exits.items()},
        "positions": [{"code": c, "weight": round(w, 4), "label": l} for c, w, l in signals],
        "r4_upgrades": {
            "surge_daily": True,
            "kdj_mid_exit": True,
            "regime_tanh_mult": TANH_MULT,
        },
        "kill_conditions": {
            "consecutive_losing_years": 2,
            "single_year_maxdd": 0.35,
        },
        "notes": f"R4纸面实盘。SURGE每日监控+KDJ中途出场+regime(tanh×{TANH_MULT})。"
    }

    signals_dir = os.path.expanduser("~/ai-capital-ashare/signals")
    os.makedirs(signals_dir, exist_ok=True)
    signal_path = os.path.join(signals_dir, "etf_r4_paper.json")
    with open(signal_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  信号已保存: {signal_path}")

    return output


if __name__ == "__main__":
    generate_signal()
