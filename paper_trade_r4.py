#!/usr/bin/env python3
"""
ETF-R4 纸面实盘信号生成器 (2026-08-02 修订3·审查修复)

R4 策略 (回测验证):
  - 月末 SURGE: breadth≥2/3且r≥0.30 → MomR²+Amihud全量行业池Top-2各50%
  - KDJ 防守选股: 防守期(0.15≤r<0.50)用KDJ信号挑行业ETF, 三阶段递补
  - CRISIS熔断: r<0.15 → 国债+黄金+货币等权
  - 不做每日SURGE, 不做KDJ中途出场

信号生成器行为: 由 run_monthly_r4.py 在「月末最后一个交易日」调用
注意: 本脚本不判断月末, 月末判断在 run_monthly_r4.py 的 is_last_trading_day()

策略: Regime(tanh×10) × MomR²(70%)+Amihud(30%) × 月末SURGE(r≥0.30) × KDJ防守
回测: 2016-2026, 年化35.1%, 夏普1.22, 回撤-35.4% (src/etf_backtest_engine.py)
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json, sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
if "--tushare" in sys.argv:
    DB = os.path.expanduser("~/ai-capital-ashare/data/etf_tushare.db")  # tushare pct_chg 重构前复权
SECTOR_PCT = 0.50; MAX_WINDOW = 50; VL = 0.15; VH = 0.40; MW = 15; AMIHUD_W = 0.30  # 2026-08-06: 70/30
TANH_MULT = 10.0  # R4: 保持10 (R3验证值, tanh×5过度防守已证伪)
# 注: 回测有「锁仓14天 + 8%回撤熔断」机制, 实盘用 if/elif(is_surge优先)天然实现锁仓:
#   SURGE触发则Top-2持有到下个月末(月末间隔~21天>14天), 不被regime覆盖。
#   8%熔断需每日监控回撤, 而每日SURGE已证伪(daily模式25.6% vs monthly 36.5%), 故实盘未实现熔断。
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
def compute_factors(close_panel, vol_panel, date, adaptive=True):
    mr2 = {}; amihud = {}
    idx_loc = close_panel.index.get_loc(date)

    # 自适应窗口 (SURGE 用 fixed w=30, 对齐回测 make_surge_daily_fn 的 adaptive=False)
    if adaptive and "510300.SH" in close_panel.columns:
        idx_c = close_panel["510300.SH"]
        rets = idx_c.pct_change().dropna()
        mkt_vol = 0.20
        if idx_loc - 1 < len(rets) and idx_loc >= 61:
            mkt_vol = rets.rolling(60).std().iloc[idx_loc - 1] * np.sqrt(252)
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
    is_surge = breadth_count >= 2 and r_val >= 0.30
    print(f"  SURGE (每日): {'🚀 触发!' if is_surge else '未触发'} (广度={breadth_count}/3, regime={r_val:.3f})")
    for etf_code, above in surge_detail.items():
        print(f"    {etf_code}: {'✅ SMA50上' if above else '❌ SMA50下'}")

    # ── 因子 (月末用) ──
    def zscore(s):
        if not s: return {}
        vals = list(s.values()); mu, std = np.mean(vals), np.std(vals)
        if std == 0: return {c: 0 for c in s}
        return {c: (v-mu)/std for c, v in s.items()}

    def build_ranked(mr2, amihud):
        z1 = zscore(mr2); z2 = zscore(amihud)
        all_codes = set(z1.keys()) | set(z2.keys())
        composite = {c: (1-AMIHUD_W)*z1.get(c,0) + AMIHUD_W*z2.get(c,0) for c in all_codes}
        return sorted(composite.items(), key=lambda x: x[1], reverse=True)

    # SURGE 用固定 w=30 (对齐回测 make_surge_daily_fn), 进攻/防守用自适应窗口
    mr2, amihud = compute_factors(close_panel, vol_panel, today, adaptive=not is_surge)
    ranked = build_ranked(mr2, amihud)

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
        # 极端危机熔断: regime<0.15 → 全防御
        if r_val < 0.15:
            mode = "极端防御"
            defense_ok = [c for c in DEFENSE if c in close_panel.columns]
            if len(defense_ok) < 2: defense_ok = DEFENSE
            w = 1.0 / len(defense_ok)
            for d in defense_ok: signals.append((d, w, '全防御'))
            print(f"\n  极端防御 (regime={r_val:.3f}<0.15):")
            for code, w, label in signals:
                print(f"    {code} ({label}): {w*100:.0f}%")
            # 跳过后续KDJ选股逻辑
            # (用signals已有内容直接跳到输出)
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
        "positions": [{"code": c, "weight": round(w, 4), "label": l} for c, w, l in signals],
        "r4_upgrades": {
            "surge_daily": True,
            "kdj_mid_exit": False,  # 已证伪: KDJ中途出场验证负面, 弃用
            "regime_tanh_mult": TANH_MULT,
        },
        "kill_conditions": {
            "consecutive_losing_years": 2,
            "single_year_maxdd": 0.35,
        },
        "notes": f"R4纸面实盘。SURGE每日监控+regime(tanh×{TANH_MULT})。"
    }

    signals_dir = os.path.expanduser("~/ai-capital-ashare/signals")
    os.makedirs(signals_dir, exist_ok=True)
    signal_name = "etf_r4_paper_tushare.json" if "--tushare" in sys.argv else "etf_r4_paper.json"
    signal_path = os.path.join(signals_dir, signal_name)
    with open(signal_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  信号已保存: {signal_path}")

    return output


if __name__ == "__main__":
    generate_signal()
