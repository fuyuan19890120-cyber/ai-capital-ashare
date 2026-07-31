#!/usr/bin/env python3
"""
ETF-R3 GitHub Actions 信号生成器
纯 sqlite3 读取, 零外部依赖(除numpy/pandas), 适配 CI 环境
每月末运行 → 生成持仓 → 飞书推送
"""
import sqlite3, json, os, sys, requests
import pandas as pd, numpy as np
from datetime import datetime

DB = 'data/etf_compact.db'
WEBHOOK = os.environ.get("FEISHU_WEBHOOK", "")
SECTOR_PCT = 50; MAX_WINDOW = 50; VL = 0.15; VH = 0.40; MW = 15; AMIHUD_W = 0.15

BROAD = ["510300.SH","510500.SH","159915.SZ","588000.SH"]
DEFENSE = ["511010.SH","518880.SH","511880.SH"]
SECTOR_ETFS = [
    "512880.SH","512800.SH","512070.SH","512000.SH","512010.SH","512170.SH","159992.SZ",
    "159928.SZ","512690.SH","512480.SH","159995.SZ","512760.SH","512660.SH","512710.SH",
    "515030.SH","515790.SH","516160.SH","512400.SH","515210.SH","159870.SZ",
    "512200.SH","516750.SH","512980.SH","159869.SZ","159939.SZ","515000.SH","512720.SH",
    "515050.SH","515880.SH","159967.SZ","562500.SH","159819.SZ","159611.SZ","512580.SH",
    "516670.SH","159865.SZ",
]
ALL_ETFS = BROAD + DEFENSE + SECTOR_ETFS

def load_data():
    con = sqlite3.connect(DB)
    ph = ','.join(['?']*len(ALL_ETFS))
    df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM daily WHERE code IN ({ph}) AND date >= '2015-01-01' ORDER BY code, date", con, params=ALL_ETFS)
    con.close()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df

def compute_factors(df, date):
    close = df.pivot(index='date', columns='code', values='close').ffill(limit=5)
    vol_p = df.pivot(index='date', columns='code', values='volume').ffill(limit=5)
    idx = close.index.get_loc(date)

    # 自适应窗口
    idx_c = close.get("510300.SH", close.iloc[:,0])
    rets = idx_c.pct_change().dropna()
    mkt_vol = 0.20
    if idx >= 60 and idx < len(rets):
        v = rets.rolling(60).std().iloc[idx] * np.sqrt(252)
        if not np.isnan(v): mkt_vol = v
    w = int(np.clip(MAX_WINDOW - (mkt_vol - VL) / (VH - VL) * (MAX_WINDOW - MW), MW, MAX_WINDOW))

    mr2, am = {}, {}
    for code in close.columns:
        c = close[code].iloc[:idx+1].dropna()
        if len(c) < w + 1: continue
        y = np.log(c.iloc[-w-1:-1].values); x = np.arange(len(y))
        if len(y) >= 10:
            try:
                slope = np.polyfit(x, y, 1)[0]; ann = np.exp(slope * 250) - 1
                y_pred = slope * x + np.polyfit(x, y, 1)[1]
                ss_res = np.sum((y - y_pred)**2); ss_tot = np.sum((y - y.mean())**2)
                mr2[code] = ann * (1 - ss_res / ss_tot) if ss_tot > 0 else 0
            except: pass
        if code in vol_p.columns and idx >= 20:
            c20 = close[code].iloc[idx-20:idx]; v20 = vol_p[code].iloc[idx-20:idx]
            if len(c20) >= 20 and not v20.isna().any():
                ra = c20.pct_change().dropna(); va = v20.iloc[1:]; ca = c20.iloc[1:]
                vals = []
                for i in range(min(len(ra), len(va))):
                    if va.iloc[i] > 0 and ca.iloc[i] > 0:
                        vals.append(abs(ra.iloc[i]) / (va.iloc[i] * ca.iloc[i]))
                if vals: am[code] = np.mean(vals) * 1e8
    return mr2, am, close, vol_p

def compute_kdj(close_p, high_p, low_p, date):
    idx = close_p.index.get_loc(date)
    signals = {}
    for code in close_p.columns:
        c = close_p[code].iloc[:idx+1]; h = high_p[code].iloc[:idx+1]; l = low_p[code].iloc[:idx+1]
        n = len(c)
        if n < 30: continue
        rsv = np.zeros(n); k = np.ones(n)*50; d = np.ones(n)*50; j = np.zeros(n)
        for i in range(8, n):
            hh = h.iloc[i-8:i+1].max(); ll = l.iloc[i-8:i+1].min()
            rsv[i] = (c.iloc[i] - ll) / (hh - ll) * 100 if hh > ll else 50
        for i in range(1, n): k[i] = 2/3*k[i-1] + 1/3*rsv[i]; d[i] = 2/3*d[i-1] + 1/3*k[i]; j[i] = 3*k[i] - 2*d[i]
        golden = (k[-2] <= d[-2] and k[-1] > d[-1])
        j_oversold = j[-2] < 20
        s = pd.Series(c.values)
        ma3 = s.rolling(3).mean(); ma6 = s.rolling(6).mean(); ma12 = s.rolling(12).mean(); ma24 = s.rolling(24).mean()
        bbi_val = (ma3.iloc[-1]+ma6.iloc[-1]+ma12.iloc[-1]+ma24.iloc[-1])/4
        if golden and j_oversold and c.iloc[-1] > bbi_val:
            signals[code] = True
    return signals

def zscore(s):
    if not s: return {}
    vals = list(s.values()); mu, std = np.mean(vals), np.std(vals)
    if std == 0: return {c: 0 for c in s}
    return {c: (v-mu)/std for c, v in s.items()}

def generate():
    print(f"=== ETF-R3 GitHub Action === {datetime.now()}")
    df = load_data()
    dates = sorted(df['date'].unique())
    today = dates[-1]
    print(f"数据日期: {today.date()}")

    # Regime
    con = sqlite3.connect(DB)
    idx_df = pd.read_sql_query("SELECT date,close FROM daily WHERE code='000300.SH' AND date>='2015-01-01' ORDER BY date", con, parse_dates={'date':'%Y%m%d'})
    con.close()
    idx_c = idx_df.set_index('date')['close']
    sma250 = idx_c.rolling(250).mean(); sma50 = idx_c.rolling(50).mean()
    dev = (idx_c - sma250) / sma250
    trend = 0.5 + 0.5 * np.tanh(dev * 10)
    gc = (sma50 > sma250).astype(float)
    regime = 0.6 * trend + 0.4 * gc
    r_val = float(regime.loc[today]) if today in regime.index else 0.5

    # 因子
    mr2, am, close_p, vol_p = compute_factors(df, today)
    z1, z2 = zscore(mr2), zscore(am)
    all_c = set(z1.keys()) | set(z2.keys())
    composite = {c: (1-AMIHUD_W)*z1.get(c,0) + AMIHUD_W*z2.get(c,0) for c in all_c}
    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)

    top2 = [c for c, _ in ranked[:2] if c in SECTOR_ETFS]
    if len(top2) < 2: top2 = [c for c, _ in ranked[:2]]

    # SURGE
    breadth = {}
    for etf in ["510300.SH","510500.SH","159915.SZ"]:
        c = close_p.get(etf, pd.Series()).dropna()
        if len(c) >= 50:
            sma50_e = c.rolling(50).mean()
            breadth[etf] = bool(c.iloc[-1] > sma50_e.iloc[-1])
        else:
            breadth[etf] = False
    breadth_count = sum(breadth.values())
    is_surge = breadth_count >= 2 and r_val >= 0.15

    # 生成持仓
    signals = []
    if r_val >= 0.50 or is_surge:
        if is_surge:
            signals = [(top2[0], 0.50, "SURGE行业1"), (top2[1], 0.50, "SURGE行业2")]
        else:
            sw = SECTOR_PCT / 200; gw = (100 - SECTOR_PCT) / 200
            signals = [(top2[0], sw, "行业1"), (top2[1], sw, "行业2")]
            for g in ["159915.SZ","588000.SH"]:
                if g in close_p.columns: signals.append((g, gw, "创业板" if "159915" in g else "科创50"))
    else:
        high_p = df.pivot(index='date', columns='code', values='high').ffill(limit=5)
        low_p = df.pivot(index='date', columns='code', values='low').ffill(limit=5)
        kdj_sig = compute_kdj(close_p, high_p, low_p, today)
        kdj_list = [c for c in kdj_sig if c in SECTOR_ETFS + ["159915.SZ","588000.SH","510300.SH","510500.SH"]]
        if len(kdj_list) >= 2:
            n = min(len(kdj_list), 4); w = 1.0 / n
            for c in kdj_list[:n]: signals.append((c, w, "KDJ选股"))
        elif len(kdj_list) == 1:
            for c, _ in ranked:
                if c not in kdj_list and c in SECTOR_ETFS: kdj_list.append(c)
                if len(kdj_list) >= 4: break
            n = max(len(kdj_list), 1); w = 1.0 / n
            for c in kdj_list[:4]: signals.append((c, w, "KDJ+Amihud"))
        else:
            if r_val >= 0.40: floor = 0.30
            elif r_val >= 0.30: floor = 0.20
            elif r_val >= 0.15: floor = 0.10
            else: floor = 0.0
            if floor > 0:
                broad_ok = [c for c in BROAD if c in close_p.columns]
                signals.extend([(broad_ok[0], floor/2, "宽基底仓"), (broad_ok[1] if len(broad_ok)>1 else broad_ok[0], floor/2, "宽基底仓")])
                defense_ok = [c for c in DEFENSE if c in close_p.columns]
                if len(defense_ok) < 2: defense_ok = DEFENSE
                remain = 1.0 - floor
                for d in defense_ok[:3]: signals.append((d, remain/3, "防御"))
            else:
                defense_ok = [c for c in DEFENSE if c in close_p.columns]
                if len(defense_ok) < 2: defense_ok = DEFENSE
                w = 1.0 / max(len(defense_ok), 1)
                for d in defense_ok[:4]: signals.append((d, w, "全防御"))

    output = {
        "strategy": "ETF-R3",
        "generated_at": datetime.now().isoformat(),
        "signal_date": str(today.date()),
        "regime": round(r_val, 3),
        "regime_label": "RISKON" if r_val >= 0.50 else ("NEUTRAL" if r_val >= 0.40 else ("RISKOFF" if r_val >= 0.30 else "CRISIS")),
        "surge": is_surge,
        "surge_detail": {"breadth_count": breadth_count, "breadth": breadth},
        "positions": [{"code": c, "weight": round(w, 4), "label": l} for c, w, l in signals],
    }
    return output, today, r_val, is_surge

def build_card(s):
    color = "green" if s["regime_label"] == "RISKON" else "red"
    surge_text = "🚀 SURGE触发!" if s["surge"] else "未触发"
    bc = s["surge_detail"]["breadth_count"]
    pos = "\n".join(f"• {p['code']} ({p['label']}): {p['weight']*100:.0f}%" for p in s["positions"])
    return {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"ETF-R3 调仓 · {s['signal_date']}"}, "template": color},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**Regime: {s['regime']:.3f} ({s['regime_label']})** | SURGE: {surge_text} (广度{bc}/3)"}},
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**持仓:**\n{pos}"}},
                {"tag": "hr"},
                {"tag": "div", "text": {"tag": "lark_md", "content": "回测: 年化21.2% / 夏普1.09 / WFE 1.62"}},
                {"tag": "note", "elements": [{"tag": "plain_text", "content": f"GitHub Actions · {datetime.now().strftime('%Y-%m-%d %H:%M')} · 纸面实盘"}]}
            ]
        }
    }

if __name__ == "__main__":
    signal, today, r_val, is_surge = generate()
    print(f"Regime={r_val:.3f} SURGE={is_surge} 持仓={len(signal['positions'])}只")
    for p in signal["positions"]:
        print(f"  {p['code']} ({p['label']}): {p['weight']*100:.0f}%")

    card = build_card(signal)
    if WEBHOOK:
        resp = requests.post(WEBHOOK, json=card)
        print(f"飞书: {'OK' if resp.status_code == 200 else f'FAIL {resp.status_code}'}")
    else:
        print("FEISHU_WEBHOOK 未设置, 跳过推送")
