#!/usr/bin/env python3
"""ETF-R3 GitHub Actions: 读 etf_compact.db → 生成信号 → 飞书推送"""
import json, os, sqlite3, requests
import pandas as pd, numpy as np
from datetime import datetime

WEBHOOK = os.environ.get("FEISHU_WEBHOOK", "")
DB = "data/etf_compact.db"
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

def load_data():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM daily ORDER BY code, date", con)
    con.close()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df

def compute_factors(df, date):
    close = df.pivot(index='date', columns='code', values='close').ffill(limit=5)
    vol_p = df.pivot(index='date', columns='code', values='volume').ffill(limit=5)
    high = df.pivot(index='date', columns='code', values='high').ffill(limit=5)
    low = df.pivot(index='date', columns='code', values='low').ffill(limit=5)
    idx = close.index.get_loc(date)
    
    rets = close.get("510300.SH", close.iloc[:,0]).pct_change().dropna()
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
                vals = [abs(ra.iloc[i])/(va.iloc[i]*ca.iloc[i]) for i in range(min(len(ra),len(va))) if va.iloc[i] > 0 and ca.iloc[i] > 0]
                if vals: am[code] = np.mean(vals) * 1e8
    return mr2, am, close, high, low

def compute_regime(close):
    c = close.get("510300.SH", close.iloc[:,0])
    sma250 = c.rolling(250).mean(); sma50 = c.rolling(50).mean()
    dev = (c - sma250) / sma250
    trend = 0.5 + 0.5 * np.tanh(dev * 10)
    gc = (sma50 > sma250).astype(float)
    return 0.6 * trend + 0.4 * gc

def compute_kdj(close, high, low, date):
    signals = {}
    idx = close.index.get_loc(date)
    for code in close.columns:
        c = close[code].iloc[:idx+1]; h = high[code].iloc[:idx+1]; l = low[code].iloc[:idx+1]
        n = len(c); 
        if n < 30: continue
        rsv = np.zeros(n); k = np.ones(n)*50; d = np.ones(n)*50; j = np.zeros(n)
        for i in range(8, n):
            hh = h.iloc[i-8:i+1].max(); ll = l.iloc[i-8:i+1].min()
            rsv[i] = (c.iloc[i]-ll)/(hh-ll)*100 if hh > ll else 50
        for i in range(1, n): k[i]=2/3*k[i-1]+1/3*rsv[i]; d[i]=2/3*d[i-1]+1/3*k[i]; j[i]=3*k[i]-2*d[i]
        golden = (k[-2] <= d[-2] and k[-1] > d[-1])
        j_oversold = j[-2] < 20
        s = pd.Series(c.values)
        ma3=s.rolling(3).mean(); ma6=s.rolling(6).mean(); ma12=s.rolling(12).mean(); ma24=s.rolling(24).mean()
        bbi = (ma3.iloc[-1]+ma6.iloc[-1]+ma12.iloc[-1]+ma24.iloc[-1])/4
        if golden and j_oversold and c.iloc[-1] > bbi:
            signals[code] = True
    return signals

def zscore(s):
    if not s: return {}
    vals = list(s.values()); mu, std = np.mean(vals), np.std(vals)
    if std == 0: return {c: 0 for c in s}
    return {c: (v-mu)/std for c, v in s.items()}

def generate():
    print(f"ETF-R3 @ {datetime.now()}")
    df = load_data()
    dates = sorted(df['date'].unique())
    today = dates[-1]
    print(f"最新数据: {today.date()}")

    close = df.pivot(index='date', columns='code', values='close').ffill(limit=5)
    regime_series = compute_regime(close)
    r_val = float(regime_series.loc[today]) if today in regime_series.index else 0.5
    label = "RISKON" if r_val >= 0.50 else "防守"
    print(f"Regime: {r_val:.3f} ({label})")

    mr2, am, close_p, high_p, low_p = compute_factors(df, today)
    z1, z2 = zscore(mr2), zscore(am)
    all_c = set(z1.keys()) | set(z2.keys())
    composite = {c: (1-AMIHUD_W)*z1.get(c,0) + AMIHUD_W*z2.get(c,0) for c in all_c}
    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)

    top2 = [c for c, _ in ranked[:2] if c in SECTOR_ETFS]
    if len(top2) < 2: top2 = [c for c, _ in ranked[:2]]

    breadth = {}
    for etf in ["510300.SH","510500.SH","159915.SZ"]:
        c = close_p.get(etf, pd.Series()).dropna()
        if len(c) >= 50:
            sma50_e = c.rolling(50).mean()
            breadth[etf] = bool(c.iloc[-1] > sma50_e.iloc[-1])
    bc = sum(breadth.values())
    is_surge = bc >= 2 and r_val >= 0.15
    print(f"SURGE: {'🚀' if is_surge else '否'} (广度{bc}/3)")

    signals = []
    if r_val >= 0.50 or is_surge:
        if is_surge:
            signals = [(top2[0], 0.50, "SURGE"), (top2[1], 0.50, "SURGE")]
        else:
            sw = SECTOR_PCT / 200; gw = (100 - SECTOR_PCT) / 200
            signals = [(top2[0], sw, "行业1"), (top2[1], sw, "行业2")]
            for g in ["159915.SZ","588000.SH"]:
                if g in close_p.columns: signals.append((g, gw, "创业板" if "159915" in g else "科创50"))
    else:
        kdj_sig = compute_kdj(close_p, high_p, low_p, today)
        kdj_list = [c for c in kdj_sig]
        if len(kdj_list) >= 2:
            n = min(len(kdj_list), 4); w = 1.0 / n
            for c in kdj_list[:n]: signals.append((c, w, "KDJ"))
        elif len(kdj_list) == 1:
            for c, _ in ranked:
                if c not in kdj_list: kdj_list.append(c)
                if len(kdj_list) >= 4: break
            for c in kdj_list[:4]: signals.append((c, 1.0/len(kdj_list), "KDJ+Amihud"))
        else:
            if r_val >= 0.40: floor = 0.30
            elif r_val >= 0.30: floor = 0.20
            elif r_val >= 0.15: floor = 0.10
            else: floor = 0.0
            if floor > 0:
                broad_ok = [c for c in BROAD if c in close_p.columns]
                signals.extend([(broad_ok[0], floor/2, "宽基"), (broad_ok[1] if len(broad_ok)>1 else broad_ok[0], floor/2, "宽基")])
                d_ok = [c for c in DEFENSE if c in close_p.columns]
                remain = 1.0 - floor
                for d in d_ok[:3]: signals.append((d, remain/3, "防御"))
            else:
                d_ok = [c for c in DEFENSE if c in close_p.columns]
                w = 1.0 / max(len(d_ok), 1)
                for d in d_ok[:4]: signals.append((d, w, "全防御"))

    return {
        "signal_date": str(today.date()), "regime": round(r_val, 3), "regime_label": label,
        "surge": is_surge, "surge_breadth": f"{bc}/3",
        "positions": [(c, round(w, 4), l) for c, w, l in signals],
    }

def send_card(s):
    color = "green" if s["regime_label"] == "RISKON" else "red"
    surge_text = "🚀 SURGE触发!" if s["surge"] else f"SURGE未触发 (广度{s['surge_breadth']})"
    pos = "\n".join(f"• {c} ({l}): {w*100:.0f}%" for c, w, l in s["positions"])
    card = {"msg_type":"interactive","card":{"header":{"title":{"tag":"plain_text","content":f"ETF-R3 调仓 · {s['signal_date']}"},"template":color},"elements":[{"tag":"div","text":{"tag":"lark_md","content":f"**Regime: {s['regime']:.3f} ({s['regime_label']})** | {surge_text}"}},{"tag":"hr"},{"tag":"div","text":{"tag":"lark_md","content":f"**持仓:**\n{pos}"}},{"tag":"hr"},{"tag":"div","text":{"tag":"lark_md","content":"回测: 21.2%/1.09/WFE 1.62"}},{"tag":"note","elements":[{"tag":"plain_text","content":f"GitHub Actions · {datetime.now():%Y-%m-%d %H:%M}"}]}]}}
    if WEBHOOK:
        resp = requests.post(WEBHOOK, json=card)
        print(f"飞书: {'OK' if resp.status_code==200 else 'FAIL '+str(resp.status_code)}")

if __name__ == "__main__":
    s = generate()
    for c, w, l in s["positions"]: print(f"  {c} ({l}): {w*100:.0f}%")
    send_card(s)
