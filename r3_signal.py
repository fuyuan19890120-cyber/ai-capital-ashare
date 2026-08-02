#!/usr/bin/env python3
"""ETF-R3 飞书推送 · akshare在线拉数据, 零依赖"""
import json, os, requests
import pandas as pd, numpy as np
from datetime import datetime, timedelta

WEBHOOK = os.environ.get("FEISHU_WEBHOOK", "")
VL, VH, MW, MAX_W, AM_W = 0.15, 0.40, 15, 50, 0.15
ETF = {
    "510300":"沪深300","510500":"中证500","159915":"创业板","588000":"科创50",
    "511010":"国债","518880":"黄金","511880":"货币",
    "512880":"证券","512800":"银行","512070":"非银","512000":"券商",
    "512010":"医药","512170":"医疗","159992":"创新药","159928":"消费","512690":"酒",
    "512480":"半导体","159995":"芯片","512760":"半导体2",
    "512660":"军工","512710":"国防","515030":"新能车","515790":"光伏","516160":"新能源",
    "512400":"有色","515210":"钢铁","159870":"化工","512200":"地产","516750":"建材",
    "512980":"传媒","159869":"游戏","159939":"科技","515000":"科技2","512720":"计算机",
    "515050":"5G","515880":"通信","159967":"制造","562500":"机器人",
    "159819":"AI","159611":"电力","512580":"环保","516670":"畜牧","159865":"农业",
}
SECTOR = [c for c,n in ETF.items() if n not in ("沪深300","中证500","创业板","科创50","国债","黄金","货币")]

def fetch():
    import akshare as ak
    s = (datetime.now() - timedelta(days=500)).strftime("%Y%m%d")
    e = datetime.now().strftime("%Y%m%d")
    frames = {}
    for code in ETF:
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=s, end_date=e, adjust="qfq")
            if df is not None and len(df) > 50:
                df = df.rename(columns={"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume"})
                df["date"] = pd.to_datetime(df["date"])
                frames[code] = df.set_index("date")[["open","high","low","close","volume"]].sort_index()
        except: pass
    print(f"数据: {len(frames)}/{len(ETF)}只ETF")
    return frames

def regime(data, d):
    c = data["510300"]["close"]
    sma250, sma50 = c.rolling(250).mean(), c.rolling(50).mean()
    dev = (c - sma250) / sma250
    return float((0.6 * (0.5 + 0.5 * np.tanh(dev * 10)) + 0.4 * (sma50 > sma250).astype(float)).loc[d]) if d in c.index else 0.5

def factors(data, d):
    close = pd.DataFrame({c: data[c]["close"] for c in data}).ffill(limit=5)
    vol = pd.DataFrame({c: data[c]["volume"] for c in data}).ffill(limit=5)
    idx = close.index.get_loc(d)
    rets = close["510300"].pct_change().dropna()
    mv = 0.20
    if idx >= 60 and idx < len(rets): mv = max(0.15, min(0.40, rets.rolling(60).std().iloc[idx] * np.sqrt(252)))
    w = int(np.clip(MAX_W - (mv - VL) / (VH - VL) * (MAX_W - MW), MW, MAX_W))
    mr2, am = {}, {}
    for code in close.columns:
        c = close[code].iloc[:idx+1].dropna()
        if len(c) < w+1: continue
        y = np.log(c.iloc[-w-1:-1].values); x = np.arange(len(y))
        if len(y) >= 10:
            try:
                slope = np.polyfit(x, y, 1)[0]; ann = np.exp(slope * 250) - 1
                y_pred = slope * x + np.polyfit(x, y, 1)[1]
                mr2[code] = ann * (1 - np.sum((y-y_pred)**2)/np.sum((y-y.mean())**2)) if np.sum((y-y.mean())**2) > 0 else 0
            except: pass
        if code in vol.columns and idx >= 20:
            c20 = close[code].iloc[idx-20:idx]; v20 = vol[code].iloc[idx-20:idx]
            if len(c20) >= 20 and not v20.isna().any():
                ra = c20.pct_change().dropna(); va, ca = v20.iloc[1:], c20.iloc[1:]
                vs = [abs(ra.iloc[i])/(va.iloc[i]*ca.iloc[i]) for i in range(min(len(ra),len(va))) if va.iloc[i] > 0 and ca.iloc[i] > 0]
                if vs: am[code] = np.mean(vs) * 1e8
    return mr2, am, close

def kdj_signal(data, d):
    sig = {}
    for code in data:
        c = data[code]["close"].loc[:d]; h = data[code]["high"].loc[:d]; l = data[code]["low"].loc[:d]
        n = len(c)
        if n < 30: continue
        rsv = np.zeros(n); k = np.ones(n)*50; d_ = np.ones(n)*50
        for i in range(8, n):
            hh = h.iloc[i-8:i+1].max(); ll = l.iloc[i-8:i+1].min()
            rsv[i] = (c.iloc[i]-ll)/(hh-ll)*100 if hh > ll else 50
        for i in range(1, n): k[i] = 2/3*k[i-1]+1/3*rsv[i]; d_[i] = 2/3*d_[i-1]+1/3*k[i]
        j = 3*k - 2*d_
        golden = k[-2] <= d_[-2] and k[-1] > d_[-1]
        s = pd.Series(c.values)
        ma3, ma6, ma12, ma24 = s.rolling(3).mean(), s.rolling(6).mean(), s.rolling(12).mean(), s.rolling(24).mean()
        bbi = (ma3.iloc[-1]+ma6.iloc[-1]+ma12.iloc[-1]+ma24.iloc[-1])/4
        if golden and j[-2] < 20 and c.iloc[-1] > bbi: sig[code] = True
    return sig

def zscore(s):
    if not s: return {}
    vals = list(s.values()); mu, std = np.mean(vals), np.std(vals)
    return {c: (v-mu)/std for c, v in s.items()} if std > 0 else {c:0 for c in s}

print(f"ETF-R3 @ {datetime.now()}")
data = fetch()
dates = sorted(data["510300"].index)
today = dates[-1]
print(f"日期: {today.date()}")

r = regime(data, today)
label = "RISKON" if r >= 0.50 else "防守"
print(f"Regime: {r:.3f} ({label})")

mr2, am, close = factors(data, today)
z1, z2 = zscore(mr2), zscore(am)
composite = {c: (1-AM_W)*z1.get(c,0) + AM_W*z2.get(c,0) for c in set(z1)|set(z2)}
ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)
top2 = [c for c,_ in ranked[:2] if c in SECTOR]
if len(top2) < 2: top2 = [c for c,_ in ranked[:2]]

br = {e: bool(close[e].dropna().iloc[-1] > close[e].dropna().rolling(50).mean().iloc[-1]) for e in ["510300","510500","159915"] if e in close}
bc = sum(br.values())
surge = bc >= 2 and r >= 0.15
print(f"SURGE: {'🚀' if surge else '否'} ({bc}/3)")

signals = []
if r >= 0.50 or surge:
    if surge:
        signals = [(top2[0], 0.50), (top2[1], 0.50)]
    else:
        signals = [(top2[0], 0.25), (top2[1], 0.25), ("159915", 0.25), ("588000", 0.25)]
else:
    kdj = kdj_signal(data, today)
    kl = [c for c in kdj]
    if len(kl) >= 2:
        for c in kl[:4]: signals.append((c, 1.0/min(len(kl),4)))
    elif len(kl) == 1:
        for c,_ in ranked:
            if c not in kl: kl.append(c)
            if len(kl) >= 4: break
        for c in kl[:4]: signals.append((c, 1.0/max(len(kl),1)))
    else:
        floor = 0.30 if r >= 0.40 else (0.20 if r >= 0.30 else (0.10 if r >= 0.15 else 0.0))
        if floor > 0:
            signals = [("510300", floor/2), ("510500", floor/2), ("511010", (1-floor)/3), ("518880", (1-floor)/3), ("511880", (1-floor)/3)]
        else:
            for d_ in ["511010","518880","511880"]: signals.append((d_, 1.0/3))

for c, w in signals: print(f"  {c} {ETF.get(c,'')}: {w*100:.0f}%")

color = "green" if label == "RISKON" else "red"
pos = "\n".join(f"• {c} {ETF.get(c,'')}: {w*100:.0f}%" for c, w in signals)
card = {"msg_type":"interactive","card":{"header":{"title":{"tag":"plain_text","content":f"ETF-R3 调仓 · {today.date()}"},"template":color},"elements":[{"tag":"div","text":{"tag":"lark_md","content":f"**Regime: {r:.3f} ({label})** | SURGE: {'🚀触发!' if surge else '未触发'} ({bc}/3)"}},{"tag":"hr"},{"tag":"div","text":{"tag":"lark_md","content":f"**持仓:**\n{pos}"}},{"tag":"hr"},{"tag":"div","text":{"tag":"lark_md","content":"akshare数据 · 回测: 21.2%/1.09/WFE 1.62"}},{"tag":"note","elements":[{"tag":"plain_text","content":f"GitHub Actions · {datetime.now():%Y-%m-%d %H:%M}"}]}]}}

if WEBHOOK:
    resp = requests.post(WEBHOOK, json=card)
    print(f"飞书: {'OK' if resp.status_code==200 else 'FAIL '+str(resp.status_code)}")
else:
    print("WEBHOOK未设置, 跳过推送")
