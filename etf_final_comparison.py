#!/usr/bin/env python3
"""
严格验证框架下三策略对比: R2四档 / R2-V2 / RE-V2
时间点池 + R1风控 + Walk-Forward + MAD归因
"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/ai-capital-ashare/.claude/worktrees/etf-r1"))
from config import START_DATE
from run_final_backtest import compute_regime
from run_v5_backtest import month_end_dates
from src.stock_backtest import run_stock_backtest
try:
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark, load_qmt_stocks
except:
    import importlib.util
    s=importlib.util.spec_from_file_location("qa","src/qmt_adapter.py")
    qa=importlib.util.module_from_spec(s);s.loader.exec_module(qa)
    load_qmt_etfs=qa.load_qmt_etfs;load_qmt_benchmark=qa.load_qmt_benchmark;load_qmt_stocks=qa.load_qmt_stocks

BROAD=["sh510300","sh510500","sz159915","sh588000"]
DEFENSE=["sh511010","sh518880","sh511880"]
ALL_SECTOR_ETFS=["sh512880","sh512800","sh512010","sh512170","sz159928","sh512690",
    "sh512480","sz159995","sh512760","sh512660","sh515030","sh515790",
    "sh512400","sh512200","sh512980","sz159939","sh515210","sz159870",
    "sz159967","sz159611","sh512580","sh516670","sh512070","sz159865","sh516750",
    "sh512710","sh515050","sh516160","sz159819","sh562500","sh515880","sh512000",
    "sz159992","sz159869","sh515000","sh515650","sz159825","sh512100",
    "sh510050","sz159949","sh513050","sh513100","sh512890","sh510880",
    "sh512670","sh512720","sh512290","sz159840","sz159755",
    "sh516780","sh516970","sh516090","sh516010","sz159883","sz159828",
    "sh515230","sz159837","sz159807","sh517380","sh588200","sh588080",
    "sz159994","sh512680","sh510230","sz159887","sh515120"]
ALL_ETFS=BROAD+DEFENSE+ALL_SECTOR_ETFS
DATA_DIR=os.path.expanduser("~/ai-capital-ashare/data")

CORR_THRESHOLD=0.7; CORR_WINDOW=60; SURGE_DD_MELT=0.08

# RE 的 SECTOR_MAP
SECTOR_MAP={
    "sh512880":["证券","资本市场","货币金融"],"sh512800":["货币金融","金融服务"],
    "sh512070":["保险","非银","其他金融"],"sh512000":["证券","资本市场"],
    "sh512010":["医药","制药","医疗","中药","生物制品"],"sh512170":["医药","医疗","制药"],
    "sz159992":["医药","医疗","制药","创新"],
    "sz159928":["食品","饮料","酒","农牧","渔业","养殖","零售","批发","农副"],
    "sh512690":["酒","饮料"],
    "sh512480":["电子","计算机","通信","半导体"],"sz159995":["电子","计算机","通信","芯片"],
    "sh512760":["电子","半导体","芯片","集成电路"],
    "sh512660":["航天","航空","船舶","航海","兵装","军工","国防","卫星"],
    "sh512710":["航天","航空","船舶","军工","国防"],
    "sh515030":["汽车","新能源"],"sh515790":["光伏","电气机械","电源"],
    "sh516160":["新能源","碳中和","清洁能源"],
    "sh512400":["金属","有色","黄金","贵金属","黑色金属","采掘","煤炭","石油","矿业","矿物"],
    "sh515210":["黑色金属","冶","金属"],"sz159870":["化学","化纤","塑料","橡胶","造纸","纺织"],
    "sh512200":["房地产","建筑","土木","建材"],"sh516750":["非金属矿物","水泥","玻璃","陶瓷"],
    "sh512980":["传媒","游戏","影视","广告","出版","文化","旅游","教育","广播"],
    "sz159869":["传媒","游戏","影视","互联网"],
    "sz159939":["软件","互联网","通信","计算机","IT","信息服务","专业技术"],
    "sh515000":["软件","互联网","计算机","IT"],"sh512720":["软件","互联网","计算机","IT"],
    "sh515050":["5G","通信设备","信息技术","通信"],"sh515880":["通信","信息技术"],
    "sz159967":["设备","制造","机械","仪器","装备","高端","智能","自动化","机器人"],
    "sh562500":["设备","制造","机械","自动化","机器人"],
    "sz159819":["软件","互联网","计算机","IT","信息技术"],
    "sz159611":["电力","热力","燃气","水的生产","公用","发电","供电","电网"],
    "sh512580":["环保","环境","生态","水利","节能","清洁"],
    "sh516670":["运输","交通","物流","铁路","道路","航运","水上","港口","快递","高速","机场"],
    "sz159865":["农业","农牧","林业","畜牧","养殖","种业","种植","饲料"],
}

# ── 共享函数 ──────────────────────────────────────────
def build_surge_melt(index_close,etf_close,regime_series,cal,rebal,lock_days):
    sma50=index_close.rolling(50).mean();sma30=index_close.rolling(30).mean()
    dev30=(index_close-sma30)/sma30
    s30=0.6*(0.5+0.5*np.tanh(dev30*10))+0.4*(sma50>sma30).astype(float)
    breadth=pd.DataFrame({e:(etf_close[e]>etf_close[e].rolling(50).mean()).astype(float)
                          for e in["sh510300","sh510500","sz159915"]}).mean(axis=1)
    s30,breadth=s30.reindex(cal),breadth.reindex(cal);base=regime_series.reindex(cal)
    trig=(base>=0.15)&(s30>=0.70)&(breadth>=2/3)
    rebal_set=set(rebal);extra,forced,d2trig=[],{},{};lock_until,last_regime,cur_trig=-1,'NEUTRAL',None
    surge_peak=None
    def reg_4(s):return'RISKON' if s>=.70 else'NEUTRAL' if s>=.50 else'RISKOFF' if s>=.30 else'CRISIS'
    for i,d in enumerate(cal):
        if i<=lock_until:
            d2trig[d]=cur_trig
            if d in rebal_set:forced[d]='RISKON';last_regime='RISKON'
            if surge_peak is not None and etf_close.get("sh510300") is not None:
                cv=etf_close["sh510300"].get(d,None)
                if cv is not None and cv/surge_peak-1<-SURGE_DD_MELT:
                    lock_until=-1;surge_peak=None;cur_trig=None
            continue
        if d in rebal_set and d in regime_series.index:last_regime=reg_4(float(regime_series.loc[d]))
        if bool(trig.get(d,False)):
            lock_until=i+lock_days-1;cur_trig=d;d2trig[d]=d;forced[d]='RISKON'
            surge_peak=etf_close.get("sh510300",{}).get(d,None) if etf_close.get("sh510300") is not None else None
            if last_regime!='RISKON' and d not in rebal_set:extra.append(d);last_regime='RISKON'
    return pd.DatetimeIndex(extra),forced,d2trig

def regime_score(close):
    s250=close.rolling(250).mean();s50=close.rolling(50).mean()
    dev=(close-s250)/s250
    return 0.6*(0.5+0.5*np.tanh(dev*10))+0.4*(s50>s250).astype(float)

def stats(rr):
    v=rr["values"]["value"];rt=v.pct_change().dropna()
    yrs=len(rt)/244;ann=(v.iloc[-1]/1e6)**(1/yrs)-1
    dd=(v/v.cummax()-1).min();sh=rt.mean()/rt.std()*np.sqrt(244)
    yearly={int(k):round(v,1)for k,v in(rt.groupby(rt.index.year).apply(lambda x:((1+x).prod()-1)*100)).items()}
    ann_no20=round(np.mean([v for k,v in yearly.items() if k!=2020 and k>=2016]),1)
    return{"ann":round(ann*100,2),"mdd":round(dd*100,1),"sharpe":round(sh,2),"yearly":yearly,"ann_no20":ann_no20}

def ann_no_year(yearly,yr):return round(np.mean([v for k,v in yearly.items() if k!=yr and k>=2016]),1) if yearly else 0

print("加载...")
etf_data=load_qmt_etfs(ALL_ETFS,verbose=False)
closes={};first_trade={}
for s in ALL_ETFS:
    if s in etf_data:
        closes[s]=etf_data[s]["close"]
        vols=etf_data[s]["volume"];ft_idx=(vols>0).idxmax() if (vols>0).any() else None
        if ft_idx is not None:first_trade[s]=ft_idx
dfc=pd.DataFrame(closes).sort_index().dropna(how="all")
ends=[dfc[s].last_valid_index() for s in BROAD+DEFENSE if s in dfc.columns]
if ends:dfc=dfc.loc[:min(ends)].ffill(limit=5)
idx_close=load_qmt_benchmark("000300.SH")
rs_raw=regime_score(idx_close["close"]).dropna();rs_raw=rs_raw[rs_raw.index>=START_DATE]
cal_full=dfc.index[dfc.index>=START_DATE];me_full=month_end_dates(cal_full)
lc=dfc.notna().cumsum()

def available_etfs(date):
    result=[]
    for etf in ALL_SECTOR_ETFS:
        if etf in first_trade and etf in dfc.columns:
            ft=first_trade[etf]
            if ft<=date-pd.Timedelta(days=126):
                if date in dfc.index and etf in dfc.columns:
                    if lc.at[date,etf] if etf in dfc.columns else 0>=20:result.append(etf)
    return result if len(result)>=5 else ALL_SECTOR_ETFS[:5]

# ── R2 选股 ──────────────────────────────────────────
def compute_mom_r2(date,pool,window=30):
    scores={}
    for code in list(pool)+["sz159915","sh588000"]:
        if code not in dfc.columns:continue
        if code in first_trade and first_trade[code]>date-pd.Timedelta(days=20):continue
        idx=dfc.index.get_loc(date);dc=dfc[code]
        if idx<window+1:continue
        y=np.log(dc.iloc[idx-window:idx].values);x=np.arange(len(y))
        if len(y)<10:continue
        slope=np.polyfit(x,y,1)[0];ar=np.exp(slope*250)-1
        y_pred=slope*x+np.polyfit(x,y,1)[1]
        ss_r=np.sum((y-y_pred)**2);ss_t=np.sum((y-y.mean())**2)
        r2=1-ss_r/ss_t if ss_t>0 else 0;scores[code]=ar*r2
    return scores

def make_sel_r2():
    cache={}
    def sel(date,d2t):
        if date in cache:return cache[date]
        pool=available_etfs(date)
        scores=compute_mom_r2(date,pool);ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)
        is_s=d2t.get(date) is not None
        if is_s:r=[e for e,_ in ranked[:2]]
        else:
            r=[e for e,_ in ranked[:2]]
            if len(r)>=2 and r[0] in dfc.columns and r[1] in dfc.columns:
                common=dfc[[r[0],r[1]]].dropna().index
                if len(common)>0:
                    recent=common[-CORR_WINDOW:]
                    if len(recent)>=20:
                        if dfc.loc[recent,r[0]].corr(dfc.loc[recent,r[1]])>CORR_THRESHOLD:
                            for e,_ in ranked[2:]:r=[r[0],e];break
            growth=[g for g in["sz159915","sh588000"]if g in dfc.columns]
            r=r+growth[:2]
        if len(r)<2:r=[s for s in BROAD if s in dfc.columns][:4]
        cache[date]=r[:4];return r[:4]
    return sel

# ── RE 选股 ──────────────────────────────────────────
sd=None;ind_map={}
def load_re_data():
    global sd,ind_map
    sd=load_qmt_stocks(min_history=250)
    mf=pd.read_csv(os.path.join(DATA_DIR,"industry_map_bs.csv"),dtype=str)
    mf=mf[mf["industryClassification"]=="证监会行业分类"];mf["bare"]=mf["code"].str[3:]
    for _,r in mf.iterrows():
        if pd.notna(r["industry"]) and r["industry"]!="":ind_map[r["bare"]]=str(r["industry"])

def compute_q100(stock_data,date):
    scores={}
    for c,df in stock_data.items():
        if date not in df.index:continue
        idx=df.index.get_loc(date)
        if idx<250:continue
        lr=df['close'].iloc[idx]/df['close'].iloc[idx-250]-1
        scores[c]=np.log1p(lr) if lr>0 else 0
    return scores

def make_sel_re():
    cache={}
    def sel(date,d2t):
        if (date,id(d2t))in cache:return cache[(date,id(d2t))]
        valid={c:s for c,s in sd.items() if date in s.index and len(s[s.index<=date])>=250}
        if not valid:r=[s for s in BROAD if s in dfc.columns];cache[(date,id(d2t))]=r;return r
        scores=compute_q100(valid,date);top30=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:30]
        votes={}
        for code,_ in top30:
            ind=ind_map.get(code,"")
            for etf,kws in SECTOR_MAP.items():
                if any(k in ind for k in kws):votes[etf]=votes.get(etf,0)+1;break
        ranked=sorted(votes.items(),key=lambda x:x[1],reverse=True)
        top=[e for e,_ in ranked[:2] if etf_ok(e,date)]
        is_s=d2t.get(date) is not None
        if is_s:r=top[:2]
        else:
            r=top[:2]
            # 相关性过滤
            if len(r)>=2 and r[0] in dfc.columns and r[1] in dfc.columns:
                common=dfc[[r[0],r[1]]].dropna().index
                if len(common)>0:
                    recent=common[-CORR_WINDOW:]
                    if len(recent)>=20:
                        if dfc.loc[recent,r[0]].corr(dfc.loc[recent,r[1]])>CORR_THRESHOLD:
                            for e,_ in ranked[2:]:r=[r[0],e];break
            growth=[g for g in["sz159915","sh588000"]if etf_ok(g,date)]
            r=r+growth[:2]
        if len(r)<2:r=[s for s in BROAD if s in dfc.columns][:4]
        cache[(date,id(d2t))]=r[:4];return r[:4]
    return sel

def etf_ok(etf,date):
    if etf not in dfc.columns:return False
    if etf not in first_trade:return False
    ft=first_trade[etf]
    if ft>date-pd.Timedelta(days=20):return False
    return True

# ── 运行 ──────────────────────────────────────────────
def run_strategy(rs,lock_days,cal,me,sel_fn):
    extra,forced,d2t=build_surge_melt(idx_close["close"],dfc,rs_raw,cal,me,lock_days)
    rd=pd.DatetimeIndex(sorted(set(me)|set(extra)))
    rr=run_stock_backtest(dfc,rs,{},top_n=4,verbose=False,execution="next_open",
        stamp_duty=True,ffill_valuation=True,df_open=None,
        rebalance_dates=rd,select_fn=lambda d:sel_fn(d,d2t),forced_regime=forced)
    return stats(rr)

# IS/OOS
is_end="2019-12-31";oos_start="2020-01-01"
cal_is=cal_full[cal_full<=is_end];me_is=[d for d in me_full if d<=pd.Timestamp(is_end)]
cal_oos=cal_full[cal_full>=oos_start];me_oos=[d for d in me_full if d>=pd.Timestamp(oos_start)]

# Regime
rs_v2=rs_raw.copy()
for d in rs_v2.index:rs_v2.loc[d]=0.85 if rs_v2.loc[d]>=0.70 else 0.20

# RE needs stock data
print("加载RE个股数据...")
load_re_data()

# Walk-Forward: 用 14d lock (IS最优)
strategies=[
    ("R2-四档",rs_raw,14,make_sel_r2()),
    ("R2-V2",rs_v2,14,make_sel_r2()),
    ("RE-V2",rs_v2,14,make_sel_re()),
]

print(f"\n{'='*70}")
print("三策略严格验证对比")
print(f"{'='*70}")
print(f"  时间点池 + R1风控(相关过滤+熔断) + Walk-Forward (IS 2015-19 → OOS 2020-26)")
print(f"  SURGE lock=14d (IS最优), breadth≥2/3")

results=[]
for name,rs,ld,sel_fn in strategies:
    print(f"\n{name}:")
    s_is=run_strategy(rs,ld,cal_is,me_is,sel_fn)
    s_oos=run_strategy(rs,ld,cal_oos,me_oos,sel_fn)
    s_full=run_strategy(rs,ld,cal_full,me_full,sel_fn)
    wfe=s_oos['ann']/s_is['ann'] if s_is['ann']>0 else 0
    print(f"  IS 年化{s_is['ann']}% 回撤{s_is['mdd']}% | OOS 年化{s_oos['ann']}% 回撤{s_oos['mdd']}% | WFE={wfe:.2f}")
    print(f"  全样本: 年化{s_full['ann']}% 回撤{s_full['mdd']}% 夏普{s_full['sharpe']}  剔除2020:{s_full['ann_no20']}%")
    results.append({"name":name,"is":s_is,"oos":s_oos,"full":s_full,"wfe":wfe})

# ── 汇总 ─
print(f"\n{'='*70}")
print("三策略全样本对比")
print(f"{'='*70}")
print(f"  {'':<12} {'年化':>8} {'回撤':>8} {'夏普':>6} {'不含2020':>9} {'WFE':>6}")
for r in results:
    s=r['full']
    print(f"  {r['name']:<12} {s['ann']:>7.2f}% {s['mdd']:>7.1f}% {s['sharpe']:>5.2f} {s['ann_no20']:>8.1f}% {r['wfe']:>5.2f}")

# 逐年
print(f"\n逐年收益:")
all_years=sorted(set().union(*[set(r['full']['yearly'].keys()) for r in results]))
print(f"  {'年':<6}",end="")
for r in results:print(f"{r['name']:>12}",end="")
print()
for yr in all_years:
    if yr>=2016:
        print(f"  {yr:<6}",end="")
        for r in results:print(f"{r['full']['yearly'].get(yr,0):>11.1f}%",end="")
        print()
