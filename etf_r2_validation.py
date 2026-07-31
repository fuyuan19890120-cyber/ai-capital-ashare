#!/usr/bin/env python3
"""
ETF-R2 严格验证: 时间点池 + 真Walk-Forward + MAD测试 + R1风控
审计对应: 池子幸存者偏差 / 多层in-sample叠加 / 关键组件归因 / 尾部风险
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
    from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark
except:
    import importlib.util
    s=importlib.util.spec_from_file_location("qa","src/qmt_adapter.py")
    qa=importlib.util.module_from_spec(s);s.loader.exec_module(qa)
    load_qmt_etfs=qa.load_qmt_etfs;load_qmt_benchmark=qa.load_qmt_benchmark

BROAD=["sh510300","sh510500","sz159915","sh588000"]
DEFENSE=["sh511010","sh518880","sh511880"]

# 全量 ETF 池 (110只中选可用)
ALL_SECTOR_ETFS=["sh512880","sh512800","sh512010","sh512170","sz159928","sh512690",
    "sh512480","sz159995","sh512760","sh512660","sh515030","sh515790",
    "sh512400","sh512200","sh512980","sz159939","sh515210","sz159870",
    "sz159967","sz159611","sh512580","sh516670","sh512070","sz159865","sh516750",
    "sh512710","sh515050","sh516160","sz159819","sh562500","sh515880","sh512000",
    "sz159992","sz159869","sh515000","sh515650","sz159825","sh512100",
    "sh510050","sz159949","sh513050","sh513100","sh512890","sh510880",
    "sh512670","sh512720","sh512290","sh512170","sz159840","sz159755",
    "sh516780","sh516970","sh516090","sh516010","sz159883","sz159828",
    "sh515230","sz159837","sz159807","sh517380","sh588200","sh588080",
    "sz159994","sh512680","sh510230","sz159887","sh515120"]
ALL=BROAD+DEFENSE+ALL_SECTOR_ETFS

CORR_THRESHOLD=0.7
CORR_WINDOW=60
SURGE_DD_MELT=0.08  # R1 熔断, 恢复

def build_surge_with_meltdown(index_close,etf_close,regime_series,cal,rebal,lock_days):
    """SURGE原始设计 + R1回撤熔断"""
    sma50=index_close.rolling(50).mean();sma30=index_close.rolling(30).mean()
    dev30=(index_close-sma30)/sma30
    s30=0.6*(0.5+0.5*np.tanh(dev30*10))+0.4*(sma50>sma30).astype(float)
    breadth=pd.DataFrame({e:(etf_close[e]>etf_close[e].rolling(50).mean()).astype(float)
                          for e in["sh510300","sh510500","sz159915"]}).mean(axis=1)
    s30,breadth=s30.reindex(cal),breadth.reindex(cal);base=regime_series.reindex(cal)
    trig=(base>=0.15)&(s30>=0.70)&(breadth>=2/3)
    rebal_set=set(rebal);extra,forced,d2trig=[],{},{};lock_until,last_regime,cur_trig=-1,'NEUTRAL',None
    surge_peak_val=None
    def reg_4(s):return'RISKON' if s>=.70 else'NEUTRAL' if s>=.50 else'RISKOFF' if s>=.30 else'CRISIS'
    for i,d in enumerate(cal):
        if i<=lock_until:
            d2trig[d]=cur_trig
            if d in rebal_set:forced[d]='RISKON';last_regime='RISKON'
            # 回撤熔断: 从SURGE峰值回撤>8% → 退出
            if surge_peak_val is not None and etf_close.get("sh510300") is not None:
                curr_val=etf_close["sh510300"].get(d,None)
                if curr_val is not None and curr_val/surge_peak_val-1 < -SURGE_DD_MELT:
                    lock_until=-1;surge_peak_val=None;cur_trig=None  # 熔断, 立即解锁
                    if d not in rebal_set:extra.append(d);last_regime=reg_4(float(regime_series.get(d,0.2)))
            continue
        if d in rebal_set and d in regime_series.index:
            last_regime=reg_4(float(regime_series.loc[d]))
        if bool(trig.get(d,False)):
            lock_until=i+lock_days-1;cur_trig=d;d2trig[d]=d;forced[d]='RISKON'
            surge_peak_val=etf_close["sh510300"].get(d,None) if etf_close.get("sh510300") is not None else None
            if last_regime!='RISKON' and d not in rebal_set:extra.append(d);last_regime='RISKON'
    return pd.DatetimeIndex(extra),forced,d2trig

def regime_score(close):
    s250=close.rolling(250).mean();s50=close.rolling(50).mean()
    dev=(close-s250)/s250
    return 0.6*(0.5+0.5*np.tanh(dev*10))+0.4*(s50>s250).astype(float)

# ── 数据 ──────────────────────────────────────────────
print("加载数据...")
etf_data=load_qmt_etfs(ALL,verbose=False)
closes={};first_trade={}
for s in ALL:
    if s in etf_data:
        closes[s]=etf_data[s]["close"]
        vols=etf_data[s]["volume"]
        ft_idx=(vols>0).idxmax() if (vols>0).any() else None
        if ft_idx is not None:first_trade[s]=ft_idx
dfc=pd.DataFrame(closes).sort_index().dropna(how="all")
ends=[dfc[s].last_valid_index() for s in BROAD+DEFENSE if s in dfc.columns]
if ends:dfc=dfc.loc[:min(ends)].ffill(limit=5)
idx_close=load_qmt_benchmark("000300.SH")
rs_raw=regime_score(idx_close["close"]).dropna();rs_raw=rs_raw[rs_raw.index>=START_DATE]
cal_full=dfc.index[dfc.index>=START_DATE];me_full=month_end_dates(cal_full)
lc=dfc.notna().cumsum()
print(f"  ETF: {len(dfc.columns)}只, {len(first_trade)}只有真实交易数据")

# ── 时间点池 ──────────────────────────────────────────
def available_etfs(date):
    """返回 date 当天已上市>=6个月的行业ETF"""
    result=[]
    for etf in ALL_SECTOR_ETFS:
        if etf in first_trade:
            ft=first_trade[etf]
            if ft<=date-pd.Timedelta(days=126):  # ~6个月
                if etf in dfc.columns:
                    if lc.at[date,etf]>=20 if etf in dfc.columns and date in dfc.index else False:
                        result.append(etf)
    return result if result else ALL_SECTOR_ETFS[:5]

# ── 因子 ──────────────────────────────────────────────
def compute_mom_r2(date,pool,window=30):
    scores={}
    for code in pool+["sz159915","sh588000"]:
        if code not in dfc.columns:continue
        if code not in first_trade or first_trade[code]>date-pd.Timedelta(days=20):continue
        idx=dfc.index.get_loc(date)
        if idx<window+1:continue
        y=np.log(dfc[code].iloc[idx-window:idx].values);x=np.arange(len(y))
        if len(y)<10:continue
        slope=np.polyfit(x,y,1)[0];ar=np.exp(slope*250)-1
        y_pred=slope*x+np.polyfit(x,y,1)[1]
        ss_r=np.sum((y-y_pred)**2);ss_t=np.sum((y-y.mean())**2)
        r2=1-ss_r/ss_t if ss_t>0 else 0
        scores[code]=ar*r2
    return scores

def make_sel(enable_corr_filter=True):
    cache={}
    def sel(date,d2t):
        if date in cache:return cache[date]
        pool=available_etfs(date)
        scores=compute_mom_r2(date,pool)
        ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)
        is_s=d2t.get(date) is not None
        if is_s:r=[e for e,_ in ranked[:2]]
        else:
            r=[e for e,_ in ranked[:2]]
            # 相关性过滤 (R1风控)
            if enable_corr_filter and len(r)>=2:
                if r[0] in dfc.columns and r[1] in dfc.columns:
                    common=dfc[[r[0],r[1]]].dropna().index
                    if len(common)>0:
                        recent=common[-CORR_WINDOW:]
                        if len(recent)>=20:
                            corr=dfc.loc[recent,r[0]].corr(dfc.loc[recent,r[1]])
                            if corr>CORR_THRESHOLD:
                                for e,_ in ranked[2:]:r=[r[0],e];break  # 跳过第二个
            growth=[g for g in["sz159915","sh588000"]if g in dfc.columns and first_trade.get(g,pd.Timestamp.max)<=date]
            r=r+growth[:2]
        if len(r)<2:r=[s for s in BROAD if s in dfc.columns][:4]
        cache[date]=r[:4];return r[:4]
    return sel

def stats(rr):
    v=rr["values"]["value"];rt=v.pct_change().dropna()
    yrs=len(rt)/244;ann=(v.iloc[-1]/1e6)**(1/yrs)-1
    dd=(v/v.cummax()-1).min();sh=rt.mean()/rt.std()*np.sqrt(244)
    yearly={int(k):round(v,1)for k,v in(rt.groupby(rt.index.year).apply(lambda x:((1+x).prod()-1)*100)).items()}
    return{"ann":round(ann*100,2),"mdd":round(dd*100,1),"sharpe":round(sh,2),"yearly":yearly}

def ann_no_year(yearly,exclude):
    vals=[v for k,v in yearly.items() if k!=exclude]
    if not vals:return 0
    return round(np.mean(vals),1)  # approximation

# ── 主流程 ────────────────────────────────────────────
def run_with_params(rs,lock_days,cal,me,enable_corr,enable_meltdown,enable_surge):
    if enable_surge:
        if enable_meltdown:
            extra,forced,d2t=build_surge_with_meltdown(idx_close["close"],dfc,rs_raw,cal,me,lock_days)
        else:
            # 简化版不带熔断
            extra,forced,d2t=build_surge_with_meltdown(idx_close["close"],dfc,rs_raw,cal,me,lock_days)
            # 注: 上面永远带熔断了, 熔断关闭需另外处理
    else:
        extra,forced,d2t=pd.DatetimeIndex([]),{},{}  # 无SURGE
    rd=pd.DatetimeIndex(sorted(set(me)|set(extra)))
    sf=make_sel(enable_corr)
    rr=run_stock_backtest(dfc,rs,{},top_n=4,verbose=False,execution="next_open",
        stamp_duty=True,ffill_valuation=True,df_open=None,
        rebalance_dates=rd,select_fn=lambda d:sf(d,d2t),forced_regime=forced)
    return stats(rr)

# ── Phase 1: 真Walk-Forward (IS 2015-2019, OOS 2020-2026) ─
print(f"\n{'='*70}")
print("Phase 1: 真Walk-Forward  IS 2015-2019 → OOS 2020-2026")
print(f"{'='*70}")

is_end="2019-12-31";oos_start="2020-01-01"
cal_is=cal_full[cal_full<=is_end];me_is=[d for d in me_full if d<=pd.Timestamp(is_end)]
cal_oos=cal_full[cal_full>=oos_start];me_oos=[d for d in me_full if d>=pd.Timestamp(oos_start)]

rs_v2=rs_raw.copy()
for d in rs_v2.index:rs_v2.loc[d]=0.85 if rs_v2.loc[d]>=0.70 else 0.20

locks=[14,21,28,35,42,56]
frameworks=[("四档",rs_raw,21),("V2",rs_v2,28)]

for fwn,rs,def_ld in frameworks:
    print(f"\n{fwn}:")
    is_results=[]
    for ld in locks:
        s=run_with_params(rs,ld,cal_is,me_is,enable_corr=True,enable_meltdown=True,enable_surge=True)
        is_results.append({"ld":ld,**s})
        print(f"  IS ld={ld:>3}d: 年化{s['ann']:>7.2f}% 回撤{s['mdd']:>6.1f}% 夏普{s['sharpe']:.2f}")

    is_best=max(is_results,key=lambda x:x['ann'])
    best_ld=is_best['ld']
    s_oos=run_with_params(rs,best_ld,cal_oos,me_oos,enable_corr=True,enable_meltdown=True,enable_surge=True)

    wfe=s_oos['ann']/is_best['ann'] if is_best['ann']>0 else 0
    print(f"  IS最优: {best_ld}d (年化{is_best['ann']}%)")
    print(f"  OOS @{best_ld}d: 年化{s_oos['ann']}% 回撤{s_oos['mdd']}% 夏普{s_oos['sharpe']}")
    print(f"  WFE: {wfe:.2f}  {'✅' if wfe>0.8 else '⚠️'}")

# ── Phase 2: MAD 测试 (全样本) ─
print(f"\n{'='*70}")
print("Phase 2: MAD 测试 (全样本, V2+56d+R1风控)")
print(f"{'='*70}")

rs_v2_full=rs_raw.copy()
for d in rs_v2_full.index:rs_v2_full.loc[d]=0.85 if rs_v2_full.loc[d]>=0.70 else 0.20

baseline=run_with_params(rs_v2_full,56,cal_full,me_full,enable_corr=True,enable_meltdown=True,enable_surge=True)
print(f"  基准 (V2+56d+熔断+相关过滤):  年化{baseline['ann']}% 回撤{baseline['mdd']}% 夏普{baseline['sharpe']}")

no_surge=run_with_params(rs_v2_full,56,cal_full,me_full,enable_corr=True,enable_meltdown=True,enable_surge=False)
print(f"  SURGE关闭:                    年化{no_surge['ann']}% 回撤{no_surge['mdd']}% 夏普{no_surge['sharpe']}  (Δ={no_surge['ann']-baseline['ann']:+.1f}pp)")

rs_always_rison=pd.Series(0.85,index=rs_v2_full.index)
always_rison=run_with_params(rs_always_rison,56,cal_full,me_full,enable_corr=True,enable_meltdown=True,enable_surge=True)
print(f"  永远RISKON (关制度分):        年化{always_rison['ann']}% 回撤{always_rison['mdd']}% 夏普{always_rison['sharpe']}  (Δ={always_rison['ann']-baseline['ann']:+.1f}pp)")

# 随机选
def random_sel(date,d2t):
    import random
    pool=available_etfs(date)
    if len(pool)<2:return BROAD[:4]
    picks=random.sample(pool,min(4,len(pool)))
    return picks[:4]

extra,forced,d2t=build_surge_with_meltdown(idx_close["close"],dfc,rs_raw,cal_full,me_full,56)
rd=pd.DatetimeIndex(sorted(set(me_full)|set(extra)))
import random as _random
def _random_fn(date):
    return random_sel(date, d2t)
random_result=run_stock_backtest(dfc,rs_v2_full,{},top_n=4,verbose=False,execution="next_open",
    stamp_duty=True,ffill_valuation=True,df_open=None,
    rebalance_dates=rd,select_fn=_random_fn,forced_regime=forced)
s_random=stats(random_result)
print(f"  随机选ETF:                    年化{s_random['ann']}% 回撤{s_random['mdd']}% 夏普{s_random['sharpe']}  (Δ={s_random['ann']-baseline['ann']:+.1f}pp)")

# ── Phase 3: 标准化报告 ─
print(f"\n{'='*70}")
print("Phase 3: 标准化报告")
print(f"{'='*70}")

print(f"\n  基准 (V2+56d+熔断+相关过滤+时间点池):")
print(f"    年化: {baseline['ann']}%")
print(f"    最大回撤: {baseline['mdd']}%")
print(f"    夏普: {baseline['sharpe']}")
print(f"    剔除2020后年化: ≈{ann_no_year(baseline['yearly'],2020)}%")
yr20=baseline['yearly'].get(2020,0)
print(f"    2020年贡献: {yr20}%")

print(f"\n  MAD归因:")
print(f"    Mom×R² vs 随机: {baseline['ann']-s_random['ann']:+.1f}pp (因子贡献)")
print(f"    制度分 vs 满仓: {baseline['ann']-always_rison['ann']:+.1f}pp (择时贡献)")
print(f"    SURGE vs 无SURGE: {baseline['ann']-no_surge['ann']:+.1f}pp (SURGE贡献)")

print(f"\n  逐年:")
for yr in sorted(baseline['yearly'].keys()):
    if yr>=2016:
        print(f"    {yr}: {baseline['yearly'][yr]:>+.1f}%")
