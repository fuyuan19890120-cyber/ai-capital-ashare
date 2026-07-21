#!/usr/bin/env python3
"""GitHub Actions 用: 批量下载全市场行业/主题ETF日线(头2只/主题, ~160只)"""
import json, os, time, re
import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "etf_sector")
os.makedirs(OUT, exist_ok=True)

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"})

EXCLUDE = ['沪深300','中证500','中证1000','中证2000','创业板','科创','上证综指','上证50','上证180',
           '深证','深100','中证A50','中证A500','中证800','国债','企债','债','货币','黄金','纳指',
           '恒生','标普','德国','日经','法国','印度','东南亚','MSCI','富时','H股','港股','原油',
           '豆粕','白银','油气','金ETF','沙特','巴西','亚太','教育','道琼斯','美国50','日本',
           '可转债','公司债','信用债','城投债','地方债','政金债','国开债','短融']

KW = ['半导体','芯片','集成电路','光伏','新能源','电池','储能','碳中和','绿电','绿色电力','电力',
      '电网','军工','国防','航天','航空','通用航空','机器人','人工智能','AI','软件','计算机',
      '信创','云计算','大数据','信息安全','信息技术','工业互联网','通信','5G','电信','物联网',
      '消费电子','电子','TMT','科技','互联网','金融科技','数字经济','证券','券商','银行','保险',
      '非银','金融','房地产','医药','医疗','医械','生物医药','创新药','中药','疫苗','汽车',
      '智能汽车','智能车','汽车零部件','机械','工程机械','工业母机','机床','智能驾驶','食品',
      '饮料','消费','影视','传媒','游戏','旅游','家电','家居家电','有色','金属','稀有金属',
      '矿业','煤炭','能源','石油','化工','材料','建材','钢铁','稀土','农业','农牧','养殖',
      '粮食','畜牧','电力公用','公用事业','一带一路','央企','国企','红利','价值','ESG',
      '物流','交运','交通运输','船舶','高铁','环保','石化','云计算','大数据','信息安全']

def theme(s):
    for k in KW:
        if k in str(s): return k
    return None

def fetch_etf_list():
    """新浪全量ETF列表(WAF绕行: 完整浏览器特征)"""
    url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/jsonp.php/"
           "IO.XSRV2.CallbackList['da_yPT46_Ll7K6WD']/Market_Center.getHQNodeDataSimple")
    r = S.get(url, params={"page": "1", "num": "5000", "sort": "symbol", "asc": "0",
                           "node": "etf_hq_fund", "[object HTMLDivElement]": "qvvne"}, timeout=30)
    t = r.text
    if "([" not in t:
        # Fallback: 用 akshare
        import akshare as ak
        df = ak.fund_etf_category_sina(symbol="ETF基金")
        df = df[df["代码"].str.match(r'^(sh|sz)\d+')]
    else:
        t = t[t.index("(["):t.rindex("])") + 2]
        df = pd.DataFrame(json.loads(t)[1:],
                          columns=["代码","名称","最新价","涨跌额","涨跌幅","买入","卖出","昨收","今开","最高","最低","成交量","成交额","",""])
    mask = df["名称"].apply(lambda x: not any(e in str(x) for e in EXCLUDE))
    sec = df[mask].copy()
    sec["主题"] = sec["名称"].apply(theme)
    sec = sec.dropna(subset=["主题"])
    sec = sec.sort_values(["主题", "代码"])
    picks = sec.groupby("主题").head(2) if len(sec) > 0 else pd.DataFrame()
    # Fallback: 若拉不到就下载固定核心清单(Actions US也被WAF拦时兜底)
    if len(picks) < 20:
        picks = pd.DataFrame([
            ("sh515070","人工智能"), ("sh512580","环保"), ("sh512710","军工龙头"),
            ("sh512070","非银"), ("sh515220","煤炭"), ("sh516150","稀土"),
            ("sh515180","红利"), ("sh512170","医疗"), ("sh515250","智能汽车"),
            ("sz159998","计算机"), ("sh512480","半导体设备"), ("sh512200","房地产"),
            ("sh515210","钢铁"), ("sh512760","半导体"), ("sh515050","5G"),
            ("sh515880","通信"), ("sh515790","光伏"), ("sh512660","军工"),
            ("sh516160","新能源"), ("sh515030","新能源车"), ("sz159869","游戏"),
            ("sz159865","养殖"), ("sh516510","云计算"),
        ], columns=["代码","名称"])
        picks = picks.drop_duplicates("代码")
    print(f"ETF清单: {len(picks)} 只", flush=True)
    return picks


def download_one(sym, out_csv):
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 5000:
        return "skip"
    try:
        r = S.get(f"https://quotes.sina.cn/cn/api/jsonp_v2.php/x=/CN_MarketDataService.getKLineData?"
                  f"symbol={sym}&scale=240&ma=no&datalen=3000", timeout=25)
        t = r.text
        lb = t.index("["); rb = t.rindex("]")
        data = json.loads(t[lb:rb + 1])
        if not data: return "empty"
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["day"])
        df = df.set_index("date").sort_index()[["open", "close"]].astype(float)
        df.to_csv(out_csv)
        return f"{len(df)}行"
    except Exception as e:
        return f"FAIL:{type(e).__name__}"


def main():
    picks = fetch_etf_list()
    print(f"待下载: {len(picks)} 只, {picks['主题'].nunique()} 主题", flush=True)
    ok = skip = fail = 0
    t0 = time.time()
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        sym, name, t_theme = r["代码"], r["名称"], r["主题"]
        out_csv = os.path.join(OUT, f"{sym}.csv")
        result = download_one(sym, out_csv)
        if result == "skip": skip += 1
        elif result.startswith("FAIL"): print(f"  ! {sym} {name[:25]} {result}", flush=True); fail += 1
        else: ok += 1
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(picks)}] ok={ok} skip={skip} fail={fail} elapsed={el/60:.1f}min", flush=True)
        time.sleep(0.25)
    print(f"DONE ok={ok} skip={skip} fail={fail} pool={len(os.listdir(OUT))}", flush=True)


if __name__ == "__main__":
    main()
