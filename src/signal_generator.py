"""
信号生成器 — 每月最后一个交易日运行

输出：
  1. 当前市场制度
  2. 选股结果（Top-15，附因子得分明细）
  3. 目标持仓权重
  4. 与上期对比的调仓清单
"""
import os, sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REGIME_THRESHOLDS, UNIVERSE, START_DATE,
    COMMISSION, SLIPPAGE,
    MAX_SINGLE_POSITION, MAX_SECTOR_PCT,
    STOP_LOSS_PCT, DRAWDOWN_WARNING, DRAWDOWN_HALT, DRAWDOWN_LIQUIDATE,
    REVERSAL_FILTER_PCT,
)
from src.data_fetcher import fetch_index_daily
from src.stock_data import (
    get_csi300_constituents, fetch_stock_daily,
    compute_stock_factors, select_top_stocks, filter_reversal_stocks,
)
from src.risk_manager import RiskManager, RiskLimits, classify_sector

# ============================================================
# 配置
# ============================================================
TOP_N = 15
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stocks")
os.makedirs(CACHE_DIR, exist_ok=True)

# 仓位映射
ALLOCATION = {
    'RISKON':  {'equity': 0.95, 'bond': 0.00, 'gold': 0.00, 'cash': 0.05},
    'NEUTRAL': {'equity': 0.60, 'bond': 0.30, 'gold': 0.05, 'cash': 0.05},
    'RISKOFF': {'equity': 0.30, 'bond': 0.50, 'gold': 0.10, 'cash': 0.10},
    'CRISIS':  {'equity': 0.00, 'bond': 0.65, 'gold': 0.15, 'cash': 0.20},
}


def compute_regime(benchmark_close, etf_prices=None):
    """
    计算当前市场制度（v4最终版：SMA250 + 广度SMA30锁仓）

    如果 etf_prices 不为空，检查广度SMA30锁仓信号
    """
    sma50 = benchmark_close.rolling(50).mean()
    sma250 = benchmark_close.rolling(250).mean()

    last_sma50 = sma50.iloc[-1]
    last_sma250 = sma250.iloc[-1]
    last_price = benchmark_close.iloc[-1]

    # SMA250基础分
    dev = (last_price - last_sma250) / last_sma250
    trend_score = 0.5 + 0.5 * np.tanh(dev * 10)
    golden_score = 1.0 if last_sma50 > last_sma250 else 0.0
    total = 0.6 * trend_score + 0.4 * golden_score

    # ===== 广度SMA30锁仓检查 =====
    lock_active = False
    if etf_prices is not None:
        lock_active = _check_breadth_lock(benchmark_close, etf_prices)
    # 日频监控的持久化锁状态(surge_monitor.py 维护): 月中触发的锁在月末调仓时生效
    if not lock_active:
        lock_active = _surge_state_active()

    if lock_active:
        total = max(total, 0.72)  # 强制至少RISKON

    # 分类
    if total >= REGIME_THRESHOLDS.get('RISKON', 0.70):
        regime = 'RISKON'
    elif total >= REGIME_THRESHOLDS.get('NEUTRAL', 0.50):
        regime = 'NEUTRAL'
    elif total >= REGIME_THRESHOLDS.get('RISKOFF', 0.30):
        regime = 'RISKOFF'
    else:
        regime = 'CRISIS'

    return {
        'regime': regime,
        'score': round(total, 3),
        'price': round(float(last_price), 2),
        'sma250': round(float(last_sma250), 2),
        'sma50': round(float(last_sma50), 2),
        'deviation_pct': round(float(dev * 100), 1),
        'lock_active': lock_active,
    }


def _surge_state_active():
    """读取 surge_monitor.py 持久化的锁状态; 缺文件/过期/超过3天未刷新均视为不活跃"""
    import json
    from datetime import datetime, timedelta
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "signals", "surge_state.json")
    try:
        with open(path) as f:
            st = json.load(f)
        if not st.get("active"):
            return False
        today = datetime.now().date()
        if st.get("expire_date") and str(today) > st["expire_date"]:
            return False
        checked = datetime.strptime(st.get("checked_date", "1970-01-01"), "%Y-%m-%d").date()
        if today - checked > timedelta(days=3):  # 监控断更超3天, 不信任陈旧锁
            return False
        return True
    except Exception:
        return False


def _check_breadth_lock(benchmark_close, etf_prices):
    """检查广度SMA30锁仓信号"""
    if etf_prices is None or len(etf_prices) < 3:
        return False

    sma30 = benchmark_close.rolling(30).mean()
    sma50 = benchmark_close.rolling(50).mean()
    sma250 = benchmark_close.rolling(250).mean()

    last_price = benchmark_close.iloc[-1]
    last_sma30 = sma30.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    last_sma250 = sma250.iloc[-1]

    # SMA250基础分
    dev_250 = (last_price - last_sma250) / last_sma250
    base_score = 0.6 * (0.5 + 0.5 * np.tanh(dev_250 * 10)) + 0.4 * (1.0 if last_sma50 > last_sma250 else 0.0)

    # 必须在非极端恐慌区
    if base_score < 0.15:
        return False

    # SMA30制度分
    dev_30 = (last_price - last_sma30) / last_sma30
    sma30_score = 0.6 * (0.5 + 0.5 * np.tanh(dev_30 * 10)) + 0.4 * (1.0 if last_sma50 > last_sma30 else 0.0)

    if sma30_score < 0.70:
        return False

    # 广度检查：3只ETF站上SMA50的比例
    breadth_now = 0
    breadth_15d_ago = 0
    for etf_name, prices in etf_prices.items():
        if isinstance(prices, pd.Series) and len(prices) >= 65:
            etf_sma50 = prices.rolling(50).mean()
            if prices.iloc[-1] > etf_sma50.iloc[-1]:
                breadth_now += 1
            if len(prices) >= 65 and prices.iloc[-16] > etf_sma50.iloc[-16]:
                breadth_15d_ago += 1

    total_etfs = len(etf_prices)
    if total_etfs < 3:
        return False

    breadth_now_pct = breadth_now / total_etfs
    breadth_15d_pct = breadth_15d_ago / total_etfs

    # 广度突破：15天内从<33%跳到>67%
    return breadth_15d_pct < 0.33 and breadth_now_pct > 0.67


def load_stock_universe():
    """加载股票池：CSI300 + 创业板指 + 科创50"""
    csi300 = set(get_csi300_constituents())

    import akshare as ak
    chinext = set(ak.index_stock_cons(symbol="399006")['品种代码'].apply(lambda x: str(x).zfill(6)))
    star50 = set(ak.index_stock_cons(symbol="000688")['品种代码'].apply(lambda x: str(x).zfill(6)))

    all_codes = sorted(csi300 | chinext | star50)

    stock_data = {}
    for code in all_codes:
        cache_path = os.path.join(CACHE_DIR, f"{code}.csv")
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if len(df) > 250:
                    stock_data[code] = df
            except Exception:
                pass

    return stock_data


def generate_signals(stock_data=None, benchmark_data=None):
    """生成交易信号"""
    now = datetime.now()

    # ===== 1. 获取基准数据 =====
    if benchmark_data is None:
        print("获取沪深300指数数据...")
        benchmark_data = fetch_index_daily("sh000300")

    if benchmark_data is None or benchmark_data.empty:
        print("❌ 无法获取基准数据")
        return None

    # ===== 2. 制度检测 =====
    # 加载ETF价格用于广度检查
    from src.universe import load_price_data, get_aligned_prices
    etf_close = get_aligned_prices(load_price_data())
    etf_prices = {c: etf_close[c] for c in ['sh510300','sh510500','sz159915'] if c in etf_close.columns}

    regime_info = compute_regime(benchmark_data['close'], etf_prices)

    # ===== 3. 加载股票数据 =====
    if stock_data is None:
        print("加载股票池...")
        stock_data = load_stock_universe()

    if not stock_data:
        print("❌ 股票数据为空")
        return None

    # ===== 4. 选股 =====
    regime = regime_info['regime']
    latest_date = benchmark_data.index[-1]

    # 过滤：最新日期有数据的股票
    valid_stocks = {}
    for code, df in stock_data.items():
        if latest_date in df.index and len(df[df.index <= latest_date]) >= 250:
            valid_stocks[code] = df

    # V4.2 反转过滤: 剔除近20日涨幅前REVERSAL_FILTER_PCT的候选(A股反转效应)
    valid_stocks = filter_reversal_stocks(valid_stocks, latest_date, exclude_pct=REVERSAL_FILTER_PCT)

    # 计算因子得分
    scores = compute_stock_factors(valid_stocks, latest_date)
    selected = select_top_stocks(scores, TOP_N)

    # 选股明细
    stock_details = []
    for code in selected:
        s = scores.get(code, 0)
        stock_details.append({'code': code, 'score': round(s, 4)})

    # ===== 4.5 风控验证（新增！）=====
    risk_limits = RiskLimits(
        max_position_pct=MAX_SINGLE_POSITION,
        max_sector_pct=MAX_SECTOR_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        drawdown_warning=DRAWDOWN_WARNING,
        drawdown_halt=DRAWDOWN_HALT,
        drawdown_liquidate=DRAWDOWN_LIQUIDATE,
    )
    risk_mgr = RiskManager(risk_limits)

    # 用等权计算每只股票的目标权重
    n = len(stock_details) if stock_details else 1
    alloc_equity = ALLOCATION[regime]['equity']
    per_stock_weight = alloc_equity / n if n > 0 else 0

    # 验证每只入选股票的合规性（模拟验证，不需要当前持仓）
    risk_warnings = []
    risk_filtered = []
    for s in stock_details:
        code = s['code']
        sector = classify_sector(code)
        # 计算该板块已有几只入选
        sector_count = sum(1 for x in risk_filtered if classify_sector(x['code']) == sector)
        sector_weight = (sector_count + 1) * per_stock_weight

        if per_stock_weight > MAX_SINGLE_POSITION:
            risk_warnings.append(f"⚠️ {code} 权重 {per_stock_weight:.1%} 超单只上限 {MAX_SINGLE_POSITION:.1%}")
        if sector_weight > MAX_SECTOR_PCT:
            risk_warnings.append(f"⚠️ {code}({sector}) 板块权重将达 {sector_weight:.1%}，跳过")
            continue  # 跳过此票，选池里下一只

        risk_filtered.append(s)

    # 如果过滤掉了股票，从备选中补上
    if len(risk_filtered) < len(stock_details):
        already = set(s['code'] for s in risk_filtered)
        remaining = [code for code in selected if code not in already]
        all_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fallback = [code for code, sc in all_ranked if code not in already][:TOP_N]
        for code in fallback:
            if len(risk_filtered) >= TOP_N:
                break
            if code not in already:
                s = scores.get(code, 0)
                sector = classify_sector(code)
                sector_count = sum(1 for x in risk_filtered if classify_sector(x['code']) == sector)
                sector_weight = (sector_count + 1) * per_stock_weight
                if sector_weight <= MAX_SECTOR_PCT:
                    risk_filtered.append({'code': code, 'score': round(s, 4)})
                    already.add(code)

    if risk_filtered:
        stock_details = risk_filtered
        selected = [s['code'] for s in stock_details]

    # 生成风控报告
    risk_report = {
        'limits': {
            'max_single_position': f"{MAX_SINGLE_POSITION:.0%}",
            'max_sector_pct': f"{MAX_SECTOR_PCT:.0%}",
            'stop_loss': f"{STOP_LOSS_PCT:.0%}",
            'drawdown_warning': f"{DRAWDOWN_WARNING:.0%}",
            'drawdown_halt': f"{DRAWDOWN_HALT:.0%}",
            'drawdown_liquidate': f"{DRAWDOWN_LIQUIDATE:.0%}",
        },
        'warnings': risk_warnings,
        'filtered_count': len(risk_warnings),
    }

    # ===== 5. 目标权重 =====
    alloc = ALLOCATION[regime]

    target_positions = {}
    if regime == 'CRISIS' or not selected:
        target_positions['国债ETF(sh511010)'] = alloc['bond']
        target_positions['黄金ETF(sh518880)'] = alloc['gold']
        target_positions['货币ETF(sh511880)'] = alloc['cash']
    else:
        n = len(selected)
        per_stock = alloc['equity'] / n
        for code in selected:
            target_positions[f'个股({code})'] = round(per_stock * 100, 2)
        if alloc['bond'] > 0:
            target_positions['国债ETF(sh511010)'] = round(alloc['bond'] * 100, 2)
        if alloc['gold'] > 0:
            target_positions['黄金ETF(sh518880)'] = round(alloc['gold'] * 100, 2)
        if alloc['cash'] > 0:
            target_positions['现金'] = round(alloc['cash'] * 100, 2)

    # ===== 6. 输出 =====
    report = {
        'date': latest_date.strftime('%Y-%m-%d'),
        'generated_at': now.strftime('%Y-%m-%d %H:%M'),
        'regime': regime_info,
        'selected_stocks': stock_details,
        'allocation': {k: f"{v*100:.0f}%" for k, v in alloc.items()},
        'target_positions': target_positions,
        'universe_size': len(valid_stocks),
        'total_stocks_loaded': len(stock_data),
        'risk_report': risk_report,
    }

    return report


def print_report(report):
    """格式化打印信号报告"""
    if report is None:
        print("无信号")
        return

    r = report['regime']
    regime_emoji = {'RISKON': '🟢', 'NEUTRAL': '🟡', 'RISKOFF': '🟠', 'CRISIS': '🔴'}

    print()
    print('=' * 60)
    print(f"  量化信号报告 — {report['date']}")
    print('=' * 60)
    print()

    # 制度
    emoji = regime_emoji.get(r['regime'], '❓')
    print(f"  📊 市场制度: {emoji} {r['regime']} (score={r['score']:.3f})")
    print(f"     沪深300: {r['price']} | SMA50: {r['sma50']} | SMA250: {r['sma250']}")
    print(f"     偏离SMA250: {r['deviation_pct']:+.1f}%")
    print()

    # 仓位
    print(f"  💰 目标仓位配置:")
    for k, v in report['allocation'].items():
        print(f"     {k}: {v}")
    print()

    # 选股
    if report['selected_stocks']:
        print(f"  📈 精选个股 (Top-{len(report['selected_stocks'])}, 从{report['universe_size']}只中选出):")
        print(f"     {'代码':<8s} {'得分':>8s}  {'板块':<6s}")
        print(f"     {'-'*28}")
        for s in report['selected_stocks'][:15]:
            sector = classify_sector(s['code'])
            print(f"     {s['code']:<8s} {s['score']:>8.4f}  {sector}")
    else:
        print(f"  🛡️ 防御模式：不持仓个股，全部配置债券/黄金/货币")

    # 风控警告
    rr = report.get('risk_report', {})
    if rr.get('warnings'):
        print()
        print(f"  🛡️ 风控提示 ({rr['filtered_count']}条):")
        for w in rr['warnings']:
            print(f"     {w}")
    if rr.get('limits'):
        limits = rr['limits']
        print(f"  📏 风控参数: 单票≤{limits['max_single_position']} | 板块≤{limits['max_sector_pct']} | 止损{limits['stop_loss']}")

    print()
    print(f"  下次调仓: 下月最后一个交易日")
    print(f"  生成时间: {report['generated_at']}")
    print('=' * 60)


if __name__ == '__main__':
    print("=" * 60)
    print("  AI Capital A-Share — 量化信号生成器")
    print("=" * 60)

    report = generate_signals()
    print_report(report)
