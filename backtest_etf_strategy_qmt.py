"""
ETF AI Capital 策略回测 — QMT 数据库版
SMA250制度检测 + 广度SMA30 SURGE猛攻 + 进攻/防守双池轮动
数据源: qmt_full.db (SQLite), 不复权 + 拆分修正
对比三组：
  (A) 买入持有沪深300
  (B) 纯宽基制度择时（baseline）
  (C) ETF AI Capital 策略（进攻池行业轮动 + SURGE）
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from src.qmt_adapter import load_qmt_etfs, load_qmt_benchmark, from_qmt_code, to_qmt_code

# ============================================================
# ETF 资产池定义
# ============================================================

# 进攻池 — 12只行业/主题ETF
OFFENSE_POOL = [
    {"name": "证券ETF",      "code": "sh512880", "ak_code": "512880"},
    {"name": "半导体ETF",    "code": "sh512760", "ak_code": "512760"},
    {"name": "芯片ETF",      "code": "sz159995", "ak_code": "159995"},
    {"name": "医药ETF",      "code": "sh512010", "ak_code": "512010"},
    {"name": "军工ETF",      "code": "sh512660", "ak_code": "512660"},
    {"name": "5GETF",        "code": "sh515050", "ak_code": "515050"},
    {"name": "酒ETF",        "code": "sh512690", "ak_code": "512690"},
    {"name": "消费ETF",      "code": "sz159928", "ak_code": "159928"},
    {"name": "光伏ETF",      "code": "sh515790", "ak_code": "515790"},
    {"name": "新能源ETF",    "code": "sh516160", "ak_code": "516160"},
    {"name": "中证1000ETF",  "code": "sh512100", "ak_code": "512100"},
    {"name": "创业板50ETF",  "code": "sz159949", "ak_code": "159949"},
]

# 高Beta子池（SURGE专用，从进攻池中挑选）
HIGH_BETA_CODES = {"sh512880", "sh512760", "sz159995", "sh512660",
                    "sh515050", "sh515790", "sh516160", "sz159949"}

# 宽基池
BROAD_POOL = [
    {"name": "沪深300",    "code": "sh510300", "ak_code": "510300"},
    {"name": "中证500",    "code": "sh510500", "ak_code": "510500"},
    {"name": "创业板",     "code": "sz159915", "ak_code": "159915"},
    {"name": "科创50",     "code": "sh588000", "ak_code": "588000"},
]

# 防守池
DEFENSE_POOL = [
    {"name": "国债ETF",    "code": "sh511010", "ak_code": "511010"},
    {"name": "企业债ETF",  "code": "sh511220", "ak_code": "511220"},
    {"name": "黄金ETF",    "code": "sh518880", "ak_code": "518880"},
    {"name": "货币ETF",    "code": "sh511880", "ak_code": "511880"},
]

ALL_ETFS = OFFENSE_POOL + BROAD_POOL + DEFENSE_POOL

# ============================================================
# 策略参数
# ============================================================
START_DATE = "2015-01-01"
END_DATE   = "2026-07-13"
INITIAL_CAPITAL = 1_000_000
ETF_COST = 0.0006          # ETF单边成本（万0.5佣金+0.05%滑点）
SURGE_LOCK_DAYS = 21       # SURGE锁仓交易日
SURGE_DD_MELT = 0.08       # SURGE回撤熔断 8%
CORR_FILTER = 0.70         # 相关性过滤阈值
MOMENTUM_W = (0.20, 0.40, 0.40)  # 1m/3m/6m 动量权重

# ============================================================
# 数据加载
# ============================================================
def load_all_data(force_refresh=False):
    """从 QMT 前复权数据库加载所有 ETF + 沪深300 指数"""
    all_etfs = OFFENSE_POOL + BROAD_POOL + DEFENSE_POOL

    print("=" * 60)
    print("  从 QMT 前复权库加载 ETF 数据...")
    print("=" * 60)

    # 批量加载 ETF（QMT 适配器自动处理 volume 过滤和前视偏误裁剪）
    code_list = [e["code"] for e in all_etfs]
    etf_data = load_qmt_etfs(code_list, start=START_DATE, end=END_DATE, verbose=True)

    # 提取 close 序列
    prices = {}
    for etf in all_etfs:
        code = etf["code"]
        if code in etf_data:
            prices[code] = etf_data[code]["close"]
            print(f"  {etf['name']} ({code}): {len(etf_data[code])}行 ✓")

    # 沪深300指数（用 000300.SH 或 510300 ETF close 作为代理）
    print("  加载 沪深300指数...", end=" ")
    try:
        hs300 = load_qmt_benchmark("000300.SH", start=START_DATE, end=END_DATE)
        hs300_close = hs300["close"]
        hs300_vol = hs300.get("volume", None)
        print(f"✓ {len(hs300)}行")
    except Exception:
        # fallback: 用 510300 ETF close
        if "sh510300" in prices:
            hs300_close = prices["sh510300"]
            hs300_vol = None
            print(f"✗ 用 sh510300 替代 ({len(hs300_close)}行)")
        else:
            hs300_close, hs300_vol = None, None
            print("✗")

    return prices, hs300_close, hs300_vol


def build_aligned_data(prices, benchmark_close):
    """对齐所有价格数据到共同日期"""
    df = pd.DataFrame(prices)
    df = df.sort_index().ffill()
    common_dates = df.index.intersection(benchmark_close.index)
    df = df.loc[common_dates]
    benchmark = benchmark_close.loc[common_dates]
    return df, benchmark


# ============================================================
# 制度检测（SMA250 + 广度SMA30）
# ============================================================
def detect_regime(benchmark, etf_close, surge_state=None):
    """
    SMA250趋势(60%) + SMA50/SMA250金叉(40%) → 制度分
    + 广度SMA30 SURGE检查

    surge_state: dict with keys 'active', 'days_remaining', 'peak_value'
    """
    sma50 = benchmark.rolling(50).mean()
    sma250 = benchmark.rolling(250).mean()

    last_price = benchmark.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    last_sma250 = sma250.iloc[-1]

    if pd.isna(last_sma250):
        return {"score": 0.5, "regime": "NEUTRAL", "surge": False}

    # 基础制度分
    dev = (last_price - last_sma250) / last_sma250
    trend_score = 0.5 + 0.5 * np.tanh(dev * 10)
    golden_score = 1.0 if last_sma50 > last_sma250 else 0.0
    base_score = 0.6 * trend_score + 0.4 * golden_score

    # ===== SURGE检查 =====
    surge_active = False
    if surge_state and surge_state.get("active"):
        # 仍在锁仓期内
        surge_state["days_remaining"] -= 1
        if surge_state["days_remaining"] > 0:
            surge_active = True
            base_score = max(base_score, 0.72)  # 保持至少RISKON
        else:
            surge_state["active"] = False  # 锁仓期满，退出SURGE
    elif not (surge_state and surge_state.get("active")):
        # 检查是否触发新的SURGE信号
        if _check_surge_trigger(benchmark, etf_close, base_score):
            surge_active = True
            if surge_state is None:
                surge_state = {}
            surge_state.update({"active": True, "days_remaining": SURGE_LOCK_DAYS, "peak_value": None})
            base_score = max(base_score, 0.72)

    # 制度分类
    if base_score >= 0.70:
        regime = "RISKON"
    elif base_score >= 0.50:
        regime = "NEUTRAL"
    elif base_score >= 0.30:
        regime = "RISKOFF"
    else:
        regime = "CRISIS"

    return {"score": base_score, "regime": regime, "surge": surge_active}


def _check_surge_trigger(benchmark, etf_close, base_score):
    """广度SMA30突破检查：15天内进攻池ETF站上SMA50比例从<33%跳到>67%"""
    if base_score < 0.15:
        return False

    sma30 = benchmark.rolling(30).mean()
    sma50 = benchmark.rolling(50).mean()
    sma250 = benchmark.rolling(250).mean()

    last_price = benchmark.iloc[-1]
    last_sma30 = sma30.iloc[-1]
    last_sma50 = sma50.iloc[-1]
    last_sma250 = sma250.iloc[-1]

    if pd.isna(last_sma30) or pd.isna(last_sma250):
        return False

    # SMA30制度分必须≥0.70
    dev_250 = (last_price - last_sma250) / last_sma250
    base_30 = 0.6 * (0.5 + 0.5 * np.tanh(dev_250 * 10)) + 0.4 * (1.0 if last_sma50 > last_sma250 else 0.0)
    if base_30 < 0.15:
        return False

    dev_30 = (last_price - last_sma30) / last_sma30
    sma30_score = 0.6 * (0.5 + 0.5 * np.tanh(dev_30 * 10)) + 0.4 * (1.0 if last_sma50 > last_sma30 else 0.0)
    if sma30_score < 0.70:
        return False

    # 进攻池广度检查
    breadth_now = _calc_breadth(etf_close, -1)
    breadth_15d = _calc_breadth(etf_close, -16)

    if breadth_now is None or breadth_15d is None:
        return False

    return breadth_15d < 0.33 and breadth_now > 0.67


def _calc_breadth(etf_close, offset):
    """计算站上SMA50的ETF比例（只算进攻池中可用的ETF）"""
    count_above = 0
    count_total = 0
    for code in [e["code"] for e in OFFENSE_POOL]:
        if code not in etf_close.columns:
            continue
        col = etf_close[code].dropna()
        if len(col) < 50:
            continue
        try:
            price = col.iloc[offset]
            sma = col.iloc[offset-49:offset+1].mean() if offset < -1 else col.iloc[-50:].mean()
            if price > sma:
                count_above += 1
            count_total += 1
        except (IndexError, ValueError):
            continue

    if count_total < 4:
        return None
    return count_above / count_total


# ============================================================
# 动量选股
# ============================================================
def compute_momentum(df_close, date, pool_codes):
    """计算池内ETF的复合动量分"""
    scores = {}
    for code in pool_codes:
        if code not in df_close.columns:
            continue
        col = df_close[code].loc[:date].dropna()
        if len(col) < 126:
            continue

        price = col.iloc[-1]
        p21 = col.iloc[-22] if len(col) >= 22 else col.iloc[0]
        p63 = col.iloc[-64] if len(col) >= 64 else col.iloc[0]
        p126 = col.iloc[-127] if len(col) >= 127 else col.iloc[0]

        m1 = (price / p21 - 1) if p21 > 0 else 0
        m3 = (price / p63 - 1) if p63 > 0 else 0
        m6 = (price / p126 - 1) if p126 > 0 else 0

        # 波动率标准化（63日）
        rets = col.pct_change().dropna().iloc[-63:]
        vol = rets.std() * np.sqrt(252) if len(rets) >= 20 else 0.3
        if vol < 0.01:
            vol = 0.01

        scores[code] = (MOMENTUM_W[0] * m1 + MOMENTUM_W[1] * m3 + MOMENTUM_W[2] * m6) / vol

    return scores


def filter_by_correlation(df_close, date, selected, scores):
    """相关性过滤：两两相关>0.70时保留高分者"""
    if len(selected) <= 1:
        return selected

    ret_data = {}
    for code in selected:
        col = df_close[code].loc[:date].dropna().iloc[-63:]
        if len(col) >= 21:
            ret_data[code] = col.pct_change().dropna()

    if len(ret_data) < 2:
        return selected

    df_ret = pd.DataFrame(ret_data)
    corr = df_ret.corr()

    # 按动量分从高到低排序
    sorted_codes = sorted(selected, key=lambda c: scores.get(c, 0), reverse=True)
    keep = []
    for code in sorted_codes:
        conflict = False
        for kept in keep:
            if code in corr.index and kept in corr.columns:
                if corr.loc[code, kept] > CORR_FILTER:
                    conflict = True
                    break
        if not conflict:
            keep.append(code)

    return keep


# ============================================================
# 主回测引擎
# ============================================================
def run_backtest(df_close, benchmark_close, use_offense=True, use_surge=True):
    """
    ETF AI Capital 策略回测

    参数:
        df_close: 对齐后的ETF收盘价
        benchmark_close: 沪深300收盘价
        use_offense: 是否使用进攻池（False=纯宽基baseline）
        use_surge: 是否使用SURGE模式
    """
    initial_capital = INITIAL_CAPITAL
    current_positions = {}  # {code: shares}
    cash = initial_capital
    portfolio_values = []
    trades_log = []
    surge_events = []
    regime_log = []

    # 预热期：前252个交易日不交易（计算SMA250用）
    warmup_days = 300

    # 获取每月最后一个交易日
    df_dates = df_close.index
    monthly_dates = df_dates[df_dates.to_series().groupby(
        df_dates.to_period('M')).transform(lambda x: x == x.max())]

    surge_state = {"active": False, "days_remaining": 0, "peak_value": None}

    # --- 定义各资产的代码 ---
    offense_codes = [e["code"] for e in OFFENSE_POOL]
    high_beta_codes = list(HIGH_BETA_CODES)
    broad_codes = [e["code"] for e in BROAD_POOL]
    defense_map = {e["code"]: e["name"] for e in DEFENSE_POOL}
    bond_code = "sh511010"
    corp_bond_code = "sh511220"
    gold_code = "sh518880"
    cash_code = "sh511880"

    for i, date in enumerate(df_dates):
        etf_slice = df_close.loc[date]
        bench_slice = benchmark_close.loc[:date]

        # --- 计算当日组合价值 ---
        position_value = 0.0
        for sym, shares in current_positions.items():
            if sym in etf_slice.index and not pd.isna(etf_slice[sym]):
                position_value += shares * etf_slice[sym]
        portfolio_value = cash + position_value
        portfolio_values.append({"date": date, "value": portfolio_value})

        # --- SURGE回撤监控（每日） ---
        if surge_state.get("active") and use_surge:
            if surge_state["peak_value"] is None or portfolio_value > surge_state["peak_value"]:
                surge_state["peak_value"] = portfolio_value
            if surge_state["peak_value"] > 0:
                dd = (portfolio_value - surge_state["peak_value"]) / surge_state["peak_value"]
                if dd < -SURGE_DD_MELT:
                    surge_state["active"] = False
                    surge_state["days_remaining"] = 0
                    if use_surge:
                        pass  # 回撤熔断触发，下个调仓日会切换

        # --- 月度调仓 ---
        if date not in monthly_dates or i < warmup_days:
            continue

        # 制度检测
        regime_result = detect_regime(bench_slice, df_close.loc[:date], surge_state if use_surge else None)
        regime = regime_result["regime"]
        surge_now = regime_result.get("surge", False)
        regime_log.append({"date": date, "regime": regime, "score": regime_result["score"], "surge": surge_now})

        # --- 确定目标仓位 ---
        target_weights = {}
        is_surge = surge_now and use_surge and use_offense

        if is_surge:
            # SURGE: 高Beta子池，动量Top-4，动量加权，100%权益
            pool = [c for c in high_beta_codes if c in df_close.columns]
            scores = compute_momentum(df_close, date, pool)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_codes = [c for c, s in ranked[:4] if s > -999]

            # 动量加权（归一化，单只≤30%）
            top_scores = {c: max(scores[c], -10) + 11 for c in top_codes}  # 平移为正
            total_s = sum(top_scores.values())
            if total_s > 0:
                raw_w = {c: s / total_s for c, s in top_scores.items()}
            else:
                raw_w = {c: 1.0 / len(top_codes) for c in top_codes}

            for c in top_codes:
                w = min(raw_w[c], 0.30)
                target_weights[c] = w

            # 超出30%的部分重新分配
            excess = 1.0 - sum(target_weights.values())
            eligible = [c for c in target_weights if target_weights[c] < 0.30]
            if eligible and excess > 0.001:
                per_extra = excess / len(eligible)
                for c in eligible:
                    new_w = min(target_weights[c] + per_extra, 0.30)
                    target_weights[c] = new_w

            if surge_state.get("active") and surge_events and surge_events[-1].get("end_date") is None:
                pass
            else:
                surge_events.append({
                    "start_date": date,
                    "end_date": None,
                    "start_value": portfolio_value,
                    "end_value": None,
                    "pnl_pct": None,
                    "melted": False,
                })

        elif regime == "RISKON" and use_offense:
            # 普通RISKON: 进攻池动量Top-6，等权，90%权益
            pool = [c for c in offense_codes if c in df_close.columns]
            scores = compute_momentum(df_close, date, pool)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_codes = [c for c, s in ranked[:8]]  # 多选2个给相关性过滤留余地
            top_codes = filter_by_correlation(df_close, date, top_codes, scores)[:6]

            per_w = 0.90 / max(len(top_codes), 1)
            for c in top_codes:
                target_weights[c] = min(per_w, 0.18)

            # 5%缓冲
            target_weights[bond_code] = 0.05
            target_weights[cash_code] = 0.05

        elif regime == "RISKON" and not use_offense:
            # Baseline: 宽基等权，100%权益
            available_broad = [c for c in broad_codes if c in df_close.columns]
            if available_broad:
                per_w = 1.0 / len(available_broad)
            else:
                available_broad, per_w = [], 0
            for c in available_broad:
                target_weights[c] = per_w

        elif regime == "NEUTRAL":
            # 宽基55% + 防御35% + 现金/黄金10%
            available_broad = [c for c in broad_codes if c in df_close.columns]
            n = max(len(available_broad), 1)
            per_w = 0.55 / n
            for c in available_broad:
                target_weights[c] = per_w
            target_weights[bond_code] = 0.25
            if corp_bond_code in df_close.columns:
                target_weights[corp_bond_code] = 0.10
            else:
                target_weights[bond_code] = target_weights.get(bond_code, 0) + 0.10
            target_weights[gold_code] = 0.05
            target_weights[cash_code] = 0.05

        elif regime == "RISKOFF":
            available_broad = [c for c in broad_codes if c in df_close.columns]
            n = max(len(available_broad), 1)
            per_w = 0.15 / n
            for c in available_broad:
                target_weights[c] = per_w
            target_weights[bond_code] = 0.40
            if corp_bond_code in df_close.columns:
                target_weights[corp_bond_code] = 0.15
            else:
                target_weights[bond_code] = target_weights.get(bond_code, 0) + 0.15
            target_weights[gold_code] = 0.15
            target_weights[cash_code] = 0.15

        else:  # CRISIS
            target_weights[bond_code] = 0.65
            target_weights[gold_code] = 0.15
            target_weights[cash_code] = 0.20

        # --- 执行调仓 ---
        # 先卖（不再需要的）
        for sym in list(current_positions.keys()):
            if sym not in target_weights or target_weights.get(sym, 0) == 0:
                if sym in etf_slice.index and not pd.isna(etf_slice[sym]):
                    shares = current_positions[sym]
                    price = etf_slice[sym]
                    cash += shares * price * (1 - ETF_COST)
                    del current_positions[sym]

        # 再买（调整到目标权重）
        for sym, weight in target_weights.items():
            if weight <= 0:
                continue
            if sym not in etf_slice.index or pd.isna(etf_slice[sym]):
                continue

            target_value = portfolio_value * weight
            price = etf_slice[sym]
            target_shares = int(target_value / price / 100) * 100  # A股ETF 100股整数倍
            current_shares = current_positions.get(sym, 0)
            diff = target_shares - current_shares

            if diff > 0:
                cost = diff * price * (1 + ETF_COST)
                if cost <= cash:
                    cash -= cost
                    current_positions[sym] = target_shares
            elif diff < 0:
                cash += abs(diff) * price * (1 - ETF_COST)
                current_positions[sym] = target_shares

        # SURGE退出处理
        if use_surge and surge_events and surge_events[-1].get("end_date") is None:
            if not surge_state.get("active"):
                surge_events[-1]["end_date"] = date
                surge_events[-1]["end_value"] = portfolio_value
                if surge_events[-1]["start_value"] > 0:
                    surge_events[-1]["pnl_pct"] = (portfolio_value - surge_events[-1]["start_value"]) / surge_events[-1]["start_value"] * 100

    # ===== 计算指标 =====
    df_values = pd.DataFrame(portfolio_values).set_index("date")
    df_values["return"] = df_values["value"].pct_change()

    metrics = _compute_metrics(df_values, initial_capital)

    return {
        "values": df_values,
        "metrics": metrics,
        "regime_log": pd.DataFrame(regime_log),
        "surge_events": surge_events,
    }


# ============================================================
# 买入持有基准
# ============================================================
def run_buy_hold(benchmark_close):
    """买入持有沪深300"""
    initial = INITIAL_CAPITAL
    rets = benchmark_close.pct_change().dropna()
    values = pd.DataFrame({
        "value": initial * (1 + rets).cumprod()
    }, index=rets.index)
    values["return"] = values["value"].pct_change()
    return {"values": values, "metrics": _compute_metrics(values, initial)}


# ============================================================
# 指标计算
# ============================================================
def _compute_metrics(df_values, initial_capital):
    returns = df_values["return"].dropna()
    if returns.empty:
        return {}

    total_ret = df_values["value"].iloc[-1] / initial_capital - 1
    n_years = len(returns) / 252
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    excess = returns - 0.02 / 252
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    annual_returns = {}
    for yr, grp in returns.groupby(returns.index.year):
        annual_returns[yr] = ((1 + grp).prod() - 1) * 100

    return {
        "total_return": total_ret * 100,
        "annual_return": annual_ret * 100,
        "max_drawdown": max_dd * 100,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "volatility": returns.std() * np.sqrt(252) * 100,
        "annual_returns": annual_returns,
    }


# ============================================================
# 报告生成
# ============================================================
def print_report(name, result, benchmark_annual=None):
    m = result["metrics"]
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    print(f"  总收益:    {m['total_return']:+.1f}%")
    print(f"  年化收益:  {m['annual_return']:+.1f}%")
    print(f"  最大回撤:  {m['max_drawdown']:+.1f}%")
    print(f"  年化波动:  {m['volatility']:+.1f}%")
    print(f"  夏普比率:  {m['sharpe_ratio']:.2f}")
    print(f"  卡玛比率:  {m['calmar_ratio']:.2f}")

    # 逐年收益
    ar = m.get("annual_returns", {})
    if ar:
        print(f"\n  逐年收益:")
        years = sorted(ar.keys())
        for yr in years:
            ret = ar[yr]
            if benchmark_annual and yr in benchmark_annual:
                diff = ret - benchmark_annual[yr]
                print(f"    {yr}: {ret:+6.1f}%  (vs 沪深300 {benchmark_annual[yr]:+6.1f}%, 超额 {diff:+6.1f}%)")
            else:
                print(f"    {yr}: {ret:+6.1f}%")

    return m


def print_surge_summary(surge_events):
    """打印 SURGE 事件记录"""
    if not surge_events:
        print("\n  🚀 SURGE 事件: 无触发")
        return

    print(f"\n  🚀 SURGE 事件记录 ({len(surge_events)}次):")
    total_pnl = 0
    for i, ev in enumerate(surge_events):
        sd = ev["start_date"].strftime("%Y-%m-%d") if hasattr(ev["start_date"], "strftime") else str(ev["start_date"])
        ed = ev["end_date"].strftime("%Y-%m-%d") if ev["end_date"] and hasattr(ev["end_date"], "strftime") else "进行中"
        pnl = ev.get("pnl_pct")
        if pnl is not None:
            total_pnl += pnl
            print(f"    #{i+1}: {sd} → {ed} | 盈亏 {pnl:+.1f}%")
        else:
            print(f"    #{i+1}: {sd} → {ed} | 未完成")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ETF AI Capital 策略 — 完整回测")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    # ---- 加载数据 ----
    prices, hs300_close, hs300_vol = load_all_data(force_refresh=False)

    if hs300_close is None or not prices:
        print("❌ 数据加载失败，无法继续")
        sys.exit(1)

    df_close, benchmark = build_aligned_data(prices, hs300_close)
    print(f"\n✅ 对齐后: {len(df_close)} 个交易日, {len(df_close.columns)} 只ETF")

    # 打印各ETF的数据范围
    print("\n📅 ETF 数据范围:")
    for col in df_close.columns:
        valid = df_close[col].dropna()
        if len(valid) > 0:
            # 找ETF名称
            etf_name = col
            for e in ALL_ETFS:
                if e["code"] == col:
                    etf_name = e["name"]
                    break
            print(f"  {etf_name} ({col}): {valid.index[0].strftime('%Y-%m-%d')} ~ {valid.index[-1].strftime('%Y-%m-%d')} ({len(valid)}行)")

    # ---- (A) 买入持有 ----
    result_bh = run_buy_hold(benchmark)

    # ---- (B) 纯宽基制度择时 ----
    print("\n⏳ 回测 (B) 纯宽基制度择时...")
    result_baseline = run_backtest(df_close, benchmark, use_offense=False, use_surge=False)

    # ---- (C) ETF AI Capital 完整策略 ----
    print("⏳ 回测 (C) ETF AI Capital 完整策略...")
    result_full = run_backtest(df_close, benchmark, use_offense=True, use_surge=True)

    # ---- (D) ETF AI Capital 无SURGE ----
    print("⏳ 回测 (D) ETF AI Capital (仅行业轮动，无SURGE)...")
    result_no_surge = run_backtest(df_close, benchmark, use_offense=True, use_surge=False)

    # ============================================================
    # 综合报告
    # ============================================================
    print("\n")
    print("=" * 60)
    print("  📊 回测结果对比")
    print("=" * 60)

    m_bh = print_report("(A) 买入持有沪深300", result_bh)
    m_bl = print_report("(B) 纯宽基制度择时（baseline）", result_baseline, m_bh.get("annual_returns"))
    m_ns = print_report("(C) ETF 行业轮动（无 SURGE）", result_no_surge, m_bh.get("annual_returns"))
    m_full = print_report("(D) ETF AI Capital 完整策略", result_full, m_bh.get("annual_returns"))

    # ---- SURGE 总结 ----
    print_surge_summary(result_full.get("surge_events", []))

    # ---- 制度分布 ----
    rl = result_full.get("regime_log", pd.DataFrame())
    if not rl.empty:
        print(f"\n  📊 制度分布 (完整策略):")
        for r in ["RISKON", "NEUTRAL", "RISKOFF", "CRISIS"]:
            cnt = (rl["regime"] == r).sum()
            total = len(rl)
            pct = cnt / total * 100 if total > 0 else 0
            surge_cnt = ((rl["regime"] == r) & (rl["surge"] == True)).sum() if "surge" in rl.columns else 0
            marker = f" (含SURGE: {surge_cnt})" if surge_cnt > 0 else ""
            print(f"    {r}: {cnt}/{total} ({pct:.0f}%){marker}")

    # ---- 汇总表 ----
    print(f"\n{'=' * 60}")
    print(f"  📋 关键指标汇总")
    print(f"{'=' * 60}")
    print(f"  {'指标':<16s} {'(A)买入持有':>12s} {'(B)宽基择时':>12s} {'(C)行业轮动':>12s} {'(D)完整策略':>12s}")
    print(f"  {'─' * 64}")
    for key, label in [("annual_return", "年化收益"), ("max_drawdown", "最大回撤"),
                        ("sharpe_ratio", "夏普比率"), ("calmar_ratio", "卡玛比率"),
                        ("volatility", "年化波动"), ("total_return", "总收益")]:
        vals = [m_bh.get(key, 0), m_bl.get(key, 0), m_ns.get(key, 0), m_full.get(key, 0)]
        if key in ("annual_return", "max_drawdown", "volatility", "total_return"):
            print(f"  {label:<16s} {vals[0]:+11.1f}% {vals[1]:+11.1f}% {vals[2]:+11.1f}% {vals[3]:+11.1f}%")
        else:
            print(f"  {label:<16s} {vals[0]:>12.2f} {vals[1]:>12.2f} {vals[2]:>12.2f} {vals[3]:>12.2f}")

    # ---- 净值曲线保存 ----
    values = pd.DataFrame({
        "buy_hold": result_bh["values"]["value"],
        "baseline": result_baseline["values"]["value"],
        "no_surge": result_no_surge["values"]["value"],
        "full_strategy": result_full["values"]["value"],
    })
    # 对齐到共同索引
    values = values.dropna()

    out_path = os.path.join(os.path.dirname(__file__), "backtests", "etf_strategy_values.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    values.to_csv(out_path)
    print(f"\n💾 净值数据已保存: {out_path}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ 回测完成")
    print(f"{'=' * 60}")
