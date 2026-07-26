"""
ETF 动量轮动 + EPO + 风险平价 策略回测 — QMT 数据库版
数据源: qmt_qfq.db (前复权)
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from src.qmt_adapter import load_qmt_etfs, to_qmt_code, bare_code

# ============================================================
# ETF 池 — 覆盖商品/宽基/行业/防御
# ============================================================
ETF_POOL = {
    # 商品
    '黄金ETF':    {'code': 'sh518880'},
    # 宽基
    '沪深300':    {'code': 'sh510300'},
    '中证500':    {'code': 'sh510500'},
    '创业板':     {'code': 'sz159915'},
    '科创50':     {'code': 'sh588000'},
    # 行业
    '军工ETF':    {'code': 'sh512660'},
    '光伏ETF':    {'code': 'sh515790'},
    '5GETF':      {'code': 'sh515050'},
    '半导体ETF':  {'code': 'sh512760'},
    '医药ETF':    {'code': 'sh512010'},
    '芯片ETF':    {'code': 'sz159995'},
    '新能源ETF':  {'code': 'sh516160'},
    '证券ETF':    {'code': 'sh512880'},
    # 防御
    '国债ETF':    {'code': 'sh511010'},
    '货币ETF':    {'code': 'sh511880'},
}

# ============================================================
# 策略参数（完全对齐原策略）
# ============================================================
MOMENTUM_DAYS = 34      # 动量回溯天数
TOP_N = 3               # 持仓数量
EPO_LOOKBACK = 1200     # EPO 回看天数（约5年）
RISK_PARITY_BLEND = 0.4 # EPO/RP 混合比例

# 交易成本
COMMISSION = 0.0002     # ETF 佣金万2
SLIPPAGE = 0.0005       # 滑点万5
ETF_COST = COMMISSION + SLIPPAGE  # 单边总成本

# 回测区间
START_DATE = '2020-04-28'
END_DATE = '2026-07-13'
INITIAL_CAPITAL = 100_000
MIN_TRADE_VALUE = 5000   # 最小交易金额，低于此跳过

# ============================================================
# 数据加载
# ============================================================
def load_data():
    """从 QMT 前复权库加载 ETF 日线数据"""
    print("=" * 60)
    print("  从 QMT 前复权库加载 ETF 数据...")
    print("=" * 60)

    code_list = [info['code'] for info in ETF_POOL.values()]
    etf_data = load_qmt_etfs(code_list, start=START_DATE, end=END_DATE, verbose=False)

    all_closes = {}
    available = {}

    for name, info in ETF_POOL.items():
        code = info['code']
        if code in etf_data:
            df = etf_data[code]
            close = df['close']
            if len(close) >= MOMENTUM_DAYS + 1:
                all_closes[name] = close
                available[name] = info
                print(f"  ✓ {name:10s} ({code}): {len(close):5d} 天, "
                      f"{close.index[0].strftime('%Y-%m-%d')} ~ {close.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"  ✗ {name:10s} ({code}): 数据不足 ({len(close)}天)")
        else:
            print(f"  ✗ {name:10s} ({code}): 未加载")

    print(f"\n  可用ETF: {len(available)}/{len(ETF_POOL)}")
    return all_closes, available

# ============================================================
# 动量打分：对数线性回归 斜率 × R²
# ============================================================
def calc_momentum_score(prices_series, date, exclude_today=True):
    """
    对给定日期，取过去 MOMENTUM_DAYS 天的收盘价，
    做对数线性回归，返回 (年化收益率, R², 得分)

    exclude_today=True: 使用 date 之前的数据（排除当日），避免前视偏差
                        对齐 JoinQuant attribute_history() 行为
    """
    # 取该日期之前的收盘价（严格排除当日，避免前视偏差）
    if exclude_today:
        prices_before = prices_series[prices_series.index < date]
    else:
        prices_before = prices_series[prices_series.index <= date]

    if len(prices_before) < MOMENTUM_DAYS + 1:
        return None

    recent = prices_before.iloc[-(MOMENTUM_DAYS + 1):]

    try:
        y = np.log(np.array(recent.values, dtype=float))
        x = np.arange(len(y), dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        annualized_returns = np.exp(slope * 250) - 1

        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.var(y, ddof=1) * (len(y) - 1)
        if ss_tot == 0:
            return None
        r_squared = 1 - ss_res / ss_tot

        score = annualized_returns * r_squared
        return (annualized_returns, r_squared, score)
    except:
        return None

# ============================================================
# 风险平价权重（逆波动率）
# ============================================================
def risk_parity_weights(closes_dict, names, date, lookback=60):
    """逆波动率等风险贡献权重"""
    vols = []
    for name in names:
        series = closes_dict[name]
        prices_before = series[series.index < date]  # 排除当日
        if len(prices_before) < lookback:
            return None
        recent = prices_before.iloc[-lookback:]
        rets = recent.pct_change().dropna()
        vol = rets.std()
        vols.append(vol if vol > 0 else 1e-10)

    inv_vols = [1.0 / v for v in vols]
    total = sum(inv_vols)
    return np.array([iv / total for iv in inv_vols])

# ============================================================
# EPO 优化权重（简化版 anchored EPO）
# ============================================================
def epo_weights(closes_dict, names, date, lookback=EPO_LOOKBACK):
    """Anchored EPO 优化"""
    n = len(names)
    if n == 0:
        return None
    if n == 1:
        return np.array([1.0])

    # 获取对齐的收益率数据（排除当日）
    returns_list = []
    for name in names:
        series = closes_dict[name]
        prices_before = series[series.index < date]  # 排除当日
        if len(prices_before) < lookback + 1:
            actual_len = len(prices_before) - 1
        else:
            actual_len = lookback
        if actual_len < 60:
            return None
        recent = prices_before.iloc[-(actual_len + 1):]
        rets = recent.pct_change().dropna()
        returns_list.append(rets.values)

    # 对齐长度
    min_len = min(len(r) for r in returns_list)
    if min_len < 60:
        return None
    returns_matrix = np.column_stack([r[-min_len:] for r in returns_list])

    try:
        # 协方差和相关矩阵
        vcov = np.cov(returns_matrix, rowvar=False)
        std = np.sqrt(np.diag(np.diag(vcov)))

        # 收缩相关矩阵（w=0.2）
        w_shrink = 0.2
        I = np.eye(n)

        try:
            corr = np.corrcoef(returns_matrix, rowvar=False)
            if np.any(np.isnan(corr)):
                corr = I
        except:
            corr = I

        shrunk_cor = (1 - w_shrink) * corr + w_shrink * I
        cov_tilde = std @ shrunk_cor @ std

        # 信号 = 均值收益
        signal = np.array([np.mean(r) for r in returns_list])

        # 锚定权重 = 逆方差
        d = np.diag(vcov)
        if np.all(d > 0):
            inv_d = 1.0 / d
            anchor = inv_d / np.sum(inv_d)
        else:
            anchor = np.ones(n) / n

        # Anchored EPO (endogenous gamma)
        try:
            inv_cov = np.linalg.solve(cov_tilde, I)
        except:
            inv_cov = np.linalg.pinv(cov_tilde)

        num = np.sqrt(anchor.T @ cov_tilde @ anchor)
        den_term = signal.T @ inv_cov @ cov_tilde @ inv_cov @ signal
        den = np.sqrt(max(den_term, 1e-10))
        gamma = num / den if den > 0 else 1.0

        epo = inv_cov @ ((1 - w_shrink) * gamma * signal + w_shrink * I @ np.diag(np.diag(vcov)) @ anchor)

        # 归一化，负权重置零
        epo = np.maximum(epo, 0)
        total = np.sum(epo)
        if total > 0:
            epo = epo / total
        else:
            epo = np.ones(n) / n

        return epo
    except:
        return None

# ============================================================
# 主回测循环
# ============================================================
def run_backtest():
    all_closes, available = load_data()
    if len(available) < 3:
        print("ETF数量不足，无法运行回测")
        return

    # 获取所有交易日（取第一个ETF的日期索引）
    first_etf = list(available.keys())[0]
    all_dates = all_closes[first_etf].index.tolist()
    start_idx = 0
    for i, d in enumerate(all_dates):
        if d >= pd.Timestamp(START_DATE):
            start_idx = i
            break

    # 初始化
    cash = INITIAL_CAPITAL
    positions = {}   # {name: shares}
    portfolio_values = []
    trades_log = []
    rebalance_dates = []

    warmup = max(EPO_LOOKBACK, MOMENTUM_DAYS * 2)

    print(f"\n{'='*60}")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print(f"  初始资金: {INITIAL_CAPITAL:,}")
    print(f"  可用ETF: {len(available)} 只")
    print(f"  动量窗口: {MOMENTUM_DAYS} 天")
    print(f"  持仓数: {TOP_N} 只")
    print(f"  EPO/RP混合: {1-RISK_PARITY_BLEND:.0%}/{RISK_PARITY_BLEND:.0%}")
    print(f"{'='*60}")

    monthly_count = 0
    last_month = None

    for i in range(start_idx, len(all_dates)):
        date = all_dates[i]
        current_month = (date.year, date.month)

        # 计算当前组合市值
        position_value = 0
        for name, shares in list(positions.items()):
            if name in all_closes:
                price_series = all_closes[name]
                prices_before = price_series[price_series.index <= date]
                if len(prices_before) > 0:
                    position_value += shares * prices_before.iloc[-1]

        total_value = cash + position_value
        portfolio_values.append({'date': date, 'value': total_value})

        # 月度调仓（每月首次）
        is_new_month = (current_month != last_month)
        if not is_new_month or i < warmup:
            last_month = current_month
            continue

        last_month = current_month
        monthly_count += 1

        # 1. 动量打分
        scores = {}
        for name in available:
            result = calc_momentum_score(all_closes[name], date)
            if result is not None:
                scores[name] = result

        # 过滤负分，排序
        ranked = [(name, s[2]) for name, s in scores.items() if s[2] > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)

        # 取前 N
        target_names = [name for name, _ in ranked[:TOP_N]]

        if len(target_names) == 0:
            # 全部动量<=0，清仓
            if monthly_count <= 3:
                print(f"  {date.strftime('%Y-%m-%d')}: 无正动量ETF, 空仓")
            for name in list(positions.keys()):
                price_series = all_closes[name]
                prices_before = price_series[price_series.index <= date]
                if len(prices_before) > 0:
                    sell_price = prices_before.iloc[-1]
                    sell_shares = positions[name]
                    sell_value = sell_shares * sell_price
                    cost = sell_value * ETF_COST
                    cash += sell_value - cost
                    trades_log.append({
                        'date': date, 'action': 'SELL', 'name': name,
                        'price': sell_price, 'shares': sell_shares, 'value': sell_value, 'cost': cost
                    })
                del positions[name]
            if monthly_count <= 3:
                print(f"  {date.strftime('%Y-%m-%d')}: 清仓完成, 现金 {cash:,.0f}")
            continue

        # 2. 计算权重
        rp_w = risk_parity_weights(all_closes, target_names, date)
        epo_w = epo_weights(all_closes, target_names, date)

        if epo_w is not None and rp_w is not None:
            weights = (1 - RISK_PARITY_BLEND) * epo_w + RISK_PARITY_BLEND * rp_w
        elif epo_w is not None:
            weights = epo_w
        elif rp_w is not None:
            weights = rp_w
        else:
            weights = np.ones(len(target_names)) / len(target_names)

        weights = weights / weights.sum()

        # 3. 执行调仓
        # 先卖出不在目标中的
        for name in list(positions.keys()):
            if name not in target_names:
                price_series = all_closes[name]
                prices_before = price_series[price_series.index <= date]
                if len(prices_before) > 0:
                    sell_price = prices_before.iloc[-1]
                    sell_shares = positions[name]
                    sell_value = sell_shares * sell_price
                    cost = sell_value * ETF_COST
                    cash += sell_value - cost
                    trades_log.append({
                        'date': date, 'action': 'SELL', 'name': name,
                        'price': sell_price, 'shares': sell_shares, 'value': sell_value, 'cost': cost
                    })
                del positions[name]

        # 再调整目标持仓
        for j, name in enumerate(target_names):
            price_series = all_closes[name]
            prices_before = price_series[price_series.index <= date]
            if len(prices_before) == 0:
                continue
            price = prices_before.iloc[-1]

            target_value = total_value * weights[j]

            # 现有持仓市值
            current_shares = positions.get(name, 0)
            current_value = current_shares * price

            diff = target_value - current_value

            if diff > MIN_TRADE_VALUE:
                # 买入
                buy_shares = int(diff / price / 100) * 100
                if buy_shares > 0:
                    buy_value = buy_shares * price
                    cost = buy_value * ETF_COST
                    if buy_value + cost <= cash:
                        cash -= buy_value + cost
                        positions[name] = current_shares + buy_shares
                        trades_log.append({
                            'date': date, 'action': 'BUY', 'name': name,
                            'price': price, 'shares': buy_shares, 'value': buy_value, 'cost': cost
                        })
            elif diff < -MIN_TRADE_VALUE:
                # 卖出
                sell_shares = min(int(-diff / price / 100) * 100, current_shares)
                if sell_shares > 0:
                    sell_value = sell_shares * price
                    cost = sell_value * ETF_COST
                    cash += sell_value - cost
                    positions[name] = current_shares - sell_shares
                    trades_log.append({
                        'date': date, 'action': 'SELL', 'name': name,
                        'price': price, 'shares': sell_shares, 'value': sell_value, 'cost': cost
                    })

        # 首次几次调仓打印日志
        if monthly_count <= 3:
            score_str = ', '.join([f'{name}({score:.3f})' for name, score in ranked[:TOP_N]])
            weight_str = ', '.join([f'{target_names[j]}:{weights[j]:.1%}' for j in range(len(target_names))])
            print(f"  [{date.strftime('%Y-%m-%d')}] 排名: {score_str}")
            print(f"    权重: {weight_str}, 总资产: {total_value:,.0f}")

    # 计算最终市值
    final_position_value = 0
    final_date = all_dates[-1]
    for name, shares in positions.items():
        if name in all_closes:
            prices_before = all_closes[name][all_closes[name].index <= final_date]
            if len(prices_before) > 0:
                final_position_value += shares * prices_before.iloc[-1]

    final_value = cash + final_position_value

    # ============================================================
    # 绩效计算
    # ============================================================
    df_values = pd.DataFrame(portfolio_values)
    df_values.set_index('date', inplace=True)

    # 日收益率
    df_values['daily_return'] = df_values['value'].pct_change()

    total_days = len(df_values)
    years = total_days / 250

    total_return = (final_value / INITIAL_CAPITAL - 1)
    annual_return = (final_value / INITIAL_CAPITAL) ** (1 / years) - 1

    # 夏普比率
    risk_free_daily = 0.02 / 250  # 2% 无风险利率
    excess = df_values['daily_return'] - risk_free_daily
    sharpe = np.sqrt(250) * excess.mean() / excess.std() if excess.std() > 0 else 0

    # 最大回撤
    cummax = df_values['value'].cummax()
    drawdown = (df_values['value'] - cummax) / cummax
    max_dd = drawdown.min()

    # 胜率（月度）
    monthly_returns = df_values['value'].resample('M').last().pct_change().dropna()
    win_rate = (monthly_returns > 0).mean()

    # 年化波动率
    annual_vol = df_values['daily_return'].std() * np.sqrt(250)

    # 卡玛比率
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # 交易统计
    df_trades = pd.DataFrame(trades_log)
    if len(df_trades) > 0:
        total_cost = df_trades['cost'].sum()
        total_turnover = df_trades['value'].sum()
    else:
        total_cost = 0
        total_turnover = 0

    # ============================================================
    # 基准对比：等权持有所有ETF
    # ============================================================
    bh_returns = []
    for date in df_values.index:
        day_ret = 0
        count = 0
        for name in available:
            series = all_closes[name]
            prices_before = series[series.index <= date]
            if len(prices_before) >= 2:
                ret = prices_before.iloc[-1] / prices_before.iloc[-2] - 1
                day_ret += ret
                count += 1
        if count > 0:
            bh_returns.append(day_ret / count)
        else:
            bh_returns.append(0)

    bh_values = [INITIAL_CAPITAL]
    for r in bh_returns:
        bh_values.append(bh_values[-1] * (1 + r))
    bh_final = bh_values[-1]
    bh_annual = (bh_final / INITIAL_CAPITAL) ** (1 / years) - 1

    # ============================================================
    # 打印结果
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  回测结果")
    print(f"{'='*60}")
    print(f"  回测区间: {df_values.index[0].strftime('%Y-%m-%d')} ~ {df_values.index[-1].strftime('%Y-%m-%d')}")
    print(f"  回测年数: {years:.1f} 年")
    print(f"  月度调仓: {monthly_count} 次")
    print(f"  总交易次数: {len(trades_log)}")
    print(f"  总交易成本: {total_cost:,.0f}")
    print(f"")
    print(f"  {'指标':<20} {'策略':<15} {'等权持有':<15}")
    print(f"  {'-'*50}")
    print(f"  {'最终市值':<20} {final_value:>13,.0f}  {bh_final:>13,.0f}")
    print(f"  {'总收益率':<20} {total_return:>14.2%}  {(bh_final/INITIAL_CAPITAL-1):>14.2%}")
    print(f"  {'年化收益率':<20} {annual_return:>14.2%}  {bh_annual:>14.2%}")
    print(f"  {'年化波动率':<20} {annual_vol:>14.2%}")
    print(f"  {'夏普比率':<20} {sharpe:>14.2f}")
    print(f"  {'最大回撤':<20} {max_dd:>14.2%}")
    print(f"  {'卡玛比率':<20} {calmar:>14.2f}")
    print(f"  {'月胜率':<20} {win_rate:>14.1%}")

    # 年度收益
    print(f"\n  {'='*50}")
    print(f"  年度收益拆解")
    print(f"  {'='*50}")
    annual_returns = df_values['value'].resample('Y').last().pct_change()
    for yr, val in annual_returns.items():
        if not pd.isna(val):
            print(f"  {yr.year}: {val:>8.2%}")

    # ETF 贡献排名（粗略：按被选中的次数）
    print(f"\n  {'='*50}")
    print(f"  ETF 动量得分均值排名")
    print(f"  {'='*50}")

    # 计算每个ETF在所有调仓日的平均动量得分
    avg_scores = {}
    for name in available:
        name_scores = []
        date_list = df_values.index.tolist()
        for d in date_list[::20]:  # 每20个交易日采样
            result = calc_momentum_score(all_closes[name], d)
            if result is not None:
                name_scores.append(result[2])  # score
        if name_scores:
            avg_scores[name] = np.mean(name_scores)

    for name, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:10s}: {score:+.4f}")

    return {
        'final_value': final_value,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_return': total_return,
        'win_rate': win_rate,
        'calmar': calmar,
    }

if __name__ == '__main__':
    result = run_backtest()
    print(f"\n{'='*60}")
    print(f"  最终结论")
    print(f"{'='*60}")
    print(f"  真实数据回测年化: {result['annual_return']:.2%}")
    print(f"  真实数据夏普比率: {result['sharpe']:.2f}")
    print(f"  真实数据最大回撤: {result['max_dd']:.2%}")
    print(f"  真实数据总收益: {result['total_return']:.2%}")
