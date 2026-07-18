# -*- coding: utf-8 -*-
"""
全链路演示: 读取60分钟档案 → 三个基线策略 × 三档成本 → 输出报告表

用法: venv/bin/python -m research.cb_intraday.run_demo
前提: 已运行采集 python -m research.cb_intraday.collect --freq 60
"""
import os

import pandas as pd

from . import config, store
from .costs import COST_SCENARIOS
from .engine import cost_sensitivity
from .signals_demo import DEMO_STRATEGIES


def main():
    bars = store.load_universe("60")
    if bars.empty:
        raise SystemExit("60分钟档案为空, 先运行: python -m research.cb_intraday.collect --freq 60")
    n_sym = bars["symbol"].nunique()
    print(f"[data] {n_sym} 只转债, {len(bars)} 根60分钟bar, "
          f"{bars['datetime'].min()} ~ {bars['datetime'].max()}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_tables = {}
    for name, fn in DEMO_STRATEGIES.items():
        print(f"\n===== {name} =====")
        tbl = cost_sensitivity(bars, fn, COST_SCENARIOS, max_positions=5)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print(tbl.to_string(index=False))
        all_tables[name] = tbl

    out = os.path.join(config.OUTPUT_DIR, "demo_results.md")
    with open(out, "w") as f:
        f.write("# 演示回测结果(基线信号 × 三档成本)\n\n")
        f.write(f"数据: {n_sym} 只在市转债 60 分钟 bar, "
                f"{bars['datetime'].min().date()} ~ {bars['datetime'].max().date()}\n\n")
        f.write("> 定位: 管线验证 + 基准线, **不是可交易策略**。\n\n")
        for name, tbl in all_tables.items():
            f.write(f"## {name}\n\n{tbl.to_markdown(index=False)}\n\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
