# -*- coding: utf-8 -*-
"""
可转债日内研究 - 每日采集器

用法(在项目根目录):
  venv/bin/python -m research.cb_intraday.collect --freq 60          # 采集全市场 60 分钟
  venv/bin/python -m research.cb_intraday.collect --freq 5,15,60     # 多频率
  venv/bin/python -m research.cb_intraday.collect --freq 1           # 东财 1 分钟(仅当日, 建议每日收盘后跑)
  venv/bin/python -m research.cb_intraday.collect --freq 60 --limit 20  # 调试: 只采前20只

窗口约束(错过即丢, 见 DATA_QUALITY.md):
  1分钟=仅当日 | 5分钟=21个交易日 | 15分钟=3个月 | 60分钟=1年
  → 1 分钟必须每日采; 5 分钟至少每两周采一次; 15/60 分钟每月采即可不断档。
"""
import argparse
import functools
import sys
import time

from . import sources, store

# cron/后台运行时管道缓冲会吞掉进度日志, 强制逐行刷新
print = functools.partial(print, flush=True)  # noqa: A001


def collect(freqs, limit=None, universe=None):
    uni = universe if universe is not None else sources.sina_universe()
    symbols = list(uni["symbol"])
    if limit:
        symbols = symbols[:limit]
    print(f"[universe] 在市转债 {len(uni)} 只, 本次采集 {len(symbols)} 只, 频率 {freqs}")

    stats = {f: {"ok": 0, "empty": 0, "fail": 0, "added": 0} for f in freqs}
    em_fail_streak = 0  # 东财熔断: 连挂3次后本轮直接走腾讯兜底, 避免反复撞风控
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        for f in freqs:
            try:
                if f == "1":
                    df = None
                    if em_fail_streak < 3:
                        try:
                            df = sources.em_trends_1min(sym)
                            em_fail_streak = 0
                        except Exception:  # noqa: BLE001
                            em_fail_streak += 1
                            if em_fail_streak == 3:
                                print("  ! 东财连续失败, 本轮剩余券直接用腾讯m1兜底")
                    if df is None:
                        # 腾讯 m1 兜底(320根≈1.3天, 无成交额字段)
                        df = sources.tencent_mkline(sym, "m1", 320)
                else:
                    df = sources.sina_kline(sym, scale=int(f))
                if df.empty:
                    stats[f]["empty"] += 1
                else:
                    stats[f]["added"] += store.upsert(sym, f, df)
                    stats[f]["ok"] += 1
            except Exception as e:  # noqa: BLE001 - 单券失败不阻塞全场
                stats[f]["fail"] += 1
                print(f"  ! {sym} freq={f} FAIL {type(e).__name__}: {str(e)[:80]}")
        if i % 25 == 0 or i == len(symbols):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(symbols) - i)
            print(f"  [{i}/{len(symbols)}] elapsed={elapsed / 60:.1f}min eta={eta / 60:.1f}min "
                  + " ".join(f"{f}min(ok={s['ok']},+{s['added']})" for f, s in stats.items()))
    for f, s in stats.items():
        print(f"[done freq={f}] ok={s['ok']} empty={s['empty']} fail={s['fail']} 新增bar={s['added']}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="可转债分钟数据采集")
    ap.add_argument("--freq", default="60", help="逗号分隔: 1,5,15,30,60")
    ap.add_argument("--limit", type=int, default=None, help="调试: 只采前N只")
    args = ap.parse_args()
    freqs = [f.strip() for f in args.freq.split(",") if f.strip()]
    stats = collect(freqs, limit=args.limit)
    total_fail = sum(s["fail"] for s in stats.values())
    sys.exit(1 if total_fail > len(stats) * 10 else 0)


if __name__ == "__main__":
    main()
