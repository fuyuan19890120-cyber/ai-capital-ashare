#!/usr/bin/env python3
"""
本机: 合并 CI 拉回的增量数据到 QMT 库 (重叠日校准 + 自动备份)

用法:
  python3 merge_etf_incremental.py            # dry-run: 只打印校准报告
  python3 merge_etf_incremental.py --commit   # 真正追加(先备份 qmt_qfq.db)

流程: 读 data/etf_incremental/etf_latest.csv → 与 QMT 库重叠日算校准比例
      → 增量(date>QMT最新日)按比例缩放 → 追加。
近期无拆分事件 → 校准比例应 ≈ 1.0。
"""
import os, sys, sqlite3, shutil
from datetime import datetime
import pandas as pd

QMT_DB = os.path.expanduser("~/ai-capital-ashare/data/qmt_qfq.db")
CSV = os.path.expanduser("~/ai-capital-ashare/data/etf_incremental/etf_latest.csv")


def main():
    commit = "--commit" in sys.argv

    if not os.path.exists(CSV):
        print(f"❌ 未找到增量文件 {CSV}")
        print("   请先在 GitHub 触发 workflow 并 git pull 到本机。")
        sys.exit(1)

    qcon = sqlite3.connect(QMT_DB)
    qmax = qcon.execute("SELECT MAX(date) FROM daily").fetchone()[0]
    qmt = pd.read_sql("SELECT code, date, close FROM daily", qcon)
    qcon.close()
    csv = pd.read_csv(CSV, dtype={"date": str})

    print(f"QMT 库最新日期: {qmax}")
    print(f"增量文件: {len(csv)} 行, {csv['code'].nunique()} 只, 区间 {csv['date'].min()}~{csv['date'].max()}")

    report, to_append = [], []
    for code in csv["code"].unique():
        c = csv[csv["code"] == code]
        q = qmt[qmt["code"] == code].set_index("date")["close"]
        overlap = c[c["date"] <= qmax]
        if overlap.empty:
            report.append((code, "无重叠", 0.0, 0, 0))
            continue
        merged = overlap.set_index("date").join(q.rename("qmt_close"), how="inner")
        if merged.empty:
            report.append((code, "重叠无匹配", 0.0, 0, 0))
            continue
        ratio = (merged["qmt_close"] / merged["close"]).median()
        err = (merged["close"] * ratio - merged["qmt_close"]).abs() / merged["qmt_close"]
        inc = c[c["date"] > qmax].copy()
        if not inc.empty:
            for col in ["open", "high", "low", "close"]:
                inc[col] = inc[col] * ratio
            to_append.append(inc)
        report.append((code, f"{ratio:.4f}", float(err.max()), len(overlap), len(inc)))

    print(f"\n{'code':<10} {'校准比例':>8} {'重叠最大误差%':>12} {'重叠天数':>6} {'新增天数':>6}")
    for code, ratio, err, n_over, n_inc in report:
        flag = "  ⚠️" if (isinstance(err, float) and err > 0.01) else ""
        print(f"{code:<10} {ratio:>8} {err*100 if isinstance(err,float) else 0:>12.2f} {n_over:>6} {n_inc:>6}{flag}")

    if not to_append:
        print("\n无新增数据 (QMT 库已最新), 退出")
        return

    total_inc = sum(len(x) for x in to_append)
    print(f"\n增量合计 {total_inc} 行")
    if not commit:
        print("[dry-run] 未写入。确认校准比例≈1.0、无⚠️后, 用 --commit 追加。")
        return

    bak = f"{QMT_DB}.bak_{datetime.now().strftime('%Y%m%d_%H%M')}"
    shutil.copy2(QMT_DB, bak)
    con = sqlite3.connect(QMT_DB)
    for df in to_append:
        df.to_sql("daily", con, if_exists="append", index=False)
    con.commit(); con.close()
    print(f"\n✅ 已追加 {total_inc} 行到 {QMT_DB}")
    print(f"   备份: {bak}")


if __name__ == "__main__":
    main()
