#!/usr/bin/env python3
"""
P0: 抓取个股日频辅助数据 (Baostock)

字段: turn(换手率%) / peTTM / pbMRQ / isST / amount(成交额元) / tradestatus
用途: PMO因子(turn)、Amihud(amount)、EP(peTTM)、ST过滤(isST)
缓存: data/stock_aux/{code}.csv, 已存在且行数足够则跳过(增量安全)

用法: venv/bin/python fetch_stock_aux.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import baostock as bs

DATA_DIR = os.path.expanduser("~/ai-capital-ashare/data")
AUX_DIR = os.path.join(DATA_DIR, "stock_aux")
os.makedirs(AUX_DIR, exist_ok=True)

FIELDS = "date,close,turn,peTTM,pbMRQ,isST,amount,tradestatus"


def bs_code(code):
    return ("sh." if code.startswith(("6", "9")) else "sz.") + code


def _is_complete(path):
    """按末行日期判断文件是否完整(复审 C9: 体积阈值会把半截下载误判为完成)"""
    try:
        with open(path, "rb") as f:
            f.seek(-200, os.SEEK_END)
            last = f.read().decode(errors="ignore").strip().splitlines()[-1]
        return last[:10] >= "2026-07-01"
    except Exception:  # noqa: BLE001
        return False


def main():
    codes = sorted(f[:-4] for f in os.listdir(os.path.join(DATA_DIR, "stocks")) if f.endswith(".csv"))
    print(f"[aux] 目标 {len(codes)} 只", flush=True)

    lg = bs.login()
    assert lg.error_code == "0", lg.error_msg

    ok = skip = fail = 0
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        out = os.path.join(AUX_DIR, f"{code}.csv")
        if os.path.exists(out) and _is_complete(out):
            skip += 1
            continue
        try:
            rs = bs.query_history_k_data_plus(
                bs_code(code), FIELDS,
                start_date="2015-01-01", end_date="2026-07-17",
                frequency="d", adjustflag="3")  # 不复权: 仅取估值/换手/成交额字段
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=FIELDS.split(","))
                df.to_csv(out, index=False)
                ok += 1
            else:
                fail += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  ! {code} {type(e).__name__}: {str(e)[:60]}", flush=True)
        if i % 50 == 0 or i == len(codes):
            el = time.time() - t0
            print(f"  [{i}/{len(codes)}] ok={ok} skip={skip} fail={fail} "
                  f"elapsed={el/60:.1f}min eta={el/max(i-skip,1)*(len(codes)-i)/60:.1f}min", flush=True)
        time.sleep(0.05)

    bs.logout()
    print(f"[done] ok={ok} skip={skip} fail={fail}", flush=True)
    sys.exit(1 if fail > 50 else 0)


if __name__ == "__main__":
    main()
