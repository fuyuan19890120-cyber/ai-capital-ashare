#!/bin/bash
# 每月末 QMT 同步完成后运行此脚本
DB=~/ai-capital-ashare/data/qmt_qfq.db
OUT=~/ai-capital-ashare/data/etf_compact.db
ETFS="000300.SH 510300.SH 510500.SH 159915.SZ 588000.SH 511010.SH 518880.SH 511880.SH 512880.SH 512800.SH 512070.SH 512000.SH 512010.SH 512170.SH 159992.SZ 159928.SZ 512690.SH 512480.SH 159995.SZ 512760.SH 512660.SH 512710.SH 515030.SH 515790.SH 516160.SH 512400.SH 515210.SH 159870.SZ 512200.SH 516750.SH 512980.SH 159869.SZ 159939.SZ 515000.SH 512720.SH 515050.SH 515880.SH 159967.SZ 562500.SH 159819.SZ 159611.SZ 512580.SH 516670.SH 159865.SZ"

cd ~/ai-capital-ashare || exit 1
rm -f "$OUT"
sqlite3 "$OUT" "CREATE TABLE daily (code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
for code in $ETFS; do
  sqlite3 "$DB" -cmd "ATTACH '$OUT' AS out" "INSERT INTO out.daily SELECT code,date,open,high,low,close,volume FROM main.daily WHERE code='$code'"
done
echo "ETF compact DB: $(ls -lh "$OUT" | awk '{print $5}'), $(sqlite3 "$OUT" 'SELECT COUNT(*) FROM daily') rows"
git add data/etf_compact.db
git commit -m "data: ETF数据更新 $(date +%Y-%m-%d)" || echo "无变化"
git push origin main
echo "done"
