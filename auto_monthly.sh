#!/bin/bash
# 月度自动运行脚本
# 由 launchd 每月 28 日触发，自动判断是否为最后交易日

cd ~/ai-capital-ashare
source venv/bin/activate

DATE=$(date +%Y-%m-%d)
echo "[$(date)] Auto-run triggered"

# 获取今天沪深300数据，判断是否交易日
python3 -c "
import akshare as ak
import pandas as pd
df = ak.stock_zh_index_daily(symbol='sh000300')
last_date = df['date'].iloc[-1]
print(f'Last trading day: {last_date}')
" > /tmp/last_trading_day.txt 2>&1

# 如果今天是周末或不是最后交易日，跳过
python3 -c "
from datetime import datetime
today = datetime.now()
if today.weekday() >= 5:
    print('Weekend, skipping')
    exit(1)

# Check if within 2 days of month end
import calendar
last_day = calendar.monthrange(today.year, today.month)[1]
if today.day < last_day - 1:
    print(f'Not month end (today={today.day}, month_end={last_day}), skipping')
    exit(1)

print('Running rebalance...')
" || { echo "[$(date)] Not rebalance date, skipped"; exit 0; }

# 执行
python refresh_data.py
python run_monthly.py

echo "[$(date)] Done"
