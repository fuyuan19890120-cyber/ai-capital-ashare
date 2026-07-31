#!/bin/bash
# ETF-R3 数据更新：每月末收盘后运行
cd ~/ai-capital-ashare
git pull
git add data/qmt_qfq.db
git commit -m "data: QMT数据库 $(date +%Y-%m-%d)" || echo "无变化"
git push origin main
echo "done"
