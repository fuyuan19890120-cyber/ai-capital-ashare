"""
策略一看板入口 — 重定向到 V4 对比页面
启动: cd ~/ai-capital-ashare && streamlit run dashboard/app.py
"""
import streamlit as st
import os

st.set_page_config(page_title="策略一 V4 对比看板", page_icon="📊", layout="wide")

html_path = os.path.join(os.path.dirname(__file__), "v4_compare.html")
with open(html_path) as f:
    html = f.read()

st.components.v1.html(html, height=3000, scrolling=True)
