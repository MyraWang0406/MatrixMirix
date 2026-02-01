"""
仓库根目录入口：加载 creative_eval_demo 完整应用。
Streamlit Cloud 填 Main file path: streamlit_app.py 即可。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import streamlit as st

# 将 creative_eval_demo 加入路径并切换工作目录
_demo = Path(__file__).resolve().parent / "creative_eval_demo"
sys.path.insert(0, str(_demo))
import os
os.chdir(str(_demo))

st.set_page_config(
    page_title="投放实验决策系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import app_demo
    app_demo.main()
except Exception as e:
    if type(e).__name__ == "StopException":
        raise
    st.error(f"运行失败: {e}")
    with st.expander("错误详情"):
        st.code(traceback.format_exc(), language="text")
