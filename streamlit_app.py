"""
Streamlit Community Cloud 唯一入口（壳）。
负责 set_page_config、异常捕获、调用真实业务 app_demo.main()
"""
from __future__ import annotations

import traceback

import streamlit as st

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
