# streamlit_app.py
from __future__ import annotations

import traceback
import streamlit as st

st.set_page_config(
    page_title="投放实验决策系统 (Decision Support System)",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import app_demo  # 现在就在 repo 根目录
    app_demo.main()
except Exception as e:
    # Streamlit 自己的 StopException 不要吞
    if type(e).__name__ == "StopException":
        raise

    st.error(f"运行失败：{e}")
    with st.expander("错误详情"):
        st.code(traceback.format_exc(), language="text")
