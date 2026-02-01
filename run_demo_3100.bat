@echo off
cd /d "%~dp0"
streamlit run app_demo.py --server.port 3100
