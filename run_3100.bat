@echo off
cd /d "%~dp0"
echo ==========================================
echo   决策看板 | http://localhost:3100
echo ==========================================
echo.
echo 【必读】请用 Chrome 或 Edge 打开上述地址
echo   Cursor 内置浏览器会 Connection Failed，请改用系统浏览器！
echo.
echo 5 秒后自动用默认浏览器打开...
start cmd /c "timeout /t 5 /nobreak > nul && start http://localhost:3100"
python -m streamlit run app_demo.py --server.port 3100
pause
