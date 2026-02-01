@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在推送到 GitHub...
git push origin main
if errorlevel 1 (
    echo.
    echo 推送失败，可能原因：网络不稳定、代理、防火墙
    echo 可尝试：1) 关闭 VPN 或换网络  2) 改用 SSH: git remote set-url origin git@github.com:MyraWang0406/AIGC-auto-ads.git
) else (
    echo 推送成功。
)
pause
