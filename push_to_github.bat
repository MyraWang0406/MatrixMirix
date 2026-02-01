@echo off
chcp 65001 >nul
echo ========== 推送到 GitHub ==========
echo.
echo 仓库: https://github.com/MyraWang0406/AIGC-auto-ads.git
echo.

cd /d "%~dp0"

if not exist ".git" (
    echo 初始化 Git...
    git init
    git remote add origin https://github.com/MyraWang0406/AIGC-auto-ads.git
)

echo 添加文件...
git add .
echo 提交...
git commit -m "最小入口 streamlit_app.py 必过部署" 2>nul || echo (无新更改可提交)
echo.
echo 推送中... (需输入 GitHub 用户名与 Token)
git branch -M main
git push -u origin main

echo.
echo 完成。若推送失败，请检查:
echo 1. 已安装 Git
echo 2. GitHub Token 有 repo 权限
echo 3. 仓库 https://github.com/MyraWang0406/AIGC-auto-ads 已创建
pause
