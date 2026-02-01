"""
在 3100 端口启动决策看板

【重要】必须用 Chrome / Edge 打开 http://localhost:3100
Cursor 内置浏览器会 Connection Failed，请改用系统默认浏览器！

用法：python run_demo_3100.py
"""
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

DIR = Path(__file__).resolve().parent
APP = DIR / "app_demo.py"
URL = "http://localhost:3100"


def _open_browser():
    time.sleep(5)
    webbrowser.open(URL)


def main():
    if not APP.exists():
        print(f"错误：找不到 {APP}")
        sys.exit(1)
    try:
        import streamlit
    except ImportError:
        print("错误：未安装 streamlit，请先执行：pip install streamlit")
        sys.exit(1)
    t = threading.Thread(target=_open_browser, daemon=True)
    t.start()
    print("=" * 55)
    print("  决策看板 ->", URL)
    print("=" * 55)
    print("  【必读】请用 Chrome 或 Edge 打开，不要用 Cursor 内置浏览器")
    print("  5 秒后将自动用系统默认浏览器打开")
    print("  按 Ctrl+C 停止服务")
    print("=" * 55)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", "3100"],
        cwd=str(DIR),
    )


if __name__ == "__main__":
    main()
