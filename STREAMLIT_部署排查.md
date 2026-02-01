# Streamlit Community Cloud 部署排查

## 一、先推送代码到 GitHub

在项目目录（含 app_demo.py 的文件夹）打开命令行，执行：

```bash
git add .
git commit -m "Streamlit Cloud 适配"
git push origin main
```

或双击运行 `push_to_github.bat`。

---

## 二、Streamlit Cloud 配置检查

| 配置项 | 正确值 | 错误示例 |
|--------|--------|----------|
| **Main file path** | `app_demo.py` | `creative_eval_demo/app_demo.py`（若 repo 根目录直接是 app_demo.py 则不要加路径） |
| **Branch** | `main` | `master` |
| **Repository** | `MyraWang0406/AIGC-auto-ads` | 确认已授权 |

**重要**：若你的 GitHub 仓库根目录下**直接**有 `app_demo.py`，则 Main file path 填 `app_demo.py`。若 app_demo.py 在 `creative_eval_demo/` 子目录，则填 `creative_eval_demo/app_demo.py`。

---

## 三、常见部署失败原因

1. **Main file path 填错** → 一直 "in the oven"，白屏
2. **缺少 decision_summary.py** → 导入失败，需推送到 GitHub
3. **requirements.txt 依赖冲突** → 当前应为 `streamlit>=1.30.0`、`pydantic>=2.0.0`，无 httpx/python-dotenv
4. **samples/、ui/ 未推送** → vertical_config.json、styles 等缺失会报错
5. **Python 版本** → Streamlit Cloud 默认 3.11，一般无需改

---

## 四、部署后验证

1. 等待 2–5 分钟
2. 访问 `https://你的app.streamlit.app/?page=health` 做健康检查
3. 若 Health 页能打开，说明依赖与入口正确
