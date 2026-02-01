# AIGC 创意评测决策看板 (AIGC-auto-ads)

基于 Streamlit 的投放素材评测与决策看板 Demo，支持结构卡片、OFAAT 变体生成、门禁评测与元素级贡献分析。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app_demo.py
```

本地指定端口：`streamlit run app_demo.py --server.port 3100`

## 云部署：Streamlit Community Cloud（推荐）

1. 将仓库推送到 GitHub
2. 打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 登录
3. **New app** → 仓库 `MyraWang0406/AIGC-auto-ads`，分支 `main`
4. **Main file path**：`app_demo.py`（必填，根目录下）
5. 点击 **Deploy**

> **说明**：`app_demo.py` 使用模拟数据，无需配置 Secrets。若部署 `app.py`（LLM 生成），需在 Settings → Secrets 中配置 `OPENROUTER_API_KEY`、`OPENROUTER_MODEL`。

### 健康检查

部署后若页面加载慢，可访问：

- `https://你的app.streamlit.app/?page=health` 或 `?health=1`
- 或点击导航栏 **Health** 按钮

用于排查依赖、环境变量等问题。

## 项目结构

```
├── app_demo.py          # 主入口（Streamlit Community Cloud 部署用）
├── app.py               # LLM 生成入口（需 OPENROUTER_API_KEY）
├── requirements.txt
├── samples/             # 示例 JSON
├── .streamlit/config.toml
└── ...
```

## 联系

myrawzm0406@163.com
