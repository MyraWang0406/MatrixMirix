# Cloudflare 部署说明

本目录包含 **Cloudflare Pages（前端）** 与 **Cloudflare Workers Python（FastAPI 后端）** 的部署结构。

## 架构

- **前端**：`web/` 目录 → Cloudflare Pages（静态站点）
- **后端**：`api/` 目录 → Cloudflare Workers Python（FastAPI）
- **业务逻辑**：复用 `prompts`、`openrouter_client`、`scoring`、`schemas`、`exporters` 等模块

## 一、本地联调（最小可运行步骤）

### 1. 安装依赖

```bash
cd creative_eval_demo
pip install -r requirements.txt fastapi uvicorn
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写：

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### 3. 启动后端

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. 启动静态前端

另开终端：

```bash
cd creative_eval_demo/web
python -m http.server 8080
```

浏览器访问：http://localhost:8080  
前端会自动请求 http://localhost:8000（API 地址留空时默认）。

### 5. 验证

- 打开 http://localhost:8080
- 点击「游戏 (game_card)」加载示例
- 点击「生成并评审」
- 查看表格与下载 CSV/Markdown

---

## 二、API 说明

### GET /healthz

健康检查。

**响应示例：**

```json
{"status": "ok"}
```

### POST /generate_and_review

输入结构卡片 + 变体数量，返回评审表格数据。

**请求体：**

```json
{
  "card": {
    "vertical": "game",
    "product_name": "《王者荣耀》新赛季",
    "target_audience": "18-35岁手游玩家",
    "key_selling_points": ["新英雄上线", "赛季皮肤免费领"],
    "tone": "热血、年轻",
    "constraints": ["不得出现未成年人游戏画面"],
    "no_exaggeration": true
  },
  "n": 5
}
```

**成功响应示例：**

```json
{
  "ok": true,
  "table": [
    {
      "index": 1,
      "variant_id": "v001",
      "headline": "...",
      "decision": "PASS",
      "fuse_level": "GREEN",
      "white_traffic_risk_final": 25,
      "clarity": 80,
      "hook_strength": 75,
      "compliance_safety": 85,
      "expected_test_value": 70,
      "summary": "..."
    }
  ],
  "csv": "序号,headline,...\n1,...",
  "markdown": "# 创意变体评审报告\n..."
}
```

**失败响应示例：**

```json
{
  "ok": false,
  "error": "请设置 OPENROUTER_API_KEY",
  "table": [],
  "csv": "",
  "markdown": ""
}
```

---

## 三、Cloudflare 部署

### 1. Workers（后端 API）

请参考 [Cloudflare Workers Python 官方文档](https://developers.cloudflare.com/workers/languages/python/)。

简要步骤（以 `pywrangler` 为例，具体以官方为准）：

```bash
# 安装 pywrangler / uv
# 参考：https://developers.cloudflare.com/workers/languages/python/packages/fastapi/

uv run pywrangler dev    # 本地调试
uv run pywrangler deploy # 部署
```

设置 Secrets：

```bash
wrangler secret put OPENROUTER_API_KEY
wrangler secret put OPENROUTER_MODEL
```

部署完成后得到 Worker URL，例如：`https://creative-eval-api.xxx.workers.dev`。

### 2. Pages（前端静态站点）

1. 打开 [Cloudflare Dashboard](https://dash.cloudflare.com/) → Pages
2. 创建项目 → 连接 Git 或直接上传
3. 构建配置：
   - 构建命令：留空或 `echo "static"`
   - 输出目录：`creative_eval_demo/web`（或把 `web/` 内容放到根目录）

4. 部署后，在页面「API 地址」输入框中填写 Worker URL（如 `https://creative-eval-api.xxx.workers.dev`）。

### 3. 跨域（CORS）

后端 FastAPI 已配置 `allow_origins=["*"]`，Pages 域名可正常访问 Worker API。若遇跨域问题，请参考 Cloudflare 文档调整。

---

## 四、注意事项

- **OPENROUTER_API_KEY**：必须设置，否则接口返回错误。
- **Python Workers**：Cloudflare Workers Python 处于快速发展期，`wrangler` 配置与 `pywrangler` 用法请以 [Cloudflare 官方文档](https://developers.cloudflare.com/workers/) 为准。
- **app.py**：原有 Streamlit 应用保留，可作为本地 demo 使用；Cloudflare 部署使用 `api/` + `web/`。
