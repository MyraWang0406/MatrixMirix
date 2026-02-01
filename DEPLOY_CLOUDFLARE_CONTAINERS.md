# Cloudflare Containers + Streamlit 部署说明

本项目通过 **Cloudflare Containers** 运行 Streamlit，**Cloudflare Worker** 作为入口代理所有 HTTP 请求到容器。

## 架构

```
用户请求 → Worker（入口）→ /healthz 直接返回
                      → 其他路径 → Streamlit 容器（0.0.0.0:8080）
```

## A) Streamlit 配置修改说明

| 修改项 | 原因 |
|--------|------|
| 移除 `port = 3100` | 端口由 CMD 指定 `--server.port=8080`，避免与 Cloudflare 约定冲突 |
| 移除 `browser.serverAddress = "localhost"` | 云端部署后前端会连错地址，导致白屏 |
| `headless = true` | 容器无浏览器，必须 headless |
| 不写死 address | 容器内必须监听 `0.0.0.0`，由 CMD 指定 |

## B) 关键文件

- `wrangler.containers.toml`：Containers + Durable Objects 配置
- `src/worker.ts`：Worker 入口 + `StreamlitContainer` 类
- `Dockerfile`：基于 `python:3.11-slim`，Streamlit 监听 8080
- `.dockerignore`：减小镜像构建上下文

## C) Secrets / 环境变量

### 设置方式

```bash
# 在项目目录执行
cd creative_eval_demo

# 设置 OpenRouter 密钥（部署前必填）
npx wrangler secret put OPENROUTER_API_KEY --config wrangler.containers.toml
npx wrangler secret put OPENROUTER_MODEL --config wrangler.containers.toml
```

Worker Secrets 会自动注入到 `env`，`StreamlitContainer` 通过 `envVars` 传给容器。本地开发仍可用 `.env` + `python-dotenv`，线上依赖上述环境变量。

### 非敏感变量

可在 `wrangler.containers.toml` 中增加：

```toml
[vars]
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

敏感信息切勿写入 `[vars]`，必须用 `wrangler secret put`。

---

## 本地验证

### 1. Docker 本地跑 Streamlit

```bash
cd creative_eval_demo
docker build --platform linux/amd64 -t creative-eval-streamlit .
docker run -p 8080:8080 -e OPENROUTER_API_KEY=sk-or-xxx creative-eval-streamlit
```

浏览器访问 http://localhost:8080，确认 Streamlit 正常。

### 2. 验证 Worker（可选）

```bash
npm install
npx wrangler dev --config wrangler.containers.toml
```

首次会拉取并启动容器，耗时较长。访问根路径应看到 Streamlit 页面，`/healthz` 返回 `{"status":"ok"}`。

---

## Cloudflare 部署

### 1. 前置条件

- Docker 已安装并在运行
- 已执行 `wrangler secret put OPENROUTER_API_KEY`

### 2. 部署命令

```bash
cd creative_eval_demo
npm install
npx wrangler deploy --config wrangler.containers.toml
```

### 3. 首次部署等待时间

容器首次 provision 需要数分钟，部署完成后访问可能出现 502/超时。可执行：

```bash
npx wrangler containers list --config wrangler.containers.toml
```

查看容器状态，待状态为 `ready` 后再访问 Worker URL。

### 4. WebSocket

Streamlit 依赖 WebSocket。Cloudflare Containers 的 `container.fetch()` 支持 WebSocket 转发，一般无需额外配置。

---

## 常见问题排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 502 / 连接超时 | 容器尚未就绪 | 等待数分钟后重试，`wrangler containers list` 查状态 |
| 白屏 / 无法加载 | `serverAddress=localhost` | 确认 `.streamlit/config.toml` 已移除该配置 |
| 容器无法访问 | 未监听 0.0.0.0 | 确认 CMD 含 `--server.address=0.0.0.0` |
| OpenRouter 调用失败 | 未配置 API Key | 执行 `wrangler secret put OPENROUTER_API_KEY` |
| 镜像体积过大 | 上下文含 .venv 等 | 检查 `.dockerignore` 是否生效 |
| 依赖安装失败 | 架构不匹配 | 使用 `--platform=linux/amd64` 构建 |

---

## 验证 Checklist

1. [ ] `.streamlit/config.toml` 无 `port`、无 `serverAddress`
2. [ ] `headless = true`
3. [ ] `docker build` 成功，`docker run` 可在 8080 访问
4. [ ] 已执行 `wrangler secret put OPENROUTER_API_KEY`
5. [ ] `npx wrangler deploy --config wrangler.containers.toml` 成功
6. [ ] 部署后等待约 3–5 分钟
7. [ ] `/healthz` 返回 `{"status":"ok"}`
8. [ ] 根路径可打开 Streamlit 页面
9. [ ] Streamlit 交互（生成/评审）可正常使用
10. [ ] `wrangler containers list` 中容器状态为 ready
