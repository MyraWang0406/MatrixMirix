# 部署指南

## 一、推送到 GitHub

**方式一：双击运行** `push_to_github.bat`

**方式二：命令行**

```bash
cd creative_eval_demo
git add .
git commit -m "更新：AIGC 创意评测决策看板"
git push origin main
```

> 推送时需输入 GitHub 用户名和 Personal Access Token（密码已不支持）。

---

## 二、云部署方案

### 方案 A：Streamlit Community Cloud（最简单）

Streamlit 官方免费托管，无需 Docker，**推荐首选**。

| 步骤 | 操作 |
|------|------|
| 1 | 打开 https://share.streamlit.io |
| 2 | 用 GitHub 登录 → **New app** |
| 3 | 仓库：`MyraWang0406/AIGC-auto-ads`，分支：`main` |
| 4 | **Main file path**：`app_demo.py` |
| 5 | 点击 **Deploy** |

部署完成后得到 `https://<app-name>.streamlit.app`。

**健康检查**：访问 `https://<app-name>.streamlit.app/?page=health` 或点击导航 **Health**，可快速排查依赖/环境变量问题。

### 方案 B：Cloudflare Containers

需要本地安装 Docker 和 Node.js，适合希望用 Cloudflare 托管的场景。

```bash
cd creative_eval_demo
npm install
npx wrangler secret put OPENROUTER_API_KEY --config wrangler.containers.toml
npx wrangler secret put OPENROUTER_MODEL --config wrangler.containers.toml
npx wrangler deploy --config wrangler.containers.toml
```

首次部署后需等待 3–5 分钟容器就绪，详见 [DEPLOY_CLOUDFLARE_CONTAINERS.md](DEPLOY_CLOUDFLARE_CONTAINERS.md)。

### 方案 C：其他 PaaS

- **Railway**：railway.app，支持从 GitHub 部署
- **Render**：render.com，支持 Web Service
- **Fly.io**：fly.io，支持 Dockerfile 部署

---

## 三、Streamlit Cloud 常见“in the oven/启动失败”排查

| 原因 | 现象 | 排查方法 |
|------|------|----------|
| **Main file path 错误** | 一直转圈、白屏 | 确认填 `app_demo.py`（根目录），不要加 `creative_eval_demo/` 前缀 |
| **requirements.txt 依赖冲突** | 部署失败、Build error | 精简依赖，移除 httpx/python-dotenv（app_demo 不需要）；保证 streamlit>=1.30、pydantic>=2.0 |
| **config.toml 写死端口** | 无法启动 | 删除 `port`、`serverAddress` 等设置，由平台自动分配 |
| **导入失败** | 页面报 ImportError | 访问 `/?page=health` 查看具体模块；检查 samples/、ui/ 等目录是否随仓库推送 |
| **内存不足** | 启动超时、频繁重启 | 减少首屏计算量；Streamlit Cloud 免费版约 1GB 内存 |

**建议**：首次部署后等待 2–5 分钟，再访问 `/?page=health` 验证。

---

## 五、关于 Cloudflare Pages

**Cloudflare Pages 仅支持静态站点**，无法直接运行 Streamlit（Python 服务端应用）。若坚持用 Cloudflare，请使用 **Cloudflare Containers**（方案 B）。
