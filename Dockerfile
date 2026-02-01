# Cloudflare Containers：Streamlit 应用
# 必须 linux/amd64，监听 0.0.0.0:8080

FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码（.dockerignore 排除无关文件）
COPY . .

# 暴露端口（Cloudflare 约定 8080）
EXPOSE 8080

# 必须在 0.0.0.0 监听，否则容器外无法访问
# headless=true 适合无头环境
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8080", \
     "--server.headless=true", \
     "--server.enableCORS=true"]
