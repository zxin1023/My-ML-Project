# 最小 Dockerfile 示例（用于可复现环境）
FROM python:3.10-slim

WORKDIR /app
# 安装系统依赖（按需修改）
RUN apt-get update && apt-get install -y git build-essential

# 复制项目文件（或用镜像构建时从 Git clone）
COPY . /app

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 默认入口（示例）
ENTRYPOINT ["python", "-m", "src.trainers"]