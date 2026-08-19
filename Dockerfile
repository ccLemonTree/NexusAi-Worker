# ─────────────────────────────────────────────────────────────────────────────
# NexusAi — Stateless Inference Service
# Base: registry.cn-hangzhou.aliyuncs.com/lemontree_images/python:3.12-slim
#
# ─────────────────────────────────────────────────────────────────────────────
FROM registry.cn-hangzhou.aliyuncs.com/lemontree_images/python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEXUSAI_HOME=/app \
    TZ=Asia/Shanghai

WORKDIR /app

# Runtime system libraries for opencv-python-headless and numpy/OpenMP + timezone setup
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (see .dockerignore for exclusions)
COPY . .

# Create logs directory (run as root, no non-root user needed for now)
RUN mkdir -p /app/logs

# Runtime settings are injected by the deployment platform.
# WORKER_MAX_CONCURRENT    Worker inference concurrency  (default: 16)
# WORKER_AUTH_TOKEN        Required shared HTTP secret
# TRITON_SERVER            Triton gRPC address
# TRITON_SERVER_VLM        Triton VLM gRPC address       (falls back to TRITON_SERVER)
# TRITON_POOL_SIZE         gRPC connection pool size      (default: 70)
# INFER_BACKEND            VLM backend: fire|openai|mindie
# STRUCT_EMBEDDING_MODEL   VLM API base URL
# STRUCT_EMBEDDING_MODEL_NAME  Model name
# MILVUS_CLIENT            Milvus connection string (use http:// not grpc://)
# MILVUS_DB_NAME           Milvus database name
# VLM_TIMEOUT              VLM inference timeout seconds  (default: 60)
# MODEL_TIMEOUT            Small-model timeout seconds    (default: 30)
# VECTOR_TIMEOUT           Vector ingest timeout seconds  (default: 30)
# EOS_TIMEOUT              EOS pre-signed URL timeout     (default: 15)

EXPOSE 8080

CMD ["python", "worker_main.py"]
