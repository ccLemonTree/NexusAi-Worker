# ─────────────────────────────────────────────────────────────────────────────
# NexusAi — Kafka Inference Worker  (multi-stage build)
#
# Stage 1 (builder): python:3.12 full image — ships with gcc, libc6-dev,
#   zlib1g-dev, and everything else needed to compile Cython extensions.
#   lz4 is installed first so aiokafka's Cython extension detects it at
#   compile time (the PyPI wheel is built without lz4 support).
#
# Stage 2 (runtime): python:3.12-slim — only the compiled venv is copied in;
#   no build toolchain in the final image.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1 : builder ────────────────────────────────────────────────────────
FROM python:3.12 AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir lz4 \
 && pip install --no-cache-dir --no-binary aiokafka -r requirements.txt

# ── Stage 2 : runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEXUSAI_HOME=/app \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Runtime-only system libs (no build tools in the final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Compiled virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Application source (see .dockerignore for exclusions)
COPY . .

# Non-root user — aligns with K8s restrictedPodSecurityPolicy
RUN useradd -m -u 1000 nexusai \
    && mkdir -p /app/logs \
    && chown -R nexusai:nexusai /app
USER nexusai

# ─── Runtime environment variables (override via K8s ConfigMap / Secret) ────
# KAFKA_BOOTSTRAP          Kafka broker address          (default: 192.168.1.115:9092)
# KAFKA_INPUT_TOPIC        Consume topic                 (default: model_analyse)
# KAFKA_RESULT_TOPIC       Produce topic                 (default: model_analyse_result)
# KAFKA_CONSUMER_GROUP     Consumer group ID             (default: nexusai-model-worker)
# KAFKA_USERNAME           SASL username — leave unset for no-auth mode
# KAFKA_PASSWORD           SASL password (Secret)
# KAFKA_MAX_CONCURRENT     In-pod concurrency            (default: 16)
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

CMD ["python", "kafka_main.py"]
