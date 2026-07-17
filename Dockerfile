# ─────────────────────────────────────────────────────────────────────────────
# NexusAi — Kafka Inference Worker
# Base: python:3.12-slim
#
# LZ4 note: the aiokafka PyPI wheel is compiled with HAS_LZ4=False (the
# maintainer's build env has no lz4).  Even building from source only
# recompiles the pre-generated .c files — the constant is already baked in.
# Fix: remove the _crecords Cython extensions after install; aiokafka then
# falls back to its pure-Python implementation which does `import lz4.frame`
# at runtime, picking up the lz4 package we install here.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEXUSAI_HOME=/app

WORKDIR /app

# Runtime system libraries for opencv-python-headless and numpy/OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && find /usr/local/lib -path "*/aiokafka/record/_crecords/*.so" -delete

# Copy application source (see .dockerignore for exclusions)
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
