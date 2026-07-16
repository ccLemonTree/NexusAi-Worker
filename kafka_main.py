"""
kafka_main.py — NexusAi Kafka Worker 独立启动入口

启动方式:
    python kafka_main.py

K8s Deployment 示例:
    command: ["python", "kafka_main.py"]

环境变量（通过 ConfigMap / Secret 注入）:
    KAFKA_BOOTSTRAP         Kafka 地址，默认 192.168.1.115:9092
    KAFKA_INPUT_TOPIC       消费 topic，默认 model_analyse
    KAFKA_RESULT_TOPIC      结果 topic，默认 model_analyse_result
    KAFKA_CONSUMER_GROUP    消费者组，默认 nexusai-model-worker
    KAFKA_MAX_CONCURRENT    Pod 内并发处理数，默认 16（需 ≤ TRITON_POOL_SIZE）
    EOS_TIMEOUT             EOS 图片下载超时秒数，默认 15
    TRITON_SERVER           Triton gRPC 地址
    TRITON_SERVER_HEADLESS  Triton headless service 地址（K8s 优先使用）
    TRITON_POOL_SIZE        Triton 连接池大小，默认 70
    ANALYSE_MAX_WORKERS     ThreadPoolExecutor 大小
    INFER_BACKEND           VLM 后端，可选 openai / mindie / fire
    NEXUSAI_HOME            项目根路径
    MILVUS_CLIENT           Milvus 连接串
    MILVUS_DB_NAME          Milvus 数据库名
"""

import asyncio
import logging
import os
import pathlib

from dotenv import load_dotenv

# 优先加载 .env（本地开发用），K8s 中由 ConfigMap/Secret 注入真实值
_env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(_env_path, override=False)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)


def main() -> None:
    from kafka.consumer import run_consumer
    logger.info("NexusAi Kafka Worker starting...")
    asyncio.run(run_consumer())


if __name__ == "__main__":
    main()
