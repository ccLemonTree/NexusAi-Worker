from __future__ import annotations

import json
import os

from aiokafka import AIOKafkaProducer
from tools.logger_tools import Kafka_Producer_logger as logger

_producer: AIOKafkaProducer | None = None


def _sasl_kwargs() -> dict:
    """
    若配置了 KAFKA_USERNAME 则返回 SASL/SCRAM 参数，否则返回空 dict。
    KAFKA_SASL_MECHANISM 默认 SCRAM-SHA-256，可改为 SCRAM-SHA-512。
    """
    username = os.getenv("KAFKA_USERNAME", "")
    if not username:
        return {}
    return {
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism":    os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256"),
        "sasl_plain_username": username,
        "sasl_plain_password": os.getenv("KAFKA_PASSWORD", ""),
    }


async def get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "192.168.1.115:9092"),
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            acks="all",
            compression_type="gzip",
            **_sasl_kwargs(),
        )
        await _producer.start()
        logger.info(f"Kafka producer started  sasl={'yes' if _sasl_kwargs() else 'no'}")
    return _producer


async def stop_producer() -> None:
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer stopped")


async def send_result(result: dict) -> None:
    topic = os.getenv("KAFKA_RESULT_TOPIC", "model_analyse_result")
    producer = await get_producer()
    await producer.send_and_wait(topic, result)
    logger.debug(f"Result sent  id={result.get('id')}  topic={topic}")
