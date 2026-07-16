from __future__ import annotations

import json
import logging
import os

from aiokafka import AIOKafkaProducer
from tools.logger_tools import Kafka_Producer_logger as logger

_producer: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AIOKafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "192.168.1.115:9092"),
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            acks="all",                   # 等待所有副本确认，防止消息丢失
            compression_type="gzip",
        )
        await _producer.start()
        logger.info("Kafka producer started")
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
