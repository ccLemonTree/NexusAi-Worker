from __future__ import annotations

import asyncio
import json
import os

from aiokafka import AIOKafkaProducer
from tools.logger_tools import Kafka_Producer_logger as logger

_producer: AIOKafkaProducer | None = None
_producer_lock = asyncio.Lock()  # 保护 producer 初始化的异步锁


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
    async with _producer_lock:
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
        # 防御性检查：producer 对象存在但未正确初始化（_producer_magic 丢失），强制重建
        elif not hasattr(_producer, "_producer_magic"):
            logger.warning("Producer 状态异常（_producer_magic 丢失），强制重建")
            try:
                await _producer.stop()
            except Exception:
                pass
            _producer = None
            return await get_producer()  # 递归重建
        return _producer


async def stop_producer() -> None:
    global _producer
    async with _producer_lock:
        if _producer is not None:
            await _producer.stop()
            _producer = None
            logger.info("Kafka producer stopped")


async def send_result(result: dict) -> None:
    topic = os.getenv("KAFKA_RESULT_TOPIC", "model_analyse_result")
    send_timeout = float(os.getenv("KAFKA_SEND_TIMEOUT", "10"))
    msg_id = result.get("id")
    try:
        producer = await get_producer()
        key = str(msg_id).encode("utf-8") if msg_id is not None else None
        # 加超时：结果 topic 不存在或元数据异常时 send_and_wait 会长时间阻塞，
        # 导致 _handle_one 卡住不释放并发槽，进而 poll 停摆、消费者被踢出组。
        await asyncio.wait_for(
            producer.send_and_wait(topic, result, key=key),
            timeout=send_timeout,
        )
        logger.debug(f"Result sent  id={msg_id}  topic={topic}")
    except asyncio.TimeoutError:
        logger.error(f"结果发送超时（>{send_timeout}s） id={msg_id}  topic={topic}，跳过")
    except AttributeError as e:
        # producer 状态异常（_producer_magic 等内部属性丢失）
        logger.error(f"Producer 状态异常 id={msg_id}: {e}，跳过本次发送")
    except Exception as e:
        # 其他异常（网络、broker 不可达等）
        logger.error(f"结果发送失败 id={msg_id}  topic={topic}: {e}")
