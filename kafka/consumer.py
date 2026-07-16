from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Set

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition

from kafka.handlers import process_message
from kafka.producer import send_result, stop_producer
from tools.logger_tools import Kafka_Consumer_logger as logger

KAFKA_BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP",      "192.168.1.115:9092")
KAFKA_INPUT_TOPIC   = os.getenv("KAFKA_INPUT_TOPIC",    "model_analyse")
KAFKA_RESULT_TOPIC  = os.getenv("KAFKA_RESULT_TOPIC",   "model_analyse_result")
KAFKA_GROUP_ID      = os.getenv("KAFKA_CONSUMER_GROUP", "nexusai-model-worker")
MAX_CONCURRENT      = int(os.getenv("KAFKA_MAX_CONCURRENT", "16"))

_shutdown_event = asyncio.Event()


def _register_signals() -> None:
    """注册 SIGTERM / SIGINT，收到信号后置位 shutdown_event 触发优雅退出。"""
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown_event.set)


async def _handle_one(
    msg,
    consumer: AIOKafkaConsumer,
    semaphore: asyncio.Semaphore,
    pending: Set[asyncio.Task],
) -> None:
    """处理单条消息：推理 → 发送结果 → commit offset。"""
    tp = TopicPartition(msg.topic, msg.partition)
    try:
        result = await process_message(msg.value)
        if result is not None:
            await send_result(result)
        # 成功：提交该分区的下一个 offset
        await consumer.commit({tp: msg.offset + 1})
    except Exception as e:
        logger.error(
            f"消息处理失败 topic={msg.topic} partition={msg.partition} offset={msg.offset}: {e}",
            exc_info=True,
        )
        # 失败：仍然 commit，跳过毒丸消息，防止队列卡死
        try:
            await consumer.commit({tp: msg.offset + 1})
        except Exception as ce:
            logger.error(f"commit 失败: {ce}")
    finally:
        semaphore.release()


async def run_consumer() -> None:
    """
    主消费循环。
    - asyncio.Semaphore 控制 Pod 内并发处理数（≤ TRITON_POOL_SIZE）
    - enable_auto_commit=False，处理完成后手动 commit
    - 收到 SIGTERM/SIGINT 后优雅退出：等待 in-flight 任务完成再 stop
    """
    _register_signals()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    pending: Set[asyncio.Task] = set()

    consumer = AIOKafkaConsumer(
        KAFKA_INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="latest",       # 新 consumer 从最新消息开始消费
        max_poll_records=MAX_CONCURRENT,  # 每次 poll 批量数与并发数对齐
        session_timeout_ms=30_000,
        heartbeat_interval_ms=10_000,
        # 处理耗时可能较长（VLM + 小模型），适当放大 poll 间隔超时
        max_poll_interval_ms=300_000,
    )

    await consumer.start()
    logger.info(
        f"Kafka consumer started | topic={KAFKA_INPUT_TOPIC} | "
        f"group={KAFKA_GROUP_ID} | max_concurrent={MAX_CONCURRENT}"
    )

    try:
        async for msg in consumer:
            if _shutdown_event.is_set():
                logger.info("Shutdown signal received, stopping consumer loop")
                break

            # 占用一个并发槽，超出上限时此处会等待直到有槽位释放
            await semaphore.acquire()

            task = asyncio.create_task(
                _handle_one(msg, consumer, semaphore, pending),
                name=f"msg-{msg.offset}",
            )
            pending.add(task)
            task.add_done_callback(pending.discard)

    finally:
        # 等待所有 in-flight 任务完成，保证优雅退出
        if pending:
            logger.info(f"Waiting for {len(pending)} in-flight tasks to finish...")
            await asyncio.gather(*pending, return_exceptions=True)

        await consumer.stop()
        await stop_producer()
        logger.info("Kafka consumer cleanly stopped")
