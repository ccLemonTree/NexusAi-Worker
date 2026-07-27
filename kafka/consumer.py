from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Set

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import (
    CommitFailedError,
    IllegalGenerationError,
    IllegalStateError,
    UnknownMemberIdError,
)
from aiokafka.structs import TopicPartition

from kafka.handlers import process_message
from kafka.producer import send_result, stop_producer
from kafka.stats import get_stats_collector, start_stats_task
from tools.logger_tools import Kafka_Consumer_logger as logger

KAFKA_BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP",      "192.168.1.115:9092")
KAFKA_INPUT_TOPIC   = os.getenv("KAFKA_INPUT_TOPIC",    "model_analyse")
KAFKA_RESULT_TOPIC  = os.getenv("KAFKA_RESULT_TOPIC",   "model_analyse_result")
KAFKA_GROUP_ID      = os.getenv("KAFKA_CONSUMER_GROUP", "nexusai-model-worker")
MAX_CONCURRENT      = int(os.getenv("KAFKA_MAX_CONCURRENT", "16"))


def _sasl_kwargs() -> dict:
    """
    若配置了 KAFKA_USERNAME 则返回 SASL/SCRAM 参数，否则返回空 dict（无认证模式）。
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

_shutdown_event = asyncio.Event()


def _register_signals() -> None:
    """
    注册优雅退出信号。
    - Linux/K8s: loop.add_signal_handler（支持 SIGTERM + SIGINT）
    - Windows:   signal.signal 降级（仅 SIGINT / Ctrl+C，SIGTERM 在 K8s 中才生效）
    """
    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown_event.set)
    except NotImplementedError:
        # Windows 不支持 add_signal_handler，降级为 signal.signal
        signal.signal(signal.SIGINT, lambda _s, _f: _shutdown_event.set())


async def _safe_commit(consumer: AIOKafkaConsumer, tp: TopicPartition, offset: int) -> None:
    """
    提交 offset，容忍 rebalance 导致的各种失效场景：
    - 分区已不在本消费者的 assignment 中（IllegalStateError）
    - 提交时组已 rebalance / generation 过期 / member 失效
      （CommitFailedError / IllegalGenerationError / UnknownMemberIdError）
    这些都是多消费者 rebalance 的正常现象：该 offset 会由新 owner 重新消费。
    统一降级为 warning，不刷错误堆栈。
    """
    # 提交前先校验分区归属，能挡掉大部分无谓提交
    if tp not in consumer.assignment():
        logger.warning(f"跳过 commit：分区 {tp} 已不属于本消费者（rebalance），offset={offset}")
        return
    try:
        await consumer.commit({tp: offset})
    except (CommitFailedError, IllegalStateError,
            IllegalGenerationError, UnknownMemberIdError) as e:
        logger.warning(
            f"commit 被 rebalance 打断，跳过 partition={tp.partition} offset={offset}: "
            f"{type(e).__name__}（该 offset 将由新 owner 重新消费）"
        )


async def _handle_one(
    msg,
    consumer: AIOKafkaConsumer,
    semaphore: asyncio.Semaphore,
    pending: Set[asyncio.Task],
) -> None:
    """处理单条消息：推理 → 发送结果 → commit offset。"""
    tp = TopicPartition(msg.topic, msg.partition)
    stats = get_stats_collector()

    try:
        # 记录消费
        stats.record_consumed()

        result = await process_message(msg.value)
        # process_message 现在总是返回结果（包括错误情况）
        await send_result(result)

        # 成功：提交该分区的下一个 offset
        await _safe_commit(consumer, tp, msg.offset + 1)
    except Exception as e:
        logger.error(
            f"消息处理失败 topic={msg.topic} partition={msg.partition} offset={msg.offset}: {e}",
            exc_info=True,
        )
        # 失败：仍然 commit，跳过毒丸消息，防止队列卡死
        await _safe_commit(consumer, tp, msg.offset + 1)
    finally:
        # 无论成功失败都统计生产（记录的是"尝试处理"的消息数）
        stats.record_produced()
        semaphore.release()


async def run_consumer() -> None:
    """
    主消费循环。
    - asyncio.Semaphore 控制 Pod 内并发处理数（≤ TRITON_POOL_SIZE）
    - enable_auto_commit=False，处理完成后手动 commit
    - 收到 SIGTERM/SIGINT 后优雅退出：等待 in-flight 任务完成再 stop
    """
    _register_signals()

    # 启动统计后台任务
    start_stats_task()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    pending: Set[asyncio.Task] = set()

    sasl = _sasl_kwargs()
    logger.info(f"Kafka SASL: {'enabled  mechanism=' + sasl['sasl_mechanism'] if sasl else 'disabled'}")

    consumer = AIOKafkaConsumer(
        KAFKA_INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        max_poll_records=MAX_CONCURRENT,
        # session_timeout: 心跳超时时长，超过此时间未收到心跳则踢出组触发 rebalance
        # 应对偶发的网络延迟和处理阻塞，设置为 60 秒
        session_timeout_ms=int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "60000")),
        # heartbeat_interval: 心跳发送间隔，应为 session_timeout 的 1/3
        heartbeat_interval_ms=int(os.getenv("KAFKA_HEARTBEAT_INTERVAL_MS", "20000")),
        # 处理慢时留足 poll 间隔，避免被误判死亡触发 rebalance（默认 10 分钟）
        max_poll_interval_ms=int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "600000")),
        # 单次请求超时（包括启动时的元数据获取和 JoinGroup），默认 40s 太短
        request_timeout_ms=int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "120000")),
        **sasl,
    )

    # 启动超时保护：避免 JoinGroup/Rebalance 时无限期卡住
    startup_timeout = int(os.getenv("KAFKA_STARTUP_TIMEOUT", "180"))  # 默认 3 分钟
    try:
        await asyncio.wait_for(consumer.start(), timeout=startup_timeout)
    except asyncio.TimeoutError:
        logger.error(
            f"Kafka consumer 启动超时（>{startup_timeout}s），可能原因：\n"
            f"  1. Broker 响应慢或不可达\n"
            f"  2. 消费者数量 > 分区数，导致 rebalance 无法完成\n"
            f"  3. 其他消费者频繁 rebalance\n"
            f"当前 topic={KAFKA_INPUT_TOPIC}，请检查 Kafka broker 状态和分区配置"
        )
        raise

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
