from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections import defaultdict
from typing import Callable, Set

from aiokafka import AIOKafkaConsumer
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.errors import (
    CommitFailedError,
    IllegalGenerationError,
    IllegalStateError,
    UnknownMemberIdError,
)
from aiokafka.structs import TopicPartition

from kafka.handlers import process_message
from kafka.producer import send_result, stop_producer
from kafka.rebalance import RebalanceEpoch, drain_tasks, skip_startup_backlog
from kafka.stats import get_stats_collector, start_stats_task, is_stats_enabled
from tools.logger_tools import Kafka_Consumer_logger as logger

KAFKA_BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP",      "192.168.1.115:9092")
KAFKA_INPUT_TOPIC   = os.getenv("KAFKA_INPUT_TOPIC",    "model_analyse")
KAFKA_RESULT_TOPIC  = os.getenv("KAFKA_RESULT_TOPIC",   "model_analyse_result")
KAFKA_GROUP_ID      = os.getenv("KAFKA_CONSUMER_GROUP", "nexusai-model-worker")
MAX_CONCURRENT      = int(os.getenv("KAFKA_MAX_CONCURRENT", "16"))
REBALANCE_DRAIN_TIMEOUT = float(os.getenv("KAFKA_REBALANCE_DRAIN_TIMEOUT", "30"))
SHUTDOWN_DRAIN_TIMEOUT = float(os.getenv("KAFKA_SHUTDOWN_DRAIN_TIMEOUT", "90"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SKIP_BACKLOG_ON_START = _env_bool("KAFKA_SKIP_BACKLOG_ON_START", True)


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


async def _safe_commit(
    consumer: AIOKafkaConsumer,
    tp: TopicPartition,
    offset: int,
    task_epoch: int,
    rebalance_state: RebalanceEpoch,
) -> None:
    """
    提交 offset，容忍 rebalance 导致的各种失效场景：
    - 分区已不在本消费者的 assignment 中（IllegalStateError）
    - 提交时组已 rebalance / generation 过期 / member 失效
      （CommitFailedError / IllegalGenerationError / UnknownMemberIdError）
    这些都是多消费者 rebalance 的正常现象：该 offset 会由新 owner 重新消费。
    统一降级为 warning，不刷错误堆栈。
    """
    # rebalance 一开始就让旧任务失效，避免它们使用过期 generation 提交。
    if not rebalance_state.can_commit(task_epoch):
        logger.warning(
            f"跳过 commit：任务属于旧 assignment，partition={tp.partition} offset={offset}"
        )
        return

    # 提交前再校验分区归属，挡掉 assignment 已更新但回调尚未完成的竞态。
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
    task_epoch: int,
    rebalance_state: RebalanceEpoch,
) -> None:
    """处理单条消息：推理 → 发送结果 → commit offset。"""
    tp = TopicPartition(msg.topic, msg.partition)
    _stats_enabled = is_stats_enabled()
    stats = get_stats_collector() if _stats_enabled else None

    try:
        # 记录消费
        if stats:
            stats.record_consumed()

        result = await process_message(msg.value)

        # 检查结果是否有错误
        if stats and result.get("error"):
            stats.record_error()

        # process_message 现在总是返回结果（包括错误情况）
        await send_result(result)

        # 成功：提交该分区的下一个 offset
        await _safe_commit(
            consumer, tp, msg.offset + 1, task_epoch, rebalance_state
        )
    except Exception as e:
        logger.error(
            f"消息处理失败 topic={msg.topic} partition={msg.partition} offset={msg.offset}: {e}",
            exc_info=True,
        )
        # 失败：仍然 commit，跳过毒丸消息，防止队列卡死
        await _safe_commit(
            consumer, tp, msg.offset + 1, task_epoch, rebalance_state
        )
    finally:
        # 无论成功失败都统计生产（记录的是"尝试处理"的消息数）
        if stats:
            stats.record_produced()
        semaphore.release()


class WorkerRebalanceListener(ConsumerRebalanceListener):
    """Invalidate stale work before revoke and initialize a new assignment."""

    def __init__(
        self,
        consumer: AIOKafkaConsumer,
        rebalance_state: RebalanceEpoch,
        tasks_for_partitions: Callable[[set[TopicPartition]], Set[asyncio.Task]],
    ) -> None:
        self.consumer = consumer
        self.rebalance_state = rebalance_state
        self.tasks_for_partitions = tasks_for_partitions
        self._assignment_lock = asyncio.Lock()

    async def on_partitions_revoked(self, revoked: set[TopicPartition]) -> None:
        epoch = self.rebalance_state.begin_rebalance()
        tasks = self.tasks_for_partitions(revoked)
        logger.info(
            f"Kafka partitions revoked | epoch={epoch} | "
            f"partitions={sorted(tp.partition for tp in revoked)} | "
            f"in_flight={len(tasks)}"
        )
        _, pending = await drain_tasks(
            tasks, timeout_seconds=REBALANCE_DRAIN_TIMEOUT
        )
        if pending:
            logger.warning(
                f"Rebalance drain 超时（{REBALANCE_DRAIN_TIMEOUT}s），"
                f"保留 {len(pending)} 个任务继续执行但禁止其 commit"
            )

    async def on_partitions_assigned(self, assigned: set[TopicPartition]) -> None:
        async with self._assignment_lock:
            try:
                offsets = await skip_startup_backlog(
                    self.consumer,
                    assigned,
                    self.rebalance_state,
                    enabled=SKIP_BACKLOG_ON_START,
                )
            except Exception as error:
                # aiokafka 会吞掉 listener 异常。保持 rebalancing 状态，主循环重试，
                # 防止首次定位尚未成功时开始拉取消息。
                logger.warning(f"首次定位 Kafka 最新 offset 失败，将重试: {error}")
                return

            if self.rebalance_state.startup_skip_pending:
                return
            self._finish_assignment(assigned, offsets)

    def _finish_assignment(
        self, assigned: set[TopicPartition], offsets: dict[TopicPartition, int]
    ) -> None:
        self.rebalance_state.finish_assignment()
        startup_offsets = {tp.partition: offset for tp, offset in offsets.items()}
        logger.info(
            f"Kafka partitions assigned | "
            f"partitions={sorted(tp.partition for tp in assigned)} | "
            f"startup_offsets={startup_offsets}"
        )

    async def retry_startup_assignment(self) -> bool:
        """Retry initialization if aiokafka swallowed an assignment error."""
        async with self._assignment_lock:
            assigned = self.consumer.assignment()
            if not assigned or not self.rebalance_state.startup_skip_pending:
                return False
            try:
                offsets = await skip_startup_backlog(
                    self.consumer,
                    assigned,
                    self.rebalance_state,
                    enabled=SKIP_BACKLOG_ON_START,
                )
            except Exception as error:
                logger.warning(f"首次定位 Kafka 最新 offset 重试失败: {error}")
                return False
            if self.rebalance_state.startup_skip_pending:
                return False
            self._finish_assignment(assigned, offsets)
            return True


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
    pending_by_partition: dict[TopicPartition, Set[asyncio.Task]] = defaultdict(set)
    rebalance_state = RebalanceEpoch()

    sasl = _sasl_kwargs()
    logger.info(f"Kafka SASL: {'enabled  mechanism=' + sasl['sasl_mechanism'] if sasl else 'disabled'}")

    consumer = AIOKafkaConsumer(
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

    def tasks_for_partitions(partitions: set[TopicPartition]) -> Set[asyncio.Task]:
        return {
            task
            for partition in partitions
            for task in pending_by_partition.get(partition, set())
            if not task.done()
        }

    listener = WorkerRebalanceListener(
        consumer, rebalance_state, tasks_for_partitions
    )
    consumer.subscribe([KAFKA_INPUT_TOPIC], listener=listener)

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

    # 输出分区分配情况，用于诊断负载不均衡
    assigned_partitions = consumer.assignment()
    partition_ids = sorted([tp.partition for tp in assigned_partitions])
    logger.info(
        f"Kafka consumer started | topic={KAFKA_INPUT_TOPIC} | "
        f"group={KAFKA_GROUP_ID} | max_concurrent={MAX_CONCURRENT} | "
        f"assigned_partitions={partition_ids} (total={len(partition_ids)})"
    )

    try:
        while not _shutdown_event.is_set():
            if rebalance_state.rebalancing:
                if rebalance_state.startup_skip_pending:
                    await listener.retry_startup_assignment()
                await asyncio.sleep(0.1)
                continue

            batches = await consumer.getmany(
                timeout_ms=1000, max_records=MAX_CONCURRENT
            )
            for tp, messages in batches.items():
                for msg in messages:
                    if _shutdown_event.is_set() or rebalance_state.rebalancing:
                        break

                    # 占用一个并发槽，超出上限时等待；等待期间可能发生 rebalance。
                    acquired = False
                    while (
                        not _shutdown_event.is_set()
                        and not rebalance_state.rebalancing
                    ):
                        try:
                            await asyncio.wait_for(semaphore.acquire(), timeout=0.5)
                            acquired = True
                            break
                        except asyncio.TimeoutError:
                            continue
                    if _shutdown_event.is_set() or rebalance_state.rebalancing:
                        if acquired:
                            semaphore.release()
                        break

                    task_epoch = rebalance_state.capture()
                    task = asyncio.create_task(
                        _handle_one(
                            msg,
                            consumer,
                            semaphore,
                            task_epoch,
                            rebalance_state,
                        ),
                        name=f"msg-{msg.partition}-{msg.offset}",
                    )
                    pending.add(task)
                    pending_by_partition[tp].add(task)

                    def remove_task(
                        completed: asyncio.Task, partition: TopicPartition = tp
                    ) -> None:
                        pending.discard(completed)
                        partition_tasks = pending_by_partition.get(partition)
                        if partition_tasks is not None:
                            partition_tasks.discard(completed)
                            if not partition_tasks:
                                pending_by_partition.pop(partition, None)

                    task.add_done_callback(remove_task)

        logger.info("Shutdown signal received, stopping consumer loop")

    finally:
        # 等待所有 in-flight 任务；超过上限时取消，未提交消息由 Kafka 接管。
        if pending:
            logger.info(f"Waiting for {len(pending)} in-flight tasks to finish...")
            _, unfinished = await drain_tasks(
                pending, timeout_seconds=SHUTDOWN_DRAIN_TIMEOUT
            )
            if unfinished:
                logger.warning(
                    f"Shutdown drain 超时（{SHUTDOWN_DRAIN_TIMEOUT}s），"
                    f"取消 {len(unfinished)} 个未完成任务"
                )
                for task in unfinished:
                    task.cancel()
                await asyncio.gather(*unfinished, return_exceptions=True)

        await consumer.stop()
        await stop_producer()
        logger.info("Kafka consumer cleanly stopped")
