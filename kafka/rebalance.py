"""Kafka worker lifecycle helpers.

The helpers in this module intentionally do not depend on aiokafka so their
state transitions can be tested without starting a broker or event loop client.
"""

import asyncio
from collections.abc import Iterable
from typing import Any


class RebalanceEpoch:
    """Track whether work belongs to the consumer's current assignment."""

    def __init__(self) -> None:
        self._epoch = 0
        self.rebalancing = True
        self.startup_skip_pending = True

    def capture(self) -> int:
        return self._epoch

    def begin_rebalance(self) -> int:
        self._epoch += 1
        self.rebalancing = True
        return self._epoch

    def finish_assignment(self) -> None:
        self.rebalancing = False

    def complete_startup_skip(self) -> None:
        self.startup_skip_pending = False

    def can_commit(self, task_epoch: int) -> bool:
        return not self.rebalancing and task_epoch == self._epoch


async def skip_startup_backlog(
    consumer: Any,
    partitions: Iterable[Any],
    state: RebalanceEpoch,
    *,
    enabled: bool,
) -> dict[Any, int]:
    """Commit assigned partitions at their current end offsets exactly once.

    The startup flag is cleared only after the commit succeeds. A later normal
    rebalance therefore keeps committed progress instead of skipping messages.
    """

    if not state.startup_skip_pending:
        return {}

    if not enabled:
        state.complete_startup_skip()
        return {}

    assigned = tuple(partitions)
    if not assigned:
        return {}

    assignment_epoch = state.capture()
    await consumer.seek_to_end(*assigned)
    offsets = {
        partition: await consumer.position(partition) for partition in assigned
    }
    await consumer.commit(offsets)
    if state.rebalancing and state.capture() == assignment_epoch:
        state.complete_startup_skip()
    return offsets


async def drain_tasks(
    tasks: Iterable[asyncio.Task[Any]], *, timeout_seconds: float
) -> tuple[set[asyncio.Task[Any]], set[asyncio.Task[Any]]]:
    """Wait up to a deadline without cancelling unfinished work."""

    active = {task for task in tasks if not task.done()}
    if not active:
        return set(), set()

    done, pending = await asyncio.wait(active, timeout=timeout_seconds)
    return done, pending
