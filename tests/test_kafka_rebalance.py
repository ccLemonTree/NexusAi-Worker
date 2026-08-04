import asyncio

import pytest

from kafka.rebalance import RebalanceEpoch, drain_tasks, skip_startup_backlog


class FakeConsumer:
    def __init__(self, positions, commit_error=None):
        self.positions = positions
        self.commit_error = commit_error
        self.seek_calls = []
        self.commit_calls = []

    async def seek_to_end(self, *partitions):
        self.seek_calls.append(partitions)

    async def position(self, partition):
        return self.positions[partition]

    async def commit(self, offsets):
        self.commit_calls.append(offsets)
        if self.commit_error is not None:
            raise self.commit_error


def test_rebalance_epoch_rejects_stale_tasks():
    state = RebalanceEpoch()
    state.finish_assignment()
    task_epoch = state.capture()

    assert state.can_commit(task_epoch)

    state.begin_rebalance()
    assert not state.can_commit(task_epoch)

    state.finish_assignment()
    assert not state.can_commit(task_epoch)
    assert state.can_commit(state.capture())


def test_skip_startup_backlog_seeks_and_commits_once():
    async def scenario():
        partitions = ("model_analyse-0", "model_analyse-1")
        consumer = FakeConsumer(
            {
                "model_analyse-0": 101,
                "model_analyse-1": 205,
            }
        )
        state = RebalanceEpoch()

        first_offsets = await skip_startup_backlog(
            consumer, partitions, state, enabled=True
        )
        second_offsets = await skip_startup_backlog(
            consumer, partitions, state, enabled=True
        )

        assert first_offsets == {
            "model_analyse-0": 101,
            "model_analyse-1": 205,
        }
        assert second_offsets == {}
        assert consumer.seek_calls == [partitions]
        assert consumer.commit_calls == [first_offsets]
        assert not state.startup_skip_pending

    asyncio.run(scenario())


def test_skip_startup_backlog_remains_pending_when_commit_fails():
    async def scenario():
        consumer = FakeConsumer(
            {"model_analyse-0": 101}, commit_error=RuntimeError("commit failed")
        )
        state = RebalanceEpoch()

        with pytest.raises(RuntimeError, match="commit failed"):
            await skip_startup_backlog(
                consumer, ("model_analyse-0",), state, enabled=True
            )

        assert state.startup_skip_pending
        assert state.rebalancing

    asyncio.run(scenario())


def test_skip_startup_backlog_remains_pending_if_epoch_changes_during_commit():
    async def scenario():
        state = RebalanceEpoch()
        consumer = FakeConsumer({"model_analyse-0": 101})
        original_commit = consumer.commit

        async def commit_then_rebalance(offsets):
            await original_commit(offsets)
            state.begin_rebalance()

        consumer.commit = commit_then_rebalance

        await skip_startup_backlog(
            consumer, ("model_analyse-0",), state, enabled=True
        )

        assert state.startup_skip_pending
        assert state.rebalancing

    asyncio.run(scenario())


def test_drain_tasks_returns_pending_without_cancelling_them():
    async def scenario():
        release = asyncio.Event()

        async def complete_now():
            return "done"

        async def wait_for_release():
            await release.wait()

        completed_task = asyncio.create_task(complete_now())
        blocked_task = asyncio.create_task(wait_for_release())

        done, pending = await drain_tasks(
            {completed_task, blocked_task}, timeout_seconds=0.01
        )

        assert completed_task in done
        assert blocked_task in pending
        assert not blocked_task.cancelled()

        release.set()
        await blocked_task

    asyncio.run(scenario())
