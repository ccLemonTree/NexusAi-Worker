# Kafka Restart Rebalance Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve the current 16-way worker concurrency while preventing stale-generation commits, skipping pre-start backlog when configured, and allowing Compose restarts to drain cleanly.

**Architecture:** Add a small rebalance lifecycle module that owns a local epoch, startup assignment initialization, and bounded task draining. Wire it into `AIOKafkaConsumer` through a rebalance listener; keep message processing and result production unchanged.

**Tech Stack:** Python 3.12, asyncio, aiokafka 0.11, pytest, Docker Compose.

---

### Task 1: Rebalance lifecycle tests

**Files:**
- Create: `tests/test_kafka_rebalance.py`
- Create: `kafka/rebalance.py`

**Step 1: Write failing tests**

Test that a captured epoch becomes invalid after revoke, assignment restores current-epoch commits, startup backlog skipping seeks and commits only once, initialization failure remains pending, and task draining is bounded.

**Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_kafka_rebalance.py -v`

Expected: collection fails because `kafka.rebalance` does not exist.

**Step 3: Implement the lifecycle module**

Add `RebalanceEpoch`, `skip_startup_backlog()`, `drain_tasks()`, and `WorkerRebalanceListener`. The listener invalidates old epochs before draining and only marks assignment ready after startup initialization succeeds.

**Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_kafka_rebalance.py -v`

Expected: all lifecycle tests pass.

### Task 2: Consumer integration

**Files:**
- Modify: `kafka/consumer.py`
- Test: `tests/test_kafka_rebalance.py`

**Step 1: Add failing integration-focused tests**

Test the commit decision used by `_safe_commit()` and listener state transitions with fake consumers/tasks.

**Step 2: Verify RED**

Run: `python -m pytest tests/test_kafka_rebalance.py -v`

Expected: stale commits are not yet rejected by the intended API.

**Step 3: Wire the minimal behavior**

- Instantiate the consumer without positional topics, then call `subscribe()` with `WorkerRebalanceListener`.
- Capture the current epoch for each task and reject stale/rebalancing commits before contacting Kafka.
- Track pending tasks by partition for bounded revoke draining.
- Replace the infinite iterator with `getmany(timeout_ms=1000)` so SIGTERM is noticed while idle.
- Drain for `KAFKA_SHUTDOWN_DRAIN_TIMEOUT`, cancel timed-out wrappers, then stop consumer and producer.

**Step 4: Verify GREEN and syntax**

Run: `python -m pytest tests/test_kafka_rebalance.py -v`

Run: `python -m compileall kafka`

Expected: tests pass and compilation exits 0.

### Task 3: Compose restart controls

**Files:**
- Modify: `docker-compose-stats.yml`

**Step 1: Add configuration**

Set `KAFKA_REBALANCE_DRAIN_TIMEOUT=30`, `KAFKA_SHUTDOWN_DRAIN_TIMEOUT=90`, `KAFKA_SKIP_BACKLOG_ON_START=true`, and `stop_grace_period: 2m` on workers.

**Step 2: Validate Compose structure**

Run: `docker compose -f docker-compose-stats.yml config`

Expected: rendered worker services contain all three variables and a 120-second stop grace period. If Docker is unavailable locally, parse YAML and validate the merged anchors structurally.

### Task 4: Regression verification

**Files:**
- Verify: `kafka/consumer.py`
- Verify: `kafka/rebalance.py`
- Verify: `tests/test_kafka_rebalance.py`
- Verify: `docker-compose-stats.yml`

**Step 1:** Run the focused tests and Python compilation again from a clean process.

**Step 2:** Run `git diff --check` and inspect the exact diff.

**Step 3:** Report behavior changes, configuration defaults, remaining dynamic-membership warning race, and the required one-worker-at-a-time rollout procedure. Do not deploy or restart production without separate authorization.
