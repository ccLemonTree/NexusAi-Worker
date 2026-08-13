# Worker Resilience Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use test-driven-development and implement task-by-task.

**Goal:** Add capacity-aware retry, automatic circuit breaking, readiness draining, and graceful worker shutdown without changing the established Kafka ownership or per-partition ordering.

**Architecture:** `NexusAi-Dispatcher` remains the only Kafka consumer and result producer. It sends inference requests through a Service, immediately retries capacity responses on a new connection, backs off infrastructure failures, and temporarily opens failed targets. `NexusAi` remains stateless and exposes liveness/readiness while draining in-flight HTTP work on shutdown.

**Tech Stack:** Python 3.12, asyncio, aiohttp, aiokafka, unittest, Docker Compose.

---

### Task 1: Dispatcher capacity retry and circuit breaker

**Files:**
- Create: `../NexusAi-Dispatcher/tests/test_worker_client.py`
- Modify: `../NexusAi-Dispatcher/dispatcher/worker_client.py`
- Modify: `../NexusAi-Dispatcher/.env.example`
- Modify: `../NexusAi-Dispatcher/docker-compose.yml`

1. Add an async test proving `429`/capacity `503` retries immediately and does not increment infrastructure failure state.
2. Add an async test proving connection failures open a target after the configured threshold, skip it during cooldown, and allow one half-open probe after cooldown.
3. Run only `python -m unittest tests.test_worker_client -v` and confirm the new tests fail for missing behavior.
4. Implement the smallest per-target circuit state using monotonic time; keep normal successful connections reusable and close failed/capacity responses before retry.
5. Run the same focused test and confirm it passes.

### Task 2: Preserve Kafka ordering and backpressure

**Files:**
- Modify: `../NexusAi-Dispatcher/tests/test_consumer.py`
- Modify only if required: `../NexusAi-Dispatcher/dispatcher/consumer.py`

1. Add a test for the established order: worker result, Kafka result acknowledgement, then input offset commit.
2. Add a test that a failure rewinds the current offset and never advances the partition.
3. Keep one in-flight task per partition; do not add an application queue or concurrent offset frontier.
4. Run only the focused consumer tests after explicit user authorization.

### Task 3: Worker readiness and graceful drain

**Files:**
- Create: `tests/test_inference_server.py`
- Modify: `inference_server.py`
- Modify: `worker_main.py`
- Modify: `docker-compose.yml`

1. Add a test proving `/ready` is 200 while accepting traffic and 503 after drain begins.
2. Add a test proving draining rejects new `/process` requests while an already accepted request can finish.
3. Run only `python -m unittest tests.test_inference_server -v` and confirm the new tests fail for missing behavior.
4. Track accepting/in-flight state in the aiohttp application. On SIGTERM, set readiness false, wait a short propagation delay, then use aiohttp cleanup with a bounded shutdown timeout.
5. Change the deployment readiness check from `/health` to `/ready`; keep `/health` as liveness.
6. Run the same focused test after explicit user authorization.

### Task 4: Documentation and release

**Files:**
- Modify: `../NexusAi-Dispatcher/README.md`
- Modify: `docs/designs/2026-08-13-dispatcher-inference-separation.md`

1. Document normal routing, immediate capacity retry, infrastructure backoff, breaker thresholds, readiness drain, and the 20-30% capacity-headroom recommendation.
2. Record that Service balancing is not exact idle-worker selection and that `taskId` deduplication remains mandatory.
3. Inspect staged diffs, then commit and push each repository only after focused verification is explicitly authorized and completed.
