# Dispatcher and Inference Service Separation

## Understanding summary

- `NexusAi` remains the GPU inference service and owns model, Triton, EOS, VLM, and Milvus code.
- `NexusAi-Dispatcher` is a separate sibling Git project and owns Kafka consumption, result production, retry, and offset commits.
- Worker replica changes must not change Kafka group membership or partition count.
- Dispatcher instances run active-active in one consumer group.
- The projects communicate only through HTTP and a stable task ID.
- Delivery is at-least-once; downstream consumers deduplicate results by `taskId`.
- No registration database, leader election, shared source package, or exactly-once Milvus layer is introduced.
- The input topic has 40 partitions at about 100 messages/minute each; 28 workers process about 150-300 messages/minute each.
- Processing stays serial inside each partition and concurrent across partitions.

## Assumptions

- Production exposes inference workers through a Kubernetes Service or equivalent ready-endpoint load balancer.
- Two Dispatchers are sufficient initially; Kafka partitions define their parallelism ceiling.
- Worker HTTP timeout is longer than the configured inference-stage timeouts.
- `WORKER_AUTH_TOKEN` is supplied as a secret when traffic crosses an untrusted network boundary.
- The two projects are built, released, and operated independently.
- Production sends requests through one ready-endpoint Service; Compose may use explicit worker URLs.
- Capacity responses (`429`/`503`) are not infrastructure failures and do not open the circuit breaker.

## Final design

```text
Kafka input
    |
    +--> NexusAi-Dispatcher (2+ active instances)
              |  POST /process, X-Task-ID=topic:partition:offset
              v
         NexusAi inference Service (dynamic GPU workers)
              |
              +--> JSON inference result
              v
         Kafka result -> commit input offset
```

Each Dispatcher pauses a partition while processing its fetched batch and handles that batch sequentially. It publishes the worker result before committing the corresponding input offset. A worker or result-publish failure seeks back to the failed offset; a rebalance cancels stale partition work. This prevents silent loss while avoiding a custom offset-frontier tracker.

The Dispatcher does not probe for an idle worker. It sends each request through the platform Service. A ready worker accepts the request within its bounded semaphore; a full worker immediately returns `429` or `503`, and the Dispatcher retries through a new connection so the Service can choose another ready endpoint. Capacity responses retry immediately and never count toward the circuit breaker. Connection failures, timeouts, and non-capacity `5xx` responses use exponential backoff and open a short circuit after repeated failures.

Kafka remains the only durable queue. There is no 1000-item application queue: while one record from a partition is in flight, that partition is paused; it resumes only after result publication and offset commit, or after a failed attempt is rewound. This is the backpressure mechanism.

On shutdown a worker first marks readiness false, waits briefly for endpoint propagation, rejects new work, drains in-flight requests up to a configured timeout, and then exits. Unexpected crashes rely on readiness removal, request timeout, retry, and circuit breaking.

At the measured low-end capacity, 28 workers provide about 4,200 messages/minute for an input rate near 4,000 messages/minute. That is only 5% headroom, so normal operation should target at least 20-30% spare capacity; resilience mechanisms cannot replace missing inference capacity.

## Decision log

1. **Separate sibling Git projects.** Rejected a monorepo split because the requirement is independent projects and releases.
2. **HTTP boundary with raw Kafka payload.** Rejected base64 wrappers and shared Python models; JSON bytes already form the contract.
3. **Platform Service discovery.** Rejected a custom worker registry, heartbeat service, and least-connections scheduler.
4. **Per-partition serial processing.** Rejected a concurrent offset tracker because serial batches are smaller and preserve ordering naturally.
5. **At-least-once delivery.** Rejected a transactional/outbox layer for this change; duplicates carry deterministic `taskId`.
6. **One inference image and one dispatcher image.** Rejected a combined image because it would preserve build and release coupling.
7. **Service balancing, not idle-worker registration.** Rejected a custom load registry because multiple active Dispatchers would hold inconsistent local views. Worker concurrency limits and immediate capacity rejection provide the load signal.
8. **Capacity retry differs from fault retry.** `429`/capacity `503` switches endpoint immediately; network failures, timeouts, and other `5xx` responses back off and contribute to the circuit breaker.
9. **Kafka-native backpressure.** Rejected a large in-memory queue; pausing the active partition preserves durability and bounds Dispatcher memory.
10. **Graceful readiness drain.** Planned shutdown removes the worker from ready endpoints before waiting for in-flight inference; crashes continue through timeout and retry.

## Known limit

If a worker finishes a `vector=true` task but its HTTP response is lost, retrying can repeat Milvus and local-image side effects. Result-topic duplicates are identifiable by `taskId`; exactly-once vector writes require a separate schema-level idempotency change.
