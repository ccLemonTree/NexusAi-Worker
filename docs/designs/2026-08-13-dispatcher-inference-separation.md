# Dispatcher and Inference Service Separation

## Understanding summary

- `NexusAi` remains the GPU inference service and owns model, Triton, EOS, VLM, and Milvus code.
- `NexusAi-Dispatcher` is a separate sibling Git project and owns Kafka consumption, result production, retry, and offset commits.
- Worker replica changes must not change Kafka group membership or partition count.
- Dispatcher instances run active-active in one consumer group.
- The projects communicate only through HTTP and a stable task ID.
- Delivery is at-least-once; downstream consumers deduplicate results by `taskId`.
- No registration database, leader election, shared source package, or exactly-once Milvus layer is introduced.

## Assumptions

- Production exposes inference workers through a Kubernetes Service or equivalent ready-endpoint load balancer.
- Two Dispatchers are sufficient initially; Kafka partitions define their parallelism ceiling.
- Worker HTTP timeout is longer than the configured inference-stage timeouts.
- `WORKER_AUTH_TOKEN` is supplied as a secret when traffic crosses an untrusted network boundary.
- The two projects are built, released, and operated independently.

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

## Decision log

1. **Separate sibling Git projects.** Rejected a monorepo split because the requirement is independent projects and releases.
2. **HTTP boundary with raw Kafka payload.** Rejected base64 wrappers and shared Python models; JSON bytes already form the contract.
3. **Platform Service discovery.** Rejected a custom worker registry, heartbeat service, and least-connections scheduler.
4. **Per-partition serial processing.** Rejected a concurrent offset tracker because serial batches are smaller and preserve ordering naturally.
5. **At-least-once delivery.** Rejected a transactional/outbox layer for this change; duplicates carry deterministic `taskId`.
6. **One inference image and one dispatcher image.** Rejected a combined image because it would preserve build and release coupling.

## Known limit

If a worker finishes a `vector=true` task but its HTTP response is lost, retrying can repeat Milvus and local-image side effects. Result-topic duplicates are identifiable by `taskId`; exactly-once vector writes require a separate schema-level idempotency change.
