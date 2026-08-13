# Dispatcher and Inference Separation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move Kafka scheduling into an independent sibling project while leaving NexusAi as an inference-only service.

**Architecture:** `NexusAi-Dispatcher` consumes Kafka and calls the `NexusAi` HTTP worker Service. The projects have separate source trees, dependencies, Dockerfiles, Compose files, and Git histories.

**Tech Stack:** Python 3.12, asyncio, aiokafka, aiohttp, Docker Compose.

---

### Task 1: Create the standalone Dispatcher project

**Files:**
- Create: `NexusAi-Dispatcher/dispatcher_main.py`
- Create: `NexusAi-Dispatcher/dispatcher/{consumer,producer,worker_client}.py`
- Create: `NexusAi-Dispatcher/{requirements.txt,Dockerfile,docker-compose.yml,README.md}`

**Steps:**

1. Add the minimal Kafka-to-HTTP-to-Kafka flow.
2. Preserve per-partition order and publish-before-commit behavior.
3. Add one small runnable offset-order self-check without executing it.
4. Initialize an independent Git repository on `codex/dispatcher-worker-decoupling`.

### Task 2: Make NexusAi inference-only

**Files:**
- Create: `inference_server.py`
- Modify: `worker_main.py`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `docker-compose-stats.yml`
- Delete: `kafka/`
- Delete: `kafka_main.py`

**Steps:**

1. Move the existing `process_message` implementation to `inference/handler.py`.
2. Expose only `/health` and `/process` from the inference image.
3. Remove Kafka credentials, Dispatcher services, and Dispatcher commands from inference deployment files.
4. Preserve all unrelated user changes.

### Task 3: Build both projects

**Commands:**

```bash
docker build -t nexusai-inference:local F:/172.24.83.167/202607070928/NexusAi
docker build -t nexusai-dispatcher:local F:/172.24.83.167/202607070928/NexusAi-Dispatcher
```

**Expected:** Both commands exit with status 0. Tests, lint, formatting, and Compose validation remain unexecuted until separately requested by the user.
