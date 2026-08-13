# NexusAI Triton Gray Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy one isolated `ai/nexusai-triton-gray` Pod containing NexusAI and Triton, then validate local-image small-model and VLM inference without Kafka or EOS.

**Architecture:** A single-replica Deployment owns a two-container Pod. NexusAI reaches Triton over Pod-local gRPC, while Triton mounts the existing Kafka model repository from `nfs-pvc-ai`; no Service selects the gray Pod.

**Tech Stack:** Kubernetes 1.25, YAML, Python 3.12, Triton Inference Server, T4 GPU, NFS PVC, Paramiko SSH.

---

### Task 1: Create the deployment manifest

**Files:**
- Create: `C:\Users\chen0\Documents\Codex\2026-07-18\ban\work\production\nexusai-triton-gray.yaml`
- Reference: `C:\Users\chen0\Documents\Codex\2026-07-18\ban\work\production\nexusai-triton-gray-design.md`

**Step 1:** Define a one-replica Deployment in namespace `ai` with unique label `app: nexusai-triton-gray` and no Service.

**Step 2:** Configure NexusAI with the approved image and environment, use `IfNotPresent` to permit the Docker-tested node cache, override the Kafka entry point with an idle loop, mount `nfs-pvc-ai` at `/ai`, and probe local port 8001.

**Step 3:** Configure Triton with the Compose-tested image, one T4 GPU, HTTP/gRPC/metrics ports, 3Gi `/dev/shm`, and PVC subpath `chen/code/huggingface/aiseefor_model_kafka` at `/models`.

Mount the NFS repository read-only at `/models-source` and build an `emptyDir` repository at `/models`. Create a Pod-local `yolov26det_firemiddle` alias for the NFS model named `yolov26det_fire_middle`; never modify the shared NFS directory.

### Task 2: Validate without changing the cluster

**Files:**
- Test: `C:\Users\chen0\Documents\Codex\2026-07-18\ban\work\production\nexusai-triton-gray.yaml`

**Step 1:** Parse the local YAML and assert it contains one Deployment, two containers, one GPU request, and no Service.

**Step 2:** Verify Harbor access, create the isolated `ai/aisf-regcred-gray` pull Secret without persisting credentials in the manifest, then send the manifest to `kubectl apply --dry-run=client -f -` on `36.138.227.241:3022`.

**Step 3:** Run `kubectl apply --dry-run=server -f -` and require exit code 0.

### Task 3: Deploy and wait for readiness

**Step 1:** Apply with `kubectl apply -f -`.

**Step 2:** Run `kubectl -n ai rollout status deployment/nexusai-triton-gray --timeout=30m`.

**Step 3:** Inspect Pod events, container readiness, GPU requests, images, and PVC mounts. If rollout fails, collect diagnostics and stop without changing existing workloads.

### Task 4: Verify dependencies and models

**Step 1:** Check Triton logs and `/v2/models` repository index for load failures.

**Step 2:** From the NexusAI container, verify local Triton gRPC, `gme-lb.gme.svc.cluster.local:8000`, and `my-release-2-milvus.milvus.svc.cluster.local:19530` connectivity.

Use `MILVUS_CLIENT=tcp://my-release-2-milvus.milvus.svc.cluster.local:19530`; the deployed `pymilvus` rejects `grpc://`, while `tcp://` was verified against the same gRPC port and database.

**Step 3:** Confirm `/app/example/14.jpeg` and `/app/tests/performance/stress_test.py` exist.

### Task 5: Execute functional smoke tests

**Step 1:** Run `python tests/performance/stress_test.py --image /app/example/14.jpeg --count 1 --concurrent 1 --labels small` and inspect handler output for errors.

**Step 2:** Run the same command with `--labels all` to cover the local VLM.

**Step 3:** Keep `vector=False`; do not write test records to Milvus.

### Task 6: Verify isolation and hand off

**Step 1:** Confirm `ai/nexusai-gpu` and `triton/triton-server` remain at 5 ready replicas.

**Step 2:** Confirm no Service selects `app=nexusai-triton-gray`.

**Step 3:** Record final Pod status, node, images, test results, and rollback command `kubectl -n ai delete deployment nexusai-triton-gray`.
