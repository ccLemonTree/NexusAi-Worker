# NexusAi Worker Project Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate the Kafka-independent Worker into its own repository while retaining the legacy HTTP service unchanged.

**Architecture:** `NexusAi` remains the legacy HTTP project. `NexusAi-Worker` is created from the existing Kafka-free worker branch, and Kafka orchestration remains only in `NexusAi-Dispatcher`.

**Tech Stack:** Python 3.12, FastAPI, Docker, Git.

---

### Task 1: Create the independent Worker repository

**Files:**
- Create: sibling repository `F:\172.24.83.167\202607070928\NexusAi-Worker`

**Step 1:** Clone `codex/dispatcher-worker-decoupling` without uncommitted files.

**Step 2:** Point its `origin` at the Worker repository, not the legacy repository.

### Task 2: Verify the Worker boundary

**Files:**
- Test: `tests/test_worker_project_boundary.py`

**Step 1:** Assert the Worker HTTP entry point exists and tracked application code has no Kafka client imports.

**Step 2:** Run the test, then compile Worker modules.

### Task 3: Publish separately

**Step 1:** Commit only the boundary test and split documentation.

**Step 2:** Push the Worker repository after its GitHub repository is available.
