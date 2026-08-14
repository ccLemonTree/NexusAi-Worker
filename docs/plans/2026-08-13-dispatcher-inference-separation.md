# Dispatcher 与推理服务拆分实施计划

> **给 Claude：** 实施时必须使用子技能 `superpowers:executing-plans`，逐项执行本计划。

**目标：** 将 Kafka 调度迁移到独立的同级项目，使 NexusAi 只保留推理服务职责。

**架构：** `NexusAi-Dispatcher` 消费 Kafka 并调用 `NexusAi` HTTP Worker Service。两个项目分别维护源码树、依赖、Dockerfile、Compose 文件和 Git 历史。

**技术栈：** Python 3.12、asyncio、aiokafka、aiohttp、Docker Compose。

---

### 任务 1：创建独立的 Dispatcher 项目

**涉及文件：**
- 创建：`NexusAi-Dispatcher/dispatcher_main.py`
- 创建：`NexusAi-Dispatcher/dispatcher/{consumer,producer,worker_client}.py`
- 创建：`NexusAi-Dispatcher/{requirements.txt,Dockerfile,docker-compose.yml,README.md}`

**实施步骤：**

1. 实现最小化的 Kafka → HTTP → Kafka 流程。
2. 保持分区内顺序，并确保先发布结果，再提交 offset。
3. 添加一个可运行的小型 offset 顺序自检，但暂不执行。
4. 在 `codex/dispatcher-worker-decoupling` 分支初始化独立 Git 仓库。

### 任务 2：让 NexusAi 只负责推理

**涉及文件：**
- 创建：`inference_server.py`
- 修改：`worker_main.py`
- 修改：`Dockerfile`
- 修改：`docker-compose.yml`
- 修改：`docker-compose-stats.yml`
- 删除：`kafka/`
- 删除：`kafka_main.py`

**实施步骤：**

1. 将现有 `process_message` 实现迁移到 `inference/handler.py`。
2. 推理镜像只提供 `/health` 和 `/process` 接口。
3. 从推理部署文件中移除 Kafka 凭据、Dispatcher 服务和 Dispatcher 命令。
4. 保留所有无关的用户改动。

### 任务 3：构建两个项目

**命令：**

```bash
docker build -t nexusai-inference:local F:/172.24.83.167/202607070928/NexusAi
docker build -t nexusai-dispatcher:local F:/172.24.83.167/202607070928/NexusAi-Dispatcher
```

**预期结果：** 两条命令都以状态码 0 退出。在用户单独要求前，不执行测试、lint、格式化和 Compose 校验。
