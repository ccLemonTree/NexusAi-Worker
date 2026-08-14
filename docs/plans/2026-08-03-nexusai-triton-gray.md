# NexusAI Triton 灰度部署实施计划

> **给 Claude：** 实施时必须使用子技能 `superpowers:executing-plans`，逐项执行本计划。

**目标：** 部署一个包含 NexusAI 和 Triton 的隔离 `ai/nexusai-triton-gray` Pod，然后在不使用 Kafka 或 EOS 的情况下验证本地图片的小模型和 VLM 推理。

**架构：** 单副本 Deployment 管理一个包含两个容器的 Pod。NexusAI 通过 Pod 内部 gRPC 访问 Triton；Triton 从 `nfs-pvc-ai` 挂载现有 Kafka 模型仓库；不创建选择该灰度 Pod 的 Service。

**技术栈：** Kubernetes 1.25、YAML、Python 3.12、Triton Inference Server、T4 GPU、NFS PVC、Paramiko SSH。

---

### 任务 1：创建设计部署清单

**涉及文件：**
- 创建：`C:\Users\chen0\Documents\Codex\2026-07-18\ban\work\production\nexusai-triton-gray.yaml`
- 参考：`C:\Users\chen0\Documents\Codex\2026-07-18\ban\work\production\nexusai-triton-gray-design.md`

**步骤 1：** 在 `ai` 命名空间中定义单副本 Deployment，使用唯一标签 `app: nexusai-triton-gray`，且不创建 Service。

**步骤 2：** 为 NexusAI 配置已批准的镜像和环境；使用 `IfNotPresent` 以允许使用已通过 Docker 测试的节点缓存；用空闲循环覆盖 Kafka 入口；将 `nfs-pvc-ai` 挂载到 `/ai`；探测本地 8001 端口。

**步骤 3：** 为 Triton 配置已通过 Compose 测试的镜像、1 块 T4 GPU、HTTP/gRPC/metrics 端口、3Gi `/dev/shm`，并将 PVC 子路径 `chen/code/huggingface/aiseefor_model_kafka` 挂载到 `/models`。

将 NFS 仓库以只读方式挂载到 `/models-source`，并在 `/models` 创建 `emptyDir` 模型仓库。为 NFS 中名为 `yolov26det_fire_middle` 的模型创建 Pod 内部别名 `yolov26det_firemiddle`；不得修改共享 NFS 目录。

### 任务 2：在不修改集群的情况下验证

**涉及文件：**
- 测试：`C:\Users\chen0\Documents\Codex\2026-07-18\ban\work\production\nexusai-triton-gray.yaml`

**步骤 1：** 解析本地 YAML，确认其中包含 1 个 Deployment、2 个容器、1 个 GPU 请求，并且不包含 Service。

**步骤 2：** 验证 Harbor 访问权限；创建隔离的 `ai/aisf-regcred-gray` 镜像拉取 Secret，且不在清单中持久化凭据；然后在 `<SSH_HOST>:<SSH_PORT>` 上将清单传给 `kubectl apply --dry-run=client -f -`。

**步骤 3：** 运行 `kubectl apply --dry-run=server -f -`，要求退出码为 0。

### 任务 3：部署并等待就绪

**步骤 1：** 使用 `kubectl apply -f -` 应用清单。

**步骤 2：** 运行 `kubectl -n ai rollout status deployment/nexusai-triton-gray --timeout=30m`。

**步骤 3：** 检查 Pod 事件、容器就绪状态、GPU 请求、镜像和 PVC 挂载。如果发布失败，收集诊断信息后停止，不得修改现有工作负载。

### 任务 4：验证依赖与模型

**步骤 1：** 检查 Triton 日志和 `/v2/models` 仓库索引中是否存在加载失败。

**步骤 2：** 从 NexusAI 容器验证本地 Triton gRPC、`gme-lb.gme.svc.cluster.local:8000` 和 `my-release-2-milvus.milvus.svc.cluster.local:19530` 的连通性。

使用 `MILVUS_CLIENT=tcp://my-release-2-milvus.milvus.svc.cluster.local:19530`；已部署的 `pymilvus` 不接受 `grpc://`，而 `tcp://` 已通过同一 gRPC 端口和数据库验证。

**步骤 3：** 确认 `/app/example/14.jpeg` 和 `/app/tests/performance/stress_test.py` 存在。

### 任务 5：执行功能冒烟测试

**步骤 1：** 运行 `python tests/performance/stress_test.py --image /app/example/14.jpeg --count 1 --concurrent 1 --labels small`，检查处理器输出中是否存在错误。

**步骤 2：** 使用 `--labels all` 运行相同命令，以覆盖本地 VLM。

**步骤 3：** 保持 `vector=False`，不要向 Milvus 写入测试记录。

### 任务 6：验证隔离并交付

**步骤 1：** 确认 `ai/nexusai-gpu` 和 `triton/triton-server` 仍各有 5 个 Ready 副本。

**步骤 2：** 确认没有 Service 选择 `app=nexusai-triton-gray`。

**步骤 3：** 记录最终 Pod 状态、所在节点、镜像、测试结果和回滚命令 `kubectl -n ai delete deployment nexusai-triton-gray`。
