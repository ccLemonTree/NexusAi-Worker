# Dispatcher 与推理服务拆分方案

## 需求理解

- `NexusAi` 继续作为 GPU 推理服务，负责模型、Triton、EOS、VLM 和 Milvus 相关代码。
- `NexusAi-Dispatcher` 是独立的同级 Git 项目，负责 Kafka 消费、结果发布、重试和 offset 提交。
- Worker 副本数量变化不得影响 Kafka 消费组成员或分区数量。
- 多个 Dispatcher 实例在同一个消费组中以 Active-Active 模式运行。
- 两个项目只通过 HTTP 和稳定的任务 ID 通信。
- 消息采用至少一次投递语义；下游消费者根据 `taskId` 对结果进行去重。
- 不引入注册数据库、主节点选举、共享源码包或 Milvus Exactly-Once 层。
- 输入主题有 40 个分区，每个分区约 100 条/分钟；28 个 Worker 各自约能处理 150～300 条/分钟。
- 同一分区内串行处理，不同分区之间并发处理。

## 前提假设

- 生产环境通过 Kubernetes Service 或功能等价的 Ready 端点负载均衡器暴露推理 Worker。
- 初期使用 2 个 Dispatcher 即可；Kafka 分区数决定其并行度上限。
- Worker HTTP 超时时间大于各推理阶段配置的超时时间。
- 流量跨越不可信网络边界时，通过 Secret 注入 `WORKER_AUTH_TOKEN`。
- 两个项目独立构建、发布和运维。
- 生产环境通过一个只包含 Ready 端点的 Service 发送请求；Compose 环境可以显式配置 Worker URL。
- 容量响应（`429`/`503`）不属于基础设施故障，不会触发熔断。

## 最终方案

```text
Kafka 输入
    |
    +--> NexusAi-Dispatcher (2+ active instances)
              |  POST /process, X-Task-ID=topic:partition:offset
              v
         NexusAi 推理 Service（动态 GPU Worker）
              |
              +--> JSON 推理结果
              v
         Kafka 结果 -> 提交输入 offset
```

每个 Dispatcher 在处理已拉取的批次时暂停对应分区，并按顺序处理该批消息。Dispatcher 先发布 Worker 返回的结果，再提交对应的输入 offset。Worker 请求或结果发布失败时，会 seek 回失败的 offset；发生 rebalance 时会取消过期的分区任务。这样既能避免消息静默丢失，也不需要自定义 offset 前沿跟踪器。

Dispatcher 不主动探测空闲 Worker，而是通过平台 Service 发送请求。处于 Ready 状态且信号量仍有容量的 Worker 会接收请求；容量已满的 Worker 立即返回 `429` 或 `503`，Dispatcher 使用新连接重试，使 Service 有机会选择其他 Ready 端点。容量响应会立即重试且不计入熔断器；连接失败、超时和非容量类 `5xx` 响应使用指数退避，并在连续失败后短暂熔断。

Kafka 仍是唯一的持久化队列。系统不设置容量为 1000 的应用内队列：某分区有消息正在处理时，该分区会暂停；只有在结果发布并提交 offset 后，或失败任务回退后，才恢复消费。这就是系统的背压机制。

Worker 关闭时会先将 readiness 标记为 false，短暂等待端点状态传播，拒绝新任务，并在配置的超时时间内等待在途请求完成后退出。异常崩溃则依靠 readiness 摘除、请求超时、重试和熔断处理。

按实测最低处理能力计算，28 个 Worker 每分钟约能处理 4200 条消息，而输入流量接近每分钟 4000 条，仅有 5% 余量。因此正常运行至少应保留 20%～30% 的空闲容量；可靠性机制不能弥补推理容量不足。

## 决策记录

1. **使用独立的同级 Git 项目。** 因需求明确要求项目和发布相互独立，所以不采用 monorepo 拆分。
2. **以原始 Kafka 消息作为 HTTP 边界。** 不使用 Base64 包装或共享 Python 模型；JSON 字节本身就是通信契约。
3. **使用平台 Service 发现。** 不采用自定义 Worker 注册中心、心跳服务和最少连接调度器。
4. **按分区串行处理。** 不使用并发 offset 跟踪器，因为串行批次更简单，并且天然保持顺序。
5. **至少一次投递。** 本次改动不引入事务或 Outbox 层；重复消息携带确定性的 `taskId`。
6. **推理和调度各使用一个镜像。** 不使用组合镜像，避免继续耦合构建与发布。
7. **使用 Service 负载均衡，不注册空闲 Worker。** 多个 Active Dispatcher 会形成不一致的本地负载视图，因此不使用自定义负载注册中心。Worker 并发限制和即时容量拒绝可以提供负载信号。
8. **容量重试与故障重试分开。** `429` 或容量类 `503` 会立即切换端点；网络故障、超时和其他 `5xx` 会退避，并计入熔断器。
9. **使用 Kafka 原生背压。** 不设置大型内存队列；暂停活跃分区既能保持持久性，也能限制 Dispatcher 的内存占用。
10. **基于 readiness 优雅摘流。** 正常关闭时先从 Ready 端点中移除 Worker，再等待在途推理完成；异常崩溃继续通过超时和重试处理。

## 已知限制

如果 Worker 完成了 `vector=true` 的任务，但 HTTP 响应丢失，重试可能会重复执行 Milvus 写入和本地图片落盘等副作用。结果主题中的重复消息可通过 `taskId` 识别；向量写入要实现 Exactly-Once，还需要单独进行数据库 Schema 层面的幂等改造。
